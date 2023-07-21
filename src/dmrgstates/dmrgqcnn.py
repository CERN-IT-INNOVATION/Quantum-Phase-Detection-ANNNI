import general as qmlgen
import circuits
import numpy as np

import pennylane as qml 
from typing import List, Tuple
from numbers import Number

from jax import vmap, jit, value_and_grad
import optax 
import jax.numpy as jnp
import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def cross_entropy(params, X, Y, vqcirc):
    OUT = vqcirc(X, params)
    
    return - jnp.mean( jnp.multiply(Y, jnp.log2(OUT))  )
    
def get_indexes_axes(n_hs, n_kappas):
    x_indexes = np.arange(0, n_kappas*n_hs+1, n_hs)
    y_indexes = np.arange(0, n_hs,            1)

    return x_indexes, y_indexes

def get_labels(h, k):
    if k == 0:
        # Added this case because of 0 encountering in division
        if h <= 1:
            return 0, 0
        else:
            return 1, 1
    elif k > -.5:
        # Left side (yes it is the left side, the phase plot is flipped due to x axis being from 0 to - kappa_max)
        if h <= qmlgen.paraferro(-k):
            return 0, 0
        else:
            return 1, 1
    else:
        # Right side
        if h <= qmlgen.paraanti(-k):
            if h <= qmlgen.b1(-k):
                return 2, 2
            else:
                return 2, 3
        else:
            return 1, 1

class qcnn:
    def __init__(self, STATES, PARAMS):
        """
        This class requires:
            STATES: an array of L wavefunctions (L, 2**n_qubits)
            PARAMS: an array of the parameters of the ANNNI model relative to the STATES wavefunction
                    PARAMS = (L, 2)
                                 |
                                h and kappa
        """ 

        # number of initial qubits of the QCNN: 
        self.n_qubits = int(np.log2(len(STATES[0]))) # I am assuming every vector in states has the same length
        self.n_outputs = 2

        self.PARAMS = PARAMS 
        self.hs     = np.unique(self.PARAMS[:,1])
        self.kappas = np.unique(self.PARAMS[:,0])

        def qcnn_circuit(state, params):
            # Wires that are not measured (through pooling)
            active_wires = np.arange(self.n_qubits)

            qml.QubitStateVector(state, wires=range(self.n_qubits))

            # Visual Separation State||QCNN
            qml.Barrier()
            qml.Barrier()

            # Index of the parameter vector
            index = 0

            # Iterate Convolution+Pooling until we only have a single wires
            index = circuits.wall_gate(active_wires, qml.RY, params, index)
            circuits.wall_cgate_serial(active_wires, qml.CNOT)
            while len(active_wires) > self.n_outputs:  # Repeat until the number of active wires
                # (non measured) is equal to n_outputs
                # Convolute
                index = circuits.convolution(active_wires, params, index)
                # Measure wires and apply rotations based on the measurement
                index, active_wires = circuits.pooling(active_wires, qml.RX, params, index)

                qml.Barrier()

            circuits.wall_cgate_serial(active_wires, qml.CNOT)
            index = circuits.wall_gate(active_wires, qml.RY, params, index)

            # Return the number of parameters
            return index + 1, active_wires

        self.STATES = STATES 

        self.device = qml.device('default.qubit.jax', wires=self.n_qubits)
        
        self.circuit_fun  = qcnn_circuit
        dummystate = np.zeros(2**self.n_qubits)
        dummystate[0] = 1
        self.n_params, self.final_active_wires = self.circuit_fun(dummystate, np.zeros(10000)) 

        def circuit(x, p):
            self.circuit_fun(x, p)
            return qml.probs([int(k) for k in self.final_active_wires])

        self.q_circuit    = qml.QNode(circuit, self.device)
        self.v_q_circuit  = vmap(self.q_circuit, (0, None))
        self.jv_q_circuit = jit(self.v_q_circuit)
        
        # LABELS
        # -> Y3 : Labels for the 3 phases
        # -> Y4 : Labels for the 3 phases + 1 (floating)    
        Y3, Y4 = [], [] 
        for h, k in self.PARAMS:
            y3, y4 = get_labels(h, k)
            Y3.append(y3)
            Y4.append(y4)
            
        self.Y3, self.Y4 = np.array(Y3), np.array(Y4)
        Y3states = []
        for y3 in Y3:
            if y3 == 0:
                Y3states.append([0,0,0,1])
            elif y3 == 1:
                Y3states.append([0,0,1,0])
            elif y3 == 2:
                Y3states.append([0,1,0,0])
            elif y3 == 3:
                Y3states.append([1,0,0,0])
        self.Y3states = np.array(Y3states)

        # Mask for the margins: 
        self.margins = np.logical_or(self.PARAMS[:,0] == 0, self.PARAMS[:,1] == 0)

    def train3(self, epochs, lr, loss_fn, sigma = 5e-1):
        Xstates = jnp.array(self.STATES[self.margins])
        Ystates = jnp.array(self.Y3states[self.margins])

        optimizer = optax.adam(learning_rate=lr)
        params = np.random.normal(0, sigma, self.n_params)
        opt_state = optimizer.init(params)

        def optimizer_update(opt, opt_state, X, p, Y):
            loss, grads = value_and_grad(loss_fn)(p, X, Y, self.v_q_circuit)
            updates, opt_state = opt.update(grads, opt_state)
            p = optax.apply_updates(p, updates)
            return p, opt_state, loss
        
        # Update the optimizer
        update = jit(lambda opt_state, X, p, Y: optimizer_update(optimizer, opt_state, X, p, Y))

        loss_train : List[float] = []

        progress = tqdm.tqdm(range(epochs))
        for epoch in range(epochs):
            params, opt_state, train_loss = update(opt_state, Xstates, params, Ystates)

            loss_train.append(train_loss)
            progress.update(1)

        self.params3 = params
        return loss_train

    def show_prediction3(self):
        nx = len(self.kappas)
        ny = len(self.hs)

        predictions = np.argmax(self.jv_q_circuit(self.STATES, self.params3),1)

        plt.imshow(np.rot90(np.reshape(predictions, (nx,ny))))

        column = 0
        trans_lines = []
        for column in range(nx):
            y0 = self.Y3[ny*column]
            trans_point  = 0
            for idx, y in enumerate(self.Y3[ny*column:ny*column + ny]):
                if y != y0:
                    trans_point = idx
                    break

            trans_lines.append(trans_point)

        trans_lines = np.array(trans_lines)

        plt.plot(ny-trans_lines, color='red')

        return predictions
    
    def showclasses(self):
        nx    = len(self.kappas)
        nxmax = max(self.kappas)
        ny    = len(self.hs)
        nymax = max(-self.hs)

        fig = plt.figure(figsize=(15, 3))

        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                                nrows_ncols=(1,4),
                                axes_pad=0.15,
                                share_all=True,
                                cbar_location="right",
                                cbar_mode="single",
                                cbar_size="7%",
                                cbar_pad=0.15,
                                )
        
        PREDS = self.jv_q_circuit(self.STATES, self.params3)


        im = grid[0].imshow(np.rot90(np.reshape(PREDS[:,0], (nx, ny))), vmin=0, vmax=1, cmap="coolwarm")
        grid[0].set_ylabel(r'$h$')
        im = grid[1].imshow(np.rot90(np.reshape(PREDS[:,1], (nx, ny))), vmin=0, vmax=1, cmap="coolwarm")
        im = grid[2].imshow(np.rot90(np.reshape(PREDS[:,2], (nx, ny))), vmin=0, vmax=1, cmap="coolwarm")
        im = grid[3].imshow(np.rot90(np.reshape(PREDS[:,3], (nx, ny))), vmin=0, vmax=1, cmap="coolwarm")

        # Colorbar
        grid[1].cax.colorbar(im)
        grid[1].cax.toggle_label(True)

        ticks_x = [      nx/4 - .5, nx/2 - .5 , 3*nx/4 - .5, nx - .5]
        ticks_y = [-.5 , ny/4 - .5, ny/2 - .5 , 3*ny/4 - .5, ny - .5]

        # Set the titles:
        for i, g in enumerate(grid):
            g.set_title(f'Prob(class {i})')
            g.set_xlabel(r'$\kappa$')
            g.set_xticks(
            ticks= ticks_x,
            labels=[np.round(k * nxmax  / 4, 2) for k in range(1, 5)],
            )
            g.set_yticks(
                ticks=ticks_y,
                labels=[np.round(k * nymax / 4, 2) for k in range(4, -1, -1)],
            )

    def __repr__(self):
        reprstr  = f'Number of qubits : {self.n_qubits}\n'
        reprstr += f'Number of params : {self.n_params}\n\n'
        reprstr += 'Circuit:\n'
        reprstr += qml.draw(self.q_circuit)(np.zeros(2**self.n_qubits), np.arange(self.n_params))
        return reprstr