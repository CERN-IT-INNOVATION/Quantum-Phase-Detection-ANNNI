import json
import time
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from jax import value_and_grad

from .utils import get_khs_train


def train(
    model,
    num_epochs: int,
    lr: float,
    k: float = 0,
    h: float = 0,
    R: int = 0,
    path: str = "",
    showbar: bool = True,
):
    khs_train = get_khs_train(model, k, h, R)
    states = jnp.array(
        [model.states[model.index_map.index((k, h))] for (k, h) in khs_train]
    )

    def loss_fun(params):
        # Output expectation values of the qubits
        # Z [-1, 1]
        Z = jnp.array(model.vjq_circuit(params, states))
        # Zn [0, 1]
        Zn = (1 + Z) / 2
        loss_value = jnp.mean(Zn)

        return loss_value

    def _update(
        optimizer,
        state,
        params,
    ):
        loss_value, grads = value_and_grad(loss_fun)(params)
        updates, optimizer_state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss_value

    params = np.random.normal(loc=0, scale=1, size=model.num_params)

    params_history = [params.tolist()]
    loss_history = [float(loss_fun(params))]

    # Set the optimizer
    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)

    progress = tqdm.tqdm(range(1, num_epochs + 1), disable=not showbar)

    # Time start training
    t_train_start = time.time()

    for epoch in progress:
        params, optimizer_state, loss_value = _update(
            optimizer,
            optimizer_state,
            params,
        )

        progress.set_description(f"COST: {loss_value:.4f}")

        params_history.append(params.tolist())
        loss_history.append(float(loss_value))

    # Time ending training
    t_train_stop = time.time()

    if path:
        result = {
            # Model
            "num_qubits": model.num_qubits,
            "ansatz": model.ansatz_name,
            "side": model.side,
            "depth": model.depth,
            # Training
            "X_train": khs_train,
            "epochs": num_epochs,
            "lr": lr,
            "time": t_train_stop - t_train_start,
            "loss": float(loss_value),
        }

        Path(path).mkdir(parents=True, exist_ok=True)

        with open(Path(path) / "train.json", "w") as f:
            json.dump(result, f, indent=4)

        with h5py.File(Path(path) / "train.h5", "w") as f:
            f.create_dataset("params_history", data=np.array(params_history))
            f.create_dataset("loss_history", data=np.array(loss_history))

    return loss_value, params, loss_history, params_history
