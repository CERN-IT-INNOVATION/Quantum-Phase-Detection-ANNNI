import json
import time
from pathlib import Path

import h5py
import numpy as np
import optax
import tqdm
from jax import value_and_grad

from .loss import cross_entropy


def train(
    model,
    num_epochs: int,
    lr: float,
    T: float = 0.25,
    bs=None,
    path: str = "",
    showbar: bool = True,
):
    states_analitical = model.states[model.analytical_mask]
    labels_analitical = model.labels[model.analytical_mask]

    def _loss(params, states, labels):
        # Output expectation values of the qubits
        pred = model.vjq_circuit(params, states)
        loss_value = cross_entropy(pred, labels, T=T)

        return loss_value

    def _update(optimizer, optimizer_state, params, states, labels):
        ce_loss, grads = value_and_grad(_loss)(params, states, labels)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, ce_loss

    params = np.random.normal(loc=0, scale=1, size=model.num_params)

    loss_history = [float(_loss(params, states_analitical, labels_analitical))]
    params_history = [params]

    # Set the optimizer
    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)

    progress = tqdm.tqdm(range(1, num_epochs + 1), disable=not showbar)

    # Time start training
    t_train_start = time.time()

    for epoch in progress:
        if bs is not None:
            idx = np.random.choice(len(states_analitical), bs, replace=False)
            states_batch = states_analitical[idx]
            labels_batch = labels_analitical[idx]
        else:
            states_batch = states_analitical
            labels_batch = labels_analitical

        params, optimizer_state, loss_value = _update(
            optimizer, optimizer_state, params, states_batch, labels_batch
        )

        progress.set_description(f"LOSS: {loss_value:.4f}")

        loss_history.append(float(loss_value))
        params_history.append(params)

    # Time ending training
    t_train_stop = time.time()

    if path:
        result = {
            # Model
            "num_qubits": model.num_qubits,
            "ansatz": model.ansatz_name,
            "side": model.side,
            # Training
            "batch_size": bs,
            "epochs": num_epochs,
            "lr": lr,
            "time": t_train_stop - t_train_start,
            "loss": float(loss_value),
            "T": T,
        }

        Path(path).mkdir(parents=True, exist_ok=True)

        with open(Path(path) / "train.json", "w") as f:
            json.dump(result, f, indent=4)

        with h5py.File(Path(path) / "train.h5", "w") as f:
            f.create_dataset("params_history", data=np.array(params_history))
            f.create_dataset("loss_history", data=np.array(loss_history))

    return loss_value, params, loss_history, params_history
