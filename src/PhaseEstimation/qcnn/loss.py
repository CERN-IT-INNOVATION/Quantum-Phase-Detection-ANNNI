import jax.numpy as jnp


def cross_entropy(pred, Y, T=0.5):
    """
    Computes the categorical cross-entropy loss with temperature scaling
    to encourage more extreme predictions.

    Parameters
    ----------
    pred : np.ndarray
        Array of the predicted probabilities.
    Y : np.ndarray
        one-hot encoded true labels.
    T : float
        temperature parameter (<1 makes the model more confident).

    Returns
    -------
    float
        Scalar value, the mean categorical cross-entropy loss.
    """
    epsilon = 1e-9  # Small value for numerical stability
    pred = jnp.clip(pred, epsilon, 1 - epsilon)  # Prevent log(0)

    # Apply sharpening (raise probabilities to the power of 1/T)
    pred_sharpened = pred ** (1 / T)
    pred_sharpened /= jnp.sum(pred_sharpened, axis=1, keepdims=True)  # Re-normalize

    loss = -jnp.sum(Y * jnp.log(pred_sharpened), axis=1)  # Compute cross-entropy
    return jnp.mean(loss)  # Return mean loss over batch


def accuracy(Ypred, Ytrue):
    accuracy = jnp.mean(jnp.argmax(Ypred, axis=1) == jnp.argmax(Ytrue, axis=1))
    return accuracy
