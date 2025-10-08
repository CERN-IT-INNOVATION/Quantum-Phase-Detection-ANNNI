import pennylane.numpy as np


def paraanti(x: np.ndarray) -> np.ndarray:
    r"""
    Transition Line: Paramagnetic-Antiphase [0.5 < x < 1.0].
    The function calculates the transition line defined by:

        .. math::
            y(x) = 1.05 \sqrt{(x - 0.5)*(x - 0.1)}

    Parameters
    ----------
    x : np.ndarray
        Input value(s) where the transition line is calculated.

    Returns
    -------
    np.ndarray
        Output array representing the transition line values.
    """

    y = 1.05 * np.sqrt(np.maximum((x - 0.5) * (x - 0.1), 0))
    y[x < 0.5] = 0
    return y


def paraferro(x: np.ndarray) -> np.ndarray:
    r"""
    Transition Line: Paramagnetic-Ferromagnetic [0.0 < x < 0.5].

    The function calculates the transition line defined by:

    .. math::
        y(x) = \frac{1 - x}{x} \left(1 - \sqrt{\frac{1 - 3x + 4x^2}{1 - x}}\right)

    Parameters
    ----------
    x : np.ndarray
        Input value(s) where the transition line is calculated. Values greater than 0.5 are set to NaN.

    Returns
    -------
    np.ndarray
        Output array representing the transition line values.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_x = np.where(x == 0, np.nan, x)
        y = ((1 - safe_x) / safe_x) * (
            1 - np.sqrt((1 - 3 * safe_x + 4 * safe_x * safe_x) / (1 - safe_x))
        )
        y[np.isnan(y)] = 1  # Replace NaN with zero or another desired value
        y[safe_x > 0.5] = 0  # Set to 0 the phase transition line out of his domain
    return y


def b1(x: np.ndarray) -> np.ndarray:
    r"""
    Floating phase Transition Line [0.5 < x < 1.0].

    The function calculates the transition line defined by:

    .. math::
        y(x) = 1.05(x - 0.5)

    Parameters
    ----------
    x : np.ndarray
        Input value(s) where the transition line is calculated.

    Returns
    -------
    np.ndarray
        Output array representing the transition line values.
    """
    y = 1.05 * (x - 0.5)
    y[x < 0.5] = 0
    return y


def peshel_emery(x: np.ndarray, y_cap: float = 2) -> np.ndarray:
    r"""
    Peshel-Emery Line.

    The function calculates the line defined by:

    .. math::
        y(x) = \frac{1}{4x} - x

    Values greater than y_cap are capped at y_cap.

    Parameters
    ----------
    x : np.ndarray
        Input value(s) where the line is calculated.
    y_cap : float
        Cap value

    Returns
    -------
    np.ndarray
        Output array representing the line values.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        y = (1 / (4 * x)) - x
        y[y > y_cap] = y_cap  # Handle large results
        y[x > 0.5] = 0  # Set to 0 the phase transition line out of his domain
        y = np.where(np.isfinite(y), y, y_cap)  # Handle infinite results
    return y


def get_phase(k, h):
    # 0 -> Ferromagnetic
    # 1 -> Paramagnetic
    # 2 -> Antiphase

    # Left side, it can be either Ferromagnetic of paramagnetic
    if k <= 0.5:
        return 0 if h <= paraferro(np.array([k]))[0] else 1
    # Right side, it can be either Paramagnetic or Antiphase
    else:
        return 2 if h <= paraanti(np.array([k]))[0] else 1


def is_analytical(k, h):
    # If the point in the phase space lies in one of the two axes, the
    # point is indeed analytical, meaning that it is possible to
    # get its phase without any approximate methods
    return k == 0 or h == 0
