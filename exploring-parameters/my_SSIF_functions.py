"""Modules defining functions needed in SSIF project."""

from typing import Callable
import numpy as np
from numpy.typing import ArrayLike

def calculate_v_KC(v0: float, f0: float, mu_KC: float, mu_c: float, n_c: float, n_m: float,
                   d: float, L0: float, D0: float, x_KC_minus: ArrayLike):
    """
    Calculate kinetochore speed.

    Left kinetochore position is used as independent variable. Assumes interkinetochore velocity ca
    be neglected, i.e. `v_KC_plus - v_KC_minus = 0`, which results in `x_KC_plus - x_KC_minus = d`.
    
    Parameters
    ----------
        v0 : float
            Motor velocity without load.
        f0 : float
            Stall force.
        mu_KC : float
            Effective friction coefficient at the kinetochore.
        mu_c : float
            Friction coefficient of single crosslinking protein.
        n_c : float
            Linear density of crosslinking proteins.
        n_m : float
            Linear density of motor proteins.
        d : float
            Interkinetochore distance.
        L0 : float
            Spindle length.
        D0 : float
            Bridging microtubule overlap.
        x_KC_minus : array-like
            Position of left kinetochore.

    Model paramaters calculated from arguments:
    x_KC_plus  : Position of right kinetochore (x_KC_plus = x_KC_minus + d).
    L_plus     : Lentgh of parallel overlap between plus-side k-fiber and plus-side bMT.
    L_minus    : Lentgh of parallel overlap between minus-side k-fiber and minus-side bMT.
    D_plus     : Length of antiparallel overlap between plus-side k-fiber and minus-side bMT.
    D_minus    : Lentgh of antiparallel overlap between minus-side k-fiber and plus side bMT.

    Returns
    -------
    array-like
        Kinetochore velocity array. Calculated at points specified in `x_KC_minus`.
    """

    x_KC_plus = x_KC_minus + d
    L_plus = L0/2 - x_KC_plus
    L_minus = L0/2 + x_KC_minus
    D_plus = (D0/2 - x_KC_plus) * np.heaviside(D0/2 - x_KC_plus, 0)
    D_minus = (D0/2 + x_KC_minus) * np.heaviside(D0/2 + x_KC_minus, 0)

    Nc_plus = n_c * L_plus
    Nc_minus = n_c * L_minus
    Nm_plus = n_m * D_plus
    Nm_minus = n_m * D_minus
    gc_plus = Nc_plus * mu_c
    gc_minus = Nc_minus * mu_c
    gm_plus = Nm_plus * f0/v0
    gm_minus = Nm_minus * f0/v0
    alpha_plus = (gc_plus + gm_plus + mu_KC)**(-1)
    alpha_minus = (gc_minus + gm_minus + mu_KC)**(-1)

    fac1 = 0.5 * v0/mu_KC
    num1 = (gc_plus + gm_plus) * (1 - alpha_plus * (gc_plus + gm_plus))
    num2 = (gc_minus + gm_minus) * (1 - alpha_minus * (gc_minus + gm_minus))
    den1 = alpha_plus * (gc_plus + gm_plus)
    den2 = alpha_minus * (gc_minus + gm_minus)

    num = num1 - num2
    den = den1 + den2
    fac2 = num/den

    return fac1 * fac2


def finite_difference(fun: Callable, t: ArrayLike, x0: float, args: tuple = None) -> ArrayLike:
    """
    Integrate dx/dt = f(t, x) using finite difference method.

    Parameters
    ----------
    fun : callable
        Derivative of x at time t (dx/dt(t)). Called as fun(*args, x0).
    t : array-like
        Time points to calculate x in.
    x0 : float
        Initial condition (value of x at t = t[0]).
    *args : tuple, optional, default = `None`
        Optional parameters passed to fun.

    Returns
    -------
    array-like
        Values of `x` at times `t`.
    """
    x = np.zeros(t.size)
    x[0] = x0

    if args is not None:
        for i in range(1, t.size):
            x[i] = x[i-1] + fun(*args, x[i-1]) * (t[i] - t[i-1])

    else:
        for i in range(1, t.size):
            x[i] = x[i-1] + fun(x[i-1]) * (t[i] - t[i-1])

    return x


def calculate_v_kMT_plus(v0: float, f0: float, mu_KC: float, mu_c: float, n_c: float, n_m: float,
                          d: float, L0: float, D0: float, x_KC_plus: ArrayLike) -> ArrayLike:
    """
    Calculate right k-fibre flux velocity.
    
    Parameters
    ----------
        v0 : float
            Motor velocity without load.
        f0 : float
            Stall force.
        mu_KC : float
            Effective friction coefficient at the kinetochore.
        mu_c : float
            Friction coefficient of single crosslinking protein.
        n_c : float
            Linear density of crosslinking proteins.
        n_m : float
            Linear density of motor proteins.
        d : float
            Interkinetochore distance.
        L0 : float
            Spindle length.
        D0 : float
            Bridging microtubule overlap.
        x_KC_plus : array-like
            Position of right kinetochore.

    Returns
    -------
        array-like
            Right kinetochore fiber velocity for given right kinetochore position.
    """
    x_KC_minus = x_KC_plus - d
    L_plus = L0/2 - x_KC_plus
    D_plus = (D0/2 - x_KC_plus) * np.heaviside(D0/2 - x_KC_plus, 0)

    Nc_plus = n_c * L_plus
    Nm_plus = n_m * D_plus
    gc_plus = Nc_plus * mu_c
    gm_plus = Nm_plus * f0/v0
    alpha_plus = (gc_plus + gm_plus + mu_KC)**(-1)

    fac1 = mu_KC * calculate_v_KC(v0, f0, mu_KC, mu_c, n_c, n_m, d, L0, D0, x_KC_minus)
    fac2 = 0.5 * v0 * (gc_plus + gm_plus)

    return alpha_plus * (fac1 + fac2)


def calculate_v_kMT_minus(v0: float, f0: float, mu_KC: float, mu_c: float, n_c: float, n_m: float,
                          d: float, L0: float, D0: float, x_KC_minus: ArrayLike) -> ArrayLike:
    """
    Calculate left k-fibre flux velocity.
    
    Parameters
    ----------
        v0 : float
            Motor velocity without load.
        f0 : float
            Stall force.
        mu_KC : float
            Effective friction coefficient at the kinetochore.
        mu_c : float
            Friction coefficient of single crosslinking protein.
        n_c : float
            Linear density of crosslinking proteins.
        n_m : float
            Linear density of motor proteins.
        d : float
            Interkinetochore distance.
        L0 : float
            Spindle length.
        D0 : float
            Bridging microtubule overlap.
        x_KC_minus : array-like
            Position of left kinetochore.

    Returns
    -------
        array-like
            Left kinetochore fiber velocity for given left kinetochore position.
    """
    L_minus = L0/2 + x_KC_minus
    D_minus = (D0/2 + x_KC_minus) * np.heaviside(D0/2 + x_KC_minus, 0)

    Nc_minus = n_c * L_minus
    Nm_minus = n_m * D_minus
    gc_minus = Nc_minus * mu_c
    gm_minus = Nm_minus * f0/v0
    alpha_minus = (gc_minus + gm_minus + mu_KC)**(-1)

    fac1 = mu_KC * calculate_v_KC(v0, f0, mu_KC, mu_c, n_c, n_m, d, L0, D0, x_KC_minus)
    fac2 = 0.5 * v0 * (gc_minus + gm_minus)

    return alpha_minus * (fac1 - fac2)


def calculate_forward_derivative(t: ArrayLike, x: ArrayLike) -> ArrayLike:
    """Calculate forward derivative of x over t using finite differences.
    
    Parameters
    ----------
        t : array-like
            Time points to calculate velocity in.
        x : array-like
            Values of position over time.

    Returns
    -------
        v : array-like
            Forward derivative values at given times. `v[-1]` is set equal to `v[-2]` in order to
            keep the same dimensionality as time array.
    """
    v = np.zeros(t.size)

    for i in range(t.size-1):
        v[i] = (x[i+1] - x[i])/(t[i+1] - t[i])

    v[-1] = v[-2] + (v[-2] - v[-3])/(t[-2] - t[-3]) * (t[-1] - t[-2])

    return v
