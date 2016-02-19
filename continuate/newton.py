# -*- coding: utf-8 -*-
""" Basic Newton methods using Krylov subspace method.

Options
--------
newton_tol : float
    Tolerrance of Newton step
newton_maxiter : int
    Max iteration number of Newton method
jacobi_alpha : float
    Infinitesimal for calculating Jacobian matrix
trusted_region : float
    Radius in model trusted region approach
hook_maxiter : int
    Max iteration number of hook step
hook_tol : int
    Relative tolerance of hook step iteration

Their default values are set in :py:data:`.default_options`
"""

import numpy as np
import scipy.sparse.linalg as linalg
from itertools import count as icount
from .misc import Logger, array_adapter
from . import krylov

default_options = {
    "newton_tol": 1e-7,
    "newton_maxiter": 100,
    "jacobi_alpha": 1e-7,
    "trusted_region": 1e-1,
    "hook_maxiter": 100,
    "hook_tol": 0.1,
}
""" default values of options

You can get these values through :py:func:`continuate.get_default_options`
"""


def Jacobi(func, x0, jacobi_alpha, fx=None, **opt):
    """
    Jacobi oprator :math:`DF(x0)`, where

    .. math::
        DF(x0)dx = (F(x0 + \\alpha dx / |dx|) - F(x0))/(\\alpha/|dx|)

    Parameters
    ----------
    func : numpy.array -> numpy.array
        A function to be differentiated
    x0 : numpy.array
        A position where the Jacobian is evaluated
    fx : numpy.array, optional
        func(x0)

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x: np.array([x[1]**2, x[0]**2])
    >>> x0 = np.array([1, 2])
    >>> J = Jacobi(f, x0, 1e-7)

    """
    if fx is None:
        fx = func(x0)

    def wrap(v):
        norm = np.linalg.norm(v)
        r = jacobi_alpha / norm
        return (func(x0 + r * v) - fx) / r

    return linalg.LinearOperator(
        (len(x0), len(x0)),
        matvec=wrap,
        dtype=x0.dtype
    )


class Hessian(object):
    """ Calculate the deviation from linear approximation """

    logger = Logger(__name__, "Hessian")

    def __init__(self, func, x0, jacobi_alpha, **opt):
        self.fx0 = func(x0)
        self.A = Jacobi(func, x0, fx=self.fx0, jacobi_alpha=1e-7)
        self.f = func
        self.x0 = x0
        self.alpha = jacobi_alpha

    def __call__(self, v):
        fxv = self.f(self.x0 + v)
        Av = self.A * v
        nrm = max(map(np.linalg.norm, [self.fx0, fxv, self.fx0 + Av]))
        return np.linalg.norm(fxv - (self.fx0 + Av)) / nrm

    def deviation(self, v):
        return self.f(self.x0 + v) - (self.fx0 + self.A * v)

    def trusted_region(self, v, eps, r0=None, p=2):
        """ Estimate the trusted region in which the deviation is smaller than `eps`.

        Parameters
        ------------
        eps : float
            Destination value of deviation
        p : float, optional (default=2)
            Iteration will end if the deviation is in `[eps/p, eps*p]`.

        Return
        -------
        r : float
            radius of the trusted region

        """
        if type(r0) is float:
            r = r0
        else:
            r = 100*self.alpha
        v = v / np.linalg.norm(v)
        p = max(p, 1.0/p)
        for c in icount():
            e = self(r*v)
            self.logger.info({
                "count": c,
                "deviation": e,
            })
            if (e > eps/p) and (e < eps*p):
                return r
            r = r * np.sqrt(eps / e)


@array_adapter
def newton_krylov(func, x0, newton_tol, newton_maxiter, **opt):
    """
    solve multi-dimensional equation :math:`F(x) = 0`
    using Newton-Krylov method.

    Parameters
    -----------
    func : numpy.array -> numpy.array
        :math:`F` in the problem :math:`F(x) = 0`
    x0 : numpy.array
        initial guess of the solution

    Returns
    --------
    x : numpy.array
        The solution :math:`x` satisfies :math:`F(x)=0`

    Examples
    ---------
    >>> import continuate
    >>> opt = continuate.get_default_options()
    >>> opt["newton_tol"] = 1e-7
    >>> f = lambda x : (x-1)**2
    >>> x0 = np.array([1.1, 0.9])
    >>> x = newton_krylov(f, x0, **opt)
    >>> np.allclose(f(x), np.zeros_like(x), atol=1e-7)
    True

    """
    logger = Logger(__name__, "Newton")
    for t in range(int(newton_maxiter)):
        fx = func(x0)
        res = np.linalg.norm(fx)
        logger.info({
            "count": t,
            "residual": res,
        })
        if res <= newton_tol:
            return x0
        A = Jacobi(func, x0, fx=fx, **opt)
        dx = krylov.gmres(A, -fx, **opt)
        x0 = x0 + dx
    raise RuntimeError("Not convergent (Newton)")


def hook_step(A, b, trusted_region, hook_maxiter, hook_tol, nu=0, **opt):
    """
    optimal hook step based on trusted region approach

    Parameters
    ----------
    A : numpy.array
        square matrix
    b : numpy.array

    nu : float, optional (default=0.0)
        initial value of Lagrange multiplier

    Returns
    --------
    x : numpy.array
        argmin of x
    mu : float
        Lagrange multiplier

    References
    ----------
    - Numerical Methods for Unconstrained Optimization and Nonlinear Equations
      J. E. Dennis, Jr. and Robert B. Schnabel
      http://dx.doi.org/10.1137/1.9781611971200
      Chapter 6.4: THE MODEL-TRUST REGION APPROACH

    """
    logger = Logger(__name__, "Hook step")
    r = trusted_region
    logger.debug({"nu0": nu, "trusted_region": r, })
    I = np.matrix(np.identity(len(b), dtype=b.dtype))
    AA = np.dot(A.T, A)
    Ab = np.dot(A.T, b)
    xi = np.linalg.solve(AA, Ab)
    if np.linalg.norm(xi) < r:
        return xi, 0
    for t in range(int(hook_maxiter)):
        xi = np.linalg.solve(AA+nu*I, Ab)
        xi_norm = np.linalg.norm(xi)
        Psi = xi_norm - r
        logger.info({
            "count": t,
            "1-|xi|/r": Psi / r,
        })
        if abs(Psi) < hook_tol * r:
            return xi, nu
        dPsi = -np.dot(xi, np.linalg.solve(AA+nu*I, xi)) / xi_norm
        nu = nu - (xi_norm*Psi) / (r*dPsi)
        if nu < 0:
            logger.debug("Negative nu, reset to zero")
            nu = 0
        logger.debug({
            "count": t,
            "Psi": Psi,
            "dPsi": dPsi,
            "nu": nu,
        })
    raise RuntimeError("Not convergent (hook-step)")


@array_adapter
def newton_krylov_hook_gen(func, x0, trusted_region, **opt):
    """ Generator of Newton-Krylov-hook iteration

    Yields
    -------
    x : numpy.array
        :math:`x_n`
    residual : float
        :math:`|F(x_n)|`
    fx : numpy.array
        :math:`F(x_n)`
    """
    logger = Logger(__name__, "NewtonKrylovHook")
    nu = 0.0
    for t in icount():
        fx = func(x0)
        res = np.linalg.norm(fx)
        logger.info({
            "count": t,
            "residual": res,
        })
        yield x0, res, fx
        A = Jacobi(func, x0, fx=fx, **opt)
        b = -fx
        H, V = krylov.arnoldi(A, b, **opt)
        g = krylov.solve_Hessenberg(H, np.linalg.norm(b))
        dx = np.dot(V[:, :len(g)], g)
        dx_norm = np.linalg.norm(dx)
        logger.info({"|dx|": dx_norm, })
        if dx_norm < trusted_region:
            logger.info('in Trusted region')
            x0 = x0 + dx
        else:
            logger.info('hook step')
            beta = np.zeros(H.shape[0])
            beta[0] = np.linalg.norm(b)
            xi, nu = hook_step(H, beta, trusted_region, nu=nu, **opt)
            dx = np.dot(V[:, :len(xi)], xi)
            x0 = x0 + dx


@array_adapter
def newton_krylov_hook(func, x0, **opt):
    """ Solve multi-dimensional nonlinear equation :math:`F(x)=0`

    Parameters
    -----------
    func : numpy.array -> numpy.array
        The function :math:`F` of the problem.
    x0 : numpy.array
        Initial guess of the solution
    opt : dict
        Options for calculation.
        Please generate by :py:func:`continuate.get_default_options`

    Examples
    ---------
    >>> import continuate
    >>> opt = continuate.get_default_options()
    >>> opt["newton_tol"] = 1e-7
    >>> f = lambda x : (x-1)**2
    >>> x0 = np.array([1.1, 0.9])
    >>> x = newton_krylov_hook(f, x0, **opt)
    >>> np.allclose(f(x), np.zeros_like(x), atol=1e-7)
    True

    """
    tol = opt["newton_tol"]
    maxiter = opt["newton_maxiter"]
    gen = newton_krylov_hook_gen(func, x0, **opt)
    for t, (x, res, _) in enumerate(gen):
        if res < tol:
            return x
        if t > maxiter:
            raise RuntimeError("Not Convergent (Newton-Krylov-hook)")
