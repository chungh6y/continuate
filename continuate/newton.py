# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse.linalg as linalg
from itertools import count as icount
from . import Logger
from . import krylov


def Jacobi(func, x0, alpha=1e-7, fx=None):
    """
    Jacobi oprator :math:`DF(x0)`, where

    .. math::
        DF(x0)dx = (F(x0 + \\alpha dx / |dx|) - F(x0))/(\\alpha/|dx|)

    Parameters
    ----------
    func: numpy.array -> numpy.array
        A function to be differentiated
    x0: numpy.array
        A position where the Jacobian is evaluated
    alpha: float

    fx: numpy.array, optional
        func(x0)

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x: np.array([x[1]**2, x[0]**2])
    >>> x0 = np.array([1, 2])
    >>> J = Jacobi(f, x0)

    """
    if fx is None:
        fx = func(x0)

    def wrap(v):
        norm = np.linalg.norm(v)
        r = alpha / norm
        return (func(x0 + r * v) - fx) / r

    return linalg.LinearOperator(
        (len(x0), len(x0)),
        matvec=wrap,
        dtype=x0.dtype
    )


class Hessian(object):
    """ Calculate the deviation from linear approximation """

    logger = Logger(__name__, "Hessian")

    def __init__(self, func, x0, alpha=1e-7):
        self.fx0 = func(x0)
        self.A = Jacobi(func, x0, fx=self.fx0, alpha=1e-7)
        self.f = func
        self.x0 = x0
        self.alpha = alpha

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


def newton(func, x0, ftol=1e-5, maxiter=100, inner_tol=1e-6):
    """
    solve multi-dimensional equation `F(x) = 0` using Newton-Krylov method.

    Parameters
    -----------
    func: numpy.array -> numpy.array
        `F`
    x0: numpy.array
        initial guess of the solution
    ftol: float, optional
        tolerance of the iteration
    maxiter: int, optional
        maximum number of trial
    inner_tol: float, optional
        tolerance of linear solver (use scipy.sparse.linalg.gmres)

    Returns
    --------
    numpy.array
        The solution `x` satisfies `F(x)=0`

    """
    logger = Logger(__name__, "Newton")
    for t in range(maxiter):
        fx = func(x0)
        res = np.linalg.norm(fx)
        logger.info({
            "count": t,
            "residual": res,
        })
        if res <= ftol:
            return x0
        A = Jacobi(func, x0, fx=fx)
        dx = krylov.gmres(A, -fx, eps=inner_tol)
        x0 = x0 + dx
    raise RuntimeError("Not convergent (Newton)")


def hook_step(A, b, r, nu=0, maxiter=100, e=0.1):
    """
    optimal hook step based on trusted region approach

    Parameters
    ----------
    A : numpy.array
        square matrix
    b : numpy.array
    r : float
        trusted region
    nu : float, optional (default=0.0)
        initial value of Lagrange multiplier
    e : float, optional (default=0.1)
        relative tolerance of residue form r

    Returns
    --------
    (numpy.array, float)
        argmin of xi, and nu (Lagrange multiplier)
        CAUTION: nu may be not accurate

    References
    ----------
    - Numerical Methods for Unconstrained Optimization and Nonlinear Equations
      J. E. Dennis, Jr. and Robert B. Schnabel
      http://dx.doi.org/10.1137/1.9781611971200
      Chapter 6.4: THE MODEL-TRUST REGION APPROACH

    """
    logger = Logger(__name__, "Hook step")
    logger.debug({"nu0": nu, "trusted_region": r, })
    I = np.matrix(np.identity(len(b), dtype=b.dtype))
    AA = np.dot(A.T, A)
    Ab = np.dot(A.T, b)
    for t in range(maxiter):
        xi = np.linalg.solve(AA+nu*I, Ab)
        xi_norm = np.linalg.norm(xi)
        Psi = xi_norm - r
        logger.info({
            "count": t,
            "1-|xi|/r": Psi / r,
        })
        if abs(Psi) < e * r:
            return xi, nu
        dPsi = -np.dot(xi, np.linalg.solve(AA+nu*I, xi)) / xi_norm
        nu = nu - (xi_norm*Psi) / (r*dPsi)
        logger.debug({
            "count": t,
            "Psi": Psi,
            "dPsi": dPsi,
            "nu": nu,
        })
    raise RuntimeError("Not convergent (hook-step)")


def newton_krylov_hook_gen(func, x0, r=1e-2, inner_tol=1e-6, **kwds):
    """ Generator of Newton-Krylov-hook iteration

    Parameters
    -----------
    r : float, optional(default=1e-2)
        Initial guess of the radius of trusted region

    Returns
    --------
    generator
        yields `(x, |func(x)|, func(x))`
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
        A = Jacobi(func, x0, fx=fx)
        b = -fx
        H, V = krylov.arnoldi(A, b, eps=inner_tol)
        g = krylov.solve_Hessenberg(H, np.linalg.norm(b))
        dx = np.dot(V[:, :len(g)], g)
        dx_norm = np.linalg.norm(dx)
        logger.info({"|dx|": dx_norm, })
        if dx_norm < r:
            logger.info('in Trusted region')
            x0 = x0 + dx
        else:
            logger.info('hook step')
            beta = np.zeros(H.shape[0])
            beta[0] = np.linalg.norm(b)
            xi, nu = hook_step(H, beta, r, nu=nu)
            dx = np.dot(V[:, :len(xi)], xi)
            x0 = x0 + dx


def newton_krylov_hook(func, x0, r=1e-2, ftol=1e-5, inner_tol=1e-6,
                       maxiter=100, **kwds):
    gen = newton_krylov_hook_gen(func, x0, r=r, inner_tol=inner_tol, **kwds)
    for t, (x, res, _) in enumerate(gen):
        if res < ftol:
            return x
        if t > maxiter:
            raise RuntimeError("Not Convergent (Newton-Krylov-hook)")
