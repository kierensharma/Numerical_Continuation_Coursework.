import numpy as np
from ode_shooting import shooting as shoot
from scipy.optimize import fsolve

def main():
    result = continuation(hoph_bifurcation, [1, 1, 5], phase_condition)
    print(realistic)

def continuation(ode, est, phase_condition):
    result = fsolve(lambda est: np.hstack(shoot(ode, est, phase_condition), 
                                        pseudo_arclength(ode, est)), est)
    return result

def pseudo_arclength(f, est):
    u0 = est[0:-1]
    continuation = np.vdot(secant(f, est), (u0 - predict(f, est)))
    return continuation

def secant(f, est):
    u0 = est[0:-1]
    T = est[-1] 

    dX = np.array(f(u0,T) - u0)
    return dX

def predict(f, est):
    u0 = est[0:-1]
    u1 = np.array(u0 + secant(f, u0))
    return u1

def hoph_bifurcation(t, u0):
    sigma = -1 
    beta = 3

    u1, u2 = u0
    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    dUdt = np.array([du1dt, du2dt])

    return dUdt

def phase_condition(ode, u0, T):  return np.array(ode(0, u0)[0])

if __name__ == '__main__':
    main()