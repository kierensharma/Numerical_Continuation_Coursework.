import numpy as np
from ode_shooting import shooting as shoot
from ode_shooting import limit_cycle_isolator as lim
from scipy.optimize import fsolve

def main():
    result = pseudo_arclength_continuation(hoph_bifurcation, [1, 1, 5], phase_condition)
    print(result)

def pseudo_arclength_continuation(ode, est, phase_condition):
    def stack(ode, est, phase_condition):
        return np.hstack((shoot(ode, est, phase_condition), 
                                            pseudo_arclength_equation(ode, est)))
    
    print(len(stack(hoph_bifurcation, [1,1,5], phase_condition)))
    result = fsolve(lambda est: stack(ode, est, phase_condition), est)
    return result

def pseudo_arclength_equation(f, est):
    u0 = est[0:-1]
    T = est[-1]

    dX = np.array(f(T, u0) - u0)
    u1 = np.array(u0 + dX)

    return np.vdot(dX, (u0 - u1))

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