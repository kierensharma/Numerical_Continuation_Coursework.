import numpy as np
from ode_shooting import shooting as shoot
from ode_shooting import limit_cycle_isolator as lim
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

def main():
    sols = natural_param_continuation(cubic, np.array([1.52137971,-2]),  np.linspace(-1.98, 2, 200))
    plt.plot(np.linspace(-2, 2, 201), sols)
    plt.xlabel('alpha')
    plt.ylabel('x')
    plt.show()
    # result = numerical_continuation(hoph_bifurcation, [1, 1, 5], phase_condition)
    # print(result)

def numerical_continuation(ode, est, phase_condition):
    def stack(ode, est, phase_condition):
        return np.hstack((shoot(ode, est, phase_condition), 
                            pseudo_arclength_equation(ode, est)))

    print(len(stack(hoph_bifurcation, [1,1,5], phase_condition)))
    
    result = fsolve(lambda est: stack(ode, est, phase_condition), est)
    return result

def natural_param_continuation(f, est, param):
    c1 = est[-1]
    u0 = est[:-1]
    print(type(u0))
    sol1 = fsolve(f, u0, c1)
    sol2 = sol1
    sols = np.array(sol1)

    for c in param:
        sol2 = fsolve(f, sol2, c)
        sols = np.append(sols, sol2)
    return sols

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

def cubic(x, c):
    # x = x[0]
    return x**3 - x + c

def phase_condition(ode, u0, T):  return np.array(ode(0, u0)[0])

if __name__ == '__main__':
    main()