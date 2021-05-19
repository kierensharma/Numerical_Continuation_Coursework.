import numpy as np
from ode_shooting import shooting as shoot
from ode_shooting import limit_cycle_isolator as lim
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

def main():
    # # Testing natural parameter continuation on the algebraic cubic equation.
    # sols = cubic_continuation(cubic, [1.52137971,-2],  np.linspace(-1.98, 2, 200))
    # plt.plot(np.linspace(-2, 2, 201), sols)
    # plt.xlabel('alpha')
    # plt.ylabel('x')
    # plt.show()

    # Initial guess for (x, y, T)
    initial_guess = [0.5, 2, 40]
    second_guess = [0.5, 2, 40]

    sols = natural_parameter_continuation(hoph_bifurcation, initial_guess, second_guess,  np.linspace(0, 2, 200), phase_condition)
    print(sols)
    # result = numerical_continuation(hoph_bifurcation, [0.5,2,40,0], [1,3,45,1], phase_condition)
    # print(result)

def cubic_continuation(eq, est, params):
    c1 = est[-1]
    u0 = est[:-1]

    sol = fsolve(eq, u0, c1)
    sols = np.array(sol)

    for c in params:
        sol = fsolve(eq, sol, c)
        sols = np.append(sols, sol)
    return sols

def pseudo_arclength_continuation(pde, initial, second, phase_condition):
    def stack(f, initial, second, phase_condition):
        est = second[:-1]
        c = second[-1]

        print(np.hstack((shoot(wrap(f, c), est, phase_condition), 
                            pseudo_arclength_equation(initial, second))))

        return np.hstack((shoot(wrap(f, c), est, phase_condition), 
                            pseudo_arclength_equation(initial, second)))

    result = fsolve(lambda second: stack(pde, initial, second, phase_condition), second)
    return result

def natural_parameter_continuation(f, est1, est2, param, phase_condition):
    sol1 = est1
    sol2 = est2
    initials = np.append(sol1, param[0])
    initials2 = np.append(sol2, param[1])
    sol = pseudo_arclength_continuation(f, initials, initials2, phase_condition)
    sols = np.array(sol)

    for c in param:
        initials = np.append(sol, c)
        initials2 = np.append(sols[-2], c)
        sol = pseudo_arclength_continuation(f, initials, initials2, phase_condition)
        sols = np.append(sols, sol)
    return sols

def pseudo_arclength_equation(initials, second_guess):
    x0 = initials[:-1]
    p0 = initials[-1]
    v0 = np.append(x0, p0)

    x1 = second_guess[:-1]
    p1 = second_guess[-1]
    v1 = np.append(x1, p1)

    secant = v1-v0
    prod = v1 - secant

    return np.dot(secant, prod)

def wrap(function: callable, params): return lambda t, U: function(t, U, params)

def hoph_bifurcation(t, u0, params):
    # beta = 3
    beta = params

    u1, u2 = u0
    du1dt = beta*u1 - u2 - u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 - u2*(u1**2 + u2**2)
    dUdt = np.array([du1dt, du2dt])

    return dUdt

def cubic(x, c):
    # x = x[0]
    return x**3 - x + c

def phase_condition(ode, u0, T):  return np.array(ode(0, u0)[0])

if __name__ == '__main__':
    main()