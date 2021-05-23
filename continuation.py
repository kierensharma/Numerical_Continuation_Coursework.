import numpy as np
from ode_shooting import shooting as shoot
from ode_shooting import limit_cycle_isolator as lim
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from numpy import linalg as LA

def main():
    # # Testing natural parameter continuation on the algebraic cubic equation.
    # sols = cubic_continuation(cubic, [1.52137971,-2],  np.linspace(-1.98, 2, 200))
    # plt.plot(np.linspace(-2, 2, 201), sols)
    # plt.xlabel('alpha')
    # plt.ylabel('x')
    # plt.show()

    # Initial guess for (x, y, T)
    initial_guess = [3, 0.018, 6.3]
    second_guess = [3, 0.018, 6.29]
    param = np.arange(3, -3, -0.01)
    # sol = pseudo_arclength_continuation(hoph_bifurcation, [1, 1, 6, 1], [1.5, 1.5, 6, 1.1], phase_condition)
    # print(sol)
    # phase_portrait_plotter(sol)


    sols = natural_parameter_continuation(hoph_bifurcation, initial_guess, second_guess,  param, phase_condition)
    # print(sols)
    norms = []
    for sol in sols:
        norms.append(LA.norm(sol[:-1]))

    plt.plot(param, norms)
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\|\|x\|\|$')
    plt.xlim([-2, 2])
    plt.ylim([-0.09, 2])
    plt.grid()
    plt.title('Pseudo-arclength Continuation of Hoph Bifurcation')
    plt.show()

def cubic_continuation(eq, est, params):
    c1 = est[-1]
    u0 = est[:-1]

    sol = fsolve(eq, u0, c1)
    sols = np.array(sol)

    for c in params:
        sol = fsolve(eq, sol, c)
        sols = np.append(sols, sol)
    return sols

def pseudo_arclength_continuation(pde, current, guess, phase_condition):
    def stack(f, current, guess, phase_condition):
        est = guess[:-1]
        c = guess[-1]

        # print(guess)

        return np.hstack((shoot(wrap(f, c), est, phase_condition), 
                            pseudo_arclength_equation(current, guess)))

    corrected = fsolve(lambda U: stack(pde, current, U, phase_condition), guess)
    return corrected

def natural_parameter_continuation(f, est1, est2, param, phase_condition):
    sols = np.array([est1])
    sols = np.append(sols, [est2], axis=0)

    for index in range(1,len(param)):
        current = np.append(sols[-1], param[index])
        guess = current + secant(np.append(sols[-2],param[index-1]), current)
        sol = pseudo_arclength_continuation(f, current, guess, phase_condition)
        sols = np.append(sols, [sol[:-1]], axis=0)

    return sols[1:]

def secant(initials, second_guess): return second_guess-initials

def pseudo_arclength_equation(initials, second_guess):
    x0 = initials[:-1]
    p0 = initials[-1]
    v0 = np.append(x0, p0)

    x1 = second_guess[:-1]
    p1 = second_guess[-1]
    v1 = np.append(x1, p1)

    # secant = v1-v0
    prod = v1 - secant(initials, second_guess)

    return np.dot(secant(initials, second_guess), prod)

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