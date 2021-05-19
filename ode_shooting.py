import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from ode_solver import solve_ode

def main():
    # Initial guess for (x, y, T)
    initial_guess = [0.5, 2, 40]
    params = [1, 0.26, 0.1]
    sol = limit_cycle_isolator(ode(predator_prey, params), initial_guess, phase_condition)
    print(sol)
    plt = phase_portrait_plotter(sol)
    # plt.plot(sol.t, sol.y[0, :])

    # sol = solve_ivp(lambda t, u: predator_prey(t, u), (0, 100), (0.75, 1))
    # plt.plot(sol.y[0, :], sol.y[1, :])
    # sol = solve_ivp(lambda t, u: predator_prey(t, u), (0, 100), (1, 1.5))
    # plt.plot(sol.y[0, :], sol.y[1, :])
    # sol = solve_ivp(lambda t, u: predator_prey(t, u), (0, 100), (1.5, 1.75))
    # plt.plot(sol.y[0, :], sol.y[1, :])

    plt.show()

# Wrapped function for numerical integrator used in 'shooting' method
def integrate(ode, u0, T):
    t = np.linspace(0, T, 50)
    sol = solve_ivp(ode, (0, T), u0)
    # sol = solve_ode(ode, u0, t, 1, 'RK4')

    return sol.y[:, -1]

# Function to set the value of dx/dt(0) = 0
def phase_condition(ode, u0, T): return np.array(ode(0, u0)[0])

# Definition of 'shooting' function which returns difference from initial conditions of initial guess ũ0
def shooting(ode, est, phase_condition):
    u0 = est[0:-1]
    T = est[-1] 
    
    return np.hstack((u0 - integrate(ode, u0, T), phase_condition(ode, u0, T)))

def orbit(ode, initialu, duration):
    sol = solve_ivp(ode, (0, duration), initialu)
    return sol

def ode(function: callable, params: list): return lambda t, U: function(t, U, params)

# Function which uses numerical root finder to isolate limit cycles, using 'shooting' function and suitable initial guess ũ0
def limit_cycle_isolator(ode, est, phase_condition):
    """Isolates a periodic orbit (limiti cycle), if one exists, from a system of ODEs.

    USAGE:
        Sol = limit_cycle_isolator(ode, est)

    INPUT:
        ode                 - function defining the ODE, or system of ODEs, to be solved. 
                              Should return the right-hand side of the ODE as a numpy.array.
        est                 - numpy.array of initial guess, ũ0, at beginnig of limit cycle. Of form (x, y, T).
        phase_condition     - function defining a suitable phase condition to help isolate the limit cycle.
    
    OUTPUT:
        Sol                 - numpy.array containing the corrected initial values for the limit cycle. 
                              If the numerical root finder failed, the returned array is empty.
    """
    try:
        ode(1, est[0:-1])
    except ValueError:
        print('Error: Please check initial guess. Should be of form (x1,..., xn, T)')
        exit()

    result = fsolve(lambda est: shooting(ode, est, phase_condition), est)
    print(result)
    isolated_sol = orbit(ode, result[0:-1], result[-1])

    return isolated_sol

def phase_portrait_plotter(sol):
    plt.plot(sol.y[0, :], sol.y[1, :], label='Isolated periodic orbit')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Phase portrait')
    plt.legend()
    
    return plt

# Definition of predator-prey equations (more realistic version of the Lokta-Volterra equations)
def predator_prey(t, u0, params: list):
    # a = 1
    # b = 0.26
    # d = 0.1

    a, b, d = params

    x, y = u0
    dxdt = x*(1 - x) - (a*x*y)/(d + x)
    dydt = (b*y)*(1 - y/x)
    dXdt = np.array([dxdt, dydt])

    return dXdt

if __name__ == '__main__':
    main()

