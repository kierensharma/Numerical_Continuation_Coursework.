import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

def main():
    # Initial guess for (x, y, T)
    initial_guess = (0.5, 2, 40)
    sol = limit_cycle_isolator(predator_prey, initial_guess)
    plt = phase_portrait_plotter(sol)
    # plt.plot(sol.t, sol.y[0, :])

    sol = solve_ivp(lambda t, u: predator_prey(t, u), (0, 100), (0.75, 1))
    plt.plot(sol.y[0, :], sol.y[1, :])
    sol = solve_ivp(lambda t, u: predator_prey(t, u), (0, 100), (1, 1.5))
    plt.plot(sol.y[0, :], sol.y[1, :])
    sol = solve_ivp(lambda t, u: predator_prey(t, u), (0, 100), (1.5, 1.75))
    plt.plot(sol.y[0, :], sol.y[1, :])

    plt.show()

# Wrapped function for numerical integrator used in 'shooting' method
def integrate(ode, u0, T):
    sol = solve_ivp(ode, (0, T), u0)

    return sol.y[:, -1]


# Function to return phase-condition of dx/dt(0) = 0
def phase_condition(ode, u0, T):  return np.array(ode(0, u0)[0])


# Definition of 'shooting' function which returns difference from initial conditions of some arbitrary initial guess ũ0, along with phase condition
def shooting(ode, est):  
    u0 = est[0:-1]
    T = est[-1] 
    
    return np.hstack((u0 - integrate(ode, u0, T), phase_condition(ode, u0, T)))

def orbit(ode, initialu, duration):
    sol = solve_ivp(ode, (0, duration), initialu)
    return sol


# Function which uses numerical root finder to isolate limit cycles, using 'shooting' function and suitable initial guess ũ0
def limit_cycle_isolator(ode, est):
    result = fsolve(lambda est: shooting(ode, est), est)
    isolated_sol = orbit(ode, result[0:-1], result[-1])

    # plt.plot(isolated_sol.y[0, :], isolated_sol.y[1, :])
    # plt.show()

    return isolated_sol

def phase_portrait_plotter(sol):
    plt.plot(sol.y[0, :], sol.y[1, :], label='Isolated periodic orbit')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Phase portrait')
    plt.legend()
    
    return plt


def predator_prey(t, u0):
    a = 1
    b = 0.26
    d = 0.1

    x, y = u0
    dxdt = x*(1 - x) - (a*x*y)/(d + x)
    dydt = (b*y)*(1 - y/x)
    dXdt = np.array([dxdt, dydt])

    return dXdt

if __name__ == '__main__':
    main()

