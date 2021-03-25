import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

def main():
    

# Wrapped function for numerical integrator used in 'shooting' method
def integrate(ode, u0, T):
    sol = solve_ivp(ode, (20, T), u0)

    return sol.y[:, -1]


# Function to return phase-condition of dx/dt(0) = 0
def phase_condition(ode, u0, T):  return np.array(ode(0, u0)[0])


# Definition of 'shooting' function which returns difference from initial conditions of some arbitrary initial guess ũ0, along with phase condition
def shooting(ode, u0, T):   return np.hstack((u0 - integrate(ode, u0, T), phase_condition(ode, u0, T)))


# Function which uses numerical root finder to isolate limit cycles, using 'shooting' function and suitable initial guess ũ0
def limit_cycle_isolator(shoot, u0):
    root = fsolve 


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

