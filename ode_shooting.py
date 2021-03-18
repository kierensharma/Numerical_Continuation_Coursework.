import numpy as np 
from ode_solver import solve_ode

def main():
    t = np.linspace(0,10,200)

    

def shoot_ode(func, a, b, z1, z2, t, tol):
    max_shoots = 15
    n = len(t)

    for i in range(max_shoots):

        Sol = solve_ode(func, [a, z1], t, 1, 'RK4')
        u2 = Sol[n-1, 0]

        if abs(b - u2) < tol:
            break

    return y[:, 0]


def predator_prey(t, u, b):
    a = 1
    d = 0.1

    x, y = u
    dxdt = x*(1 - x) - (a*x*y)/(d + x)
    dydt = (b*y)*(1 - y/x)
    dXdt = np.array([dxdt, dydt])

    return dXdt

