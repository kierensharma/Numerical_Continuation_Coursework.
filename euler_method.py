import numpy as np
from matplotlib import pyplot as plt

# Single Euler step function, with step size h
def euler_step(f, x0, t0, h):
    x1 =  x0 + h*f(x0,t0)

    return x1

# Uses 'euler_step' between two points (x1,t1) and (x2,t2), with step delta_max
def solve_to(f, x0, t1, t2, delta_max):
    t_current = t1
    x_current = x0

    while t_current < t2:
        if t_current + delta_max > t2:
            delta_max = t2 - t_current
        
        x_current = euler_step(f, x_current, t_current, delta_max)

        t_current = t_current + delta_max

    return x_current

# Uses 'solve_to' to generate a seriese of estimates for x1,x2,x3,...
def solve_ode(func, x0, t, delta_max):
    x = np.zeros(len(t))
    x[0] = x0
    for n in range(len(t)-1):
        x[n+1] = solve_to(func, x[n], t[n], t[n+1], delta_max)

    return x

# Calculates the global error by suming the differences between the actual and estimated 'x' values
def error(xs, ts):
    error =  sum(abs(x-np.exp(t)) for x, t in zip(xs, ts))

    return error

if __name__ == '__main__':
    x0 = 1
    ts = np.linspace(0,10,100)
    f = lambda x,t: x 

    errors = []
    delta_max_range = np.logspace(-6, 0, 20)

    for delta_max in delta_max_range:
        xs = solve_ode(f, x0, ts, delta_max)
        errors.append(error(xs, ts))

    plt.plot(delta_max_range, errors)
    plt.xlabel('Stepsise')
    plt.ylabel('Error')
    plt.grid(True)
    plt.title("Error plot for Euler method")
    plt.show()
