import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# Single 4th-order Runga Kutta step function.
def RK4_step(f, x0, t0, h):
    k1 = h*f(x0, t0)
    k2 = h*f(x0 + 0.5*k1, t0 + 0.5*h)
    k3 = h*f(x0 + 0.5*k2, t0 + 0.5*h)
    k4 = h*f(x0 + k3, t0 + h)

    x1 = x0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

    return x1

# Uses 'RK4_step' between two points (x1,t1) and (x2,t2), with step delta_max
def solve_to(f, x0, t1, t2, delta_max):
    t_current = t1
    x_current = x0

    while t_current < t2:
        if t_current + delta_max > t2:
            delta_max = t2 - t_current
        
        x_current = RK4_step(f, x_current, t_current, delta_max)

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
    error =  sum([abs(x-np.exp(t)) for x, t in zip(xs, ts)])

    return error

if __name__ == '__main__':
    x0 = 1
    ts = np.linspace(0,10,100)
    f = lambda x,t: x

    # xs = solve_ode(f, x0, ts, 1)
    # x_true = np.exp(ts)

    # plt.plot(ts,xs,'b.-',ts,x_true,'r-')
    # plt.legend(['RK4','True'])
    # plt.grid(True)
    # plt.title("Solution of $x'=x , x(0)=1$")
    # plt.show()
    

    errors = []
    delta_max_range = np.logspace(-6, -1, 20)

    for delta_max in tqdm(delta_max_range):
        xs = solve_ode(f, x0, ts, delta_max)
        errors.append(error(xs, ts))

    plt.plot(delta_max_range, errors)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Step-size')
    plt.ylabel('Error')
    plt.grid(True)
    plt.title("Plot of global error against step-size, for RK4 method")
    plt.savefig('Figures/RK4_error_plot.pdf')
    plt.show()
