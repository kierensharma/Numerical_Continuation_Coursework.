import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.integrate import odeint

def main(filename=None):
    X0 = [0, 1]
    t = np.linspace(0,10,200)

    x, v = solve_ode(func, X0, t, 1, 'RK4')
    sol_true = odeint(func, X0, t)
    true_x_sol= sol_true[:, 0]
    true_v_sol = sol_true[:, 1]

    fig = plot_solution(t, x, v, true_x_sol, true_v_sol)

    if filename is None:
        # (show on screen)
        plt.show()
    else:
        # (save to file)
        fig.savefig(filename)


# Single Euler step function, with step size h.
def euler_step(f, x0, t0, h):
    x1 =  x0 + h*f(x0,t0)

    return x1

# Single 4th-order Runga Kutta step function.
def RK4_step(f, x0, t0, h):
    k1 = h*f(x0, t0)
    k2 = h*f(x0 + 0.5*k1, t0 + 0.5*h)
    k3 = h*f(x0 + 0.5*k2, t0 + 0.5*h)
    k4 = h*f(x0 + k3, t0 + h)

    x1 = x0 + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

    return x1

# Uses 'euler_step' or 'RK4_step between two points (x1,t1) and (x2,t2), with step delta_max.
def solve_to(f, x0, t1, t2, delta_max, method):
    t_current = t1
    x_current = x0

    if method == 'Euler':
        while t_current < t2:
            if t_current + delta_max > t2:
                delta_max = t2 - t_current
            
            x_current = euler_step(f, x_current, t_current, delta_max)
            t_current = t_current + delta_max

    elif method == 'RK4':
        while t_current < t2:
            if t_current + delta_max > t2:
                delta_max = t2 - t_current
            
            x_current = RK4_step(f, x_current, t_current, delta_max)
            t_current = t_current + delta_max
    
    else:
        print('Please provide a valid method; Euler or RK4')

    return x_current

# Uses 'solve_to' to generate a seriese of estimates for x1,x2,x3,...
def solve_ode(func, X0, t, delta_max, method):
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    Sol = np.column_stack((x, v))
    Sol[0][0] = X0[0]
    Sol[0][1] = X0[1]

    for n in range(len(t)-1):
        Sol[n+1] = solve_to(func, Sol[n], t[n], t[n+1], delta_max, method)

    x = Sol[:, 0]
    v = Sol[:, 1]

    return x, v

# Calculates the global error by suming the differences between the actual and estimated 'x' values
def error(xs, ts):
    error =  sum([abs(x-np.exp(t)) for x, t in zip(xs, ts)])

    return error

# X = np.arrray([x, v])
def func(X, t):
    x, v = X
    dxdt = v
    dvdt = -x
    dXdt = [dxdt, dvdt]

    return np.asarray(dXdt)


# Function to plot values of x and t, alongside real solution
def plot_solution(t, x, v, true_x_sol, true_v_sol):
    fig = plt.figure()
    plt.title('Solution of system of ODEs')

    plt.plot(t, x, color='green', linewidth=2, label=r'$x$')
    plt.plot(t, v, color='blue', linewidth=2, label=r'$v$')

    plt.plot(t, true_x_sol, 'g.-', linewidth=2, label=r'$True x$')
    plt.plot(t, true_v_sol, 'b.-', linewidth=2, label=r'$True v$')

    plt.xlabel('t')
    plt.grid()
    plt.legend()

    return fig


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    main(*args)