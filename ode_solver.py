import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.integrate import odeint

def main(filename=None):
    X0 = [0, 1]
    ts = np.linspace(0,10,200)
    f = lambda x,t: x

    method = 'RK4'
    t, x, v = solve_ode(f, X0, ts, 1, method)
    sol_true = odeint(f_shm, X0, ts)
    true_x_sol= sol_true[:, 0]
    true_v_sol = sol_true[:, 1]

    fig = plot_solution(t, x, v, true_x_sol, true_v_sol)
    plt.show


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
def solve_ode(func, x0, t, delta_max, method):
    x = np.zeros(len(t))
    x[0] = x0
    for n in range(len(t)-1):
        x[n+1] = solve_to(func, x[n], t[n], t[n+1], delta_max, method)

    return t, x, v

# Calculates the global error by suming the differences between the actual and estimated 'x' values
def error(xs, ts):
    error =  sum([abs(x-np.exp(t)) for x, t in zip(xs, ts)])

    return error

# X = np.arrray([x, v])
def f_shm(X, t):
    x, v = X
    dxdt = v
    dvdt = -x
    dXdt = [dxdt, dvdt]
    return dXdt


# Function to plot values of x and t, alongside real solution
def plot_solution(t, x, v, true_x_sol, true_v_sol):
    fig = plt.figure()
    plt.set_title('Solution of system of ODEs')

    plt.plot(t, x, color='green', linewidth=2, label=r'$x$')
    plt.plot(t, v, color='blue', linewidth=2, label=r'$v$')

    plt.plot(t, true_x_sol, color='g.-', linewidth=2, label=r'$True x$')
    plt.plot(t, true_v_sol, color='b.-', linewidth=2, label=r'$True v$')

    plt.set_xlabel('t')
    plt.legend()

    return fig


if __name__ == '__main__':
   

    plt.plot(ts,xs,'b.-',ts,x_true,'r-')
    plt.legend([method,'True'])
    plt.grid(True)
    plt.title("Solution of $x'=x , x(0)=1$")
    plt.show()

    # Euler_errors = []
    # RK4_errors = []
    # delta_max_range = np.logspace(-6, -1, 20)

    # for delta_max in tqdm(delta_max_range):
    #     Euler_xs = solve_ode(f, x0, ts, delta_max, method='Euler')
    #     RK4_xs = solve_ode(f, x0, ts, delta_max, method='RK4')
    #     Euler_errors.append(error(Euler_xs, ts))
    #     RK4_errors.append(error(RK4_xs, ts))


    # plt.plot(delta_max_range, Euler_errors, 'b-', delta_max_range, RK4_errors, 'r-')
    # plt.legend(['Euler method','RK4 method'])
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Step-size')
    # plt.ylabel('Error')
    # plt.grid(True)
    # plt.title("Plot of global error against step-size, for both methods")
    # plt.savefig('Both_error_plot.pdf')
    # plt.show()
