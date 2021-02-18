import numpy as np
from matplotlib import pyplot as plt

# Single Euler step function, with step size h
def euler_step(f, x0, t0, h):
    x1 =  x0 + h*f(x0,t0)

    return x1

# Uses 'euler_step' between two points (x1,t1) and (x2,t2), with step delta_max
def solve_to(f, x0, t1, t2):
    delta_max = 5
    t_current = t1
    x_current = x0

    while t_current < t2:
        if t_current + delta_max > t2:
            delta_max = t2 - t_current
        
        x_current = euler_step(f, x_current, t_current, delta_max)

        t_current = t_current + delta_max

    return x_current

# Uses 'solve_to' to generate a seriese of estimates for x1,x2,x3,...
def solve_ode(func, x0, t):
    x = np.zeros(len(t))
    x[0] = x0
    for n in range(len(t)-1):
        x[n+1] = solve_to(func, x[n], t[n], t[n+1])

    return x

# Defines ODE function.
def func_1(x, t):
    return x


if __name__ == '__main__':
    x0 = 1
    t = np.linspace(0,10,100)
    x = solve_ode(func_1, x0, t)

    x_true = np.exp(t)

    plt.plot(t,x,'b.-',t,x_true,'r-')
    plt.legend(['Euler','True'])
    plt.grid(True)
    plt.title("Solution of $x'=x , x(0)=1$")
    plt.show()
