import numpy as np
from matplotlib import pyplot as plt

# Single Euler step function, with step size h
def euler_step(f,x0,t0,h):
    x1 =  x0 + h*f(x0,t0)

    return x1

def RK4_step(f,x0,t0,h):

    return x1

# Uses 'euler_step' between two points (x1,t1) and (x2,t2), with step delta_max
def solve_to(f, x0, t1, t2, method):
    delta_max = 0.01
    steps = (t2-t1)/delta_max
    t = np.linspace(t1,t2,steps)
    x = np.zeros(len(t))
    x[0] = x0

    if method == 'Euler':
        for n in range(0,len(t)-1):
            x[n+1] = euler_step(f,x[n],t[n],delta_max)

    elif method == 'RK4':
        for n in range(0,len(t)-1):
            x[n+1] = RK4_step()

    else:
        print('Error: Please define a valid step method.')

    return x[-1]

# Uses 'solve_to' to generate a seriese of estimates for x1,x2,x3,...
def solve_ode(func, x0, t, method):
    x = np.zeros(len(t))
    x[0] = x0
    for n in range(0,len(t)-1):
        x[n+1] = solve_to(func, x[n], t[n], t[n+1])

    plt.plot(t,x)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.show()

    return x


if __name__ == '__main__':
    func_1 = lambda x,t: x
    x0 = 1
    t = np.linspace(0,10,100)
    solve_ode(func_1, x0, t)
