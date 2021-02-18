import numpy as np
from matplotlib import pyplot as plt

# Single Euler step function, with step size h
def euler_step(f,x0,t0,h):
    x1 =  x0 + h*f(x0,t0)

    return x1

# Uses 'euler_step' between two points (x1,t1) and (x2,t2), with step delta_max
def solve_to(f, x0, t0, t2, delta_max):
    steps = (t2-t0)/delta_max
    t = np.linspace(t0,t2,steps)
    x = np.zeros(len(t))
    x[0] = x0

    for n in range(0,len(t)-1):
        x[n+1] = euler_step(f,x[n],t[n],delta_max)

    plt.plot(t,x)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.show()

# Uses 'solve_to'
def solve_ode():


if __name__ == '__main__':
    func_1 = lambda x,t: x
    solve_to(f=func_1, x0=1, t0=0, t2=2, delta_max=0.1)
