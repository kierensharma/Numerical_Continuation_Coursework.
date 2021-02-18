import numpy as np
from matplotlib import pyplot as plt

def euler_step(f,x0,t0,h):
    x1 =  x0 + h*f(x0,t0)
return x1

def solve_to(t1, t2, delta_max):
    steps = (t2-t1)/delta_max
    t = np.linspace(t1,t2,steps)


def solve_ode():


if __name__ == '__main__':
