import numpy as np

def main(filename=None):
    x0 = 0.4
    a = 1
    # b in [0.1,0.5]
    b = 0.1
    d = 0.1
    dxdt[0] = 0


# X = np.arrray([x, y])
def predator_prey(X, t, a, b, d):
    x, y = x
    dxdt = x*(1 - x) - (a*x*y)/(d + x)
    dydt = b*y*(1 - y/x)
    dXdt = [dxdt, dvdt]

    return np.asarray(dXdt)
