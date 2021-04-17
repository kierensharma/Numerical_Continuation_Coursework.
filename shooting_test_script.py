import numpy as np
from ode_shooting import limit_cycle_isolator as lim

def main():
    result = myfunc(0.5)
    if abs(result - 2.4794255) < 1e-6:  # value computed with a calculator
        print("Test passed")
    else:
        print("Test failed")

def hoph_bifurcation(t, u0):
    sigma = -1 
    beta = 1

    u1, u2 = u0
    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2(u1**2 + u2**2)
    dUdt = np.array([du1dt, du2dt])

    return dUdt

if __name__ == '__main__':
    main()