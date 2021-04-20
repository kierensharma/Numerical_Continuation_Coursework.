import numpy as np
from ode_shooting import limit_cycle_isolator as lim
from ode_shooting import phase_portrait_plotter as phase_plt

def main(filename=None):
    shooting_result = lim(hoph_bifurcation, [1, 1, 5], phase_condition)
    exact_result = explicit_sol(shooting_result.t)
    tol = 1e-2

    if np.allclose(shooting_result.y, exact_result, atol=tol):
        print("Passed test")
    else:
        print("Failed test")

    plt = phase_plt(shooting_result)
    plt.plot(exact_result[0], exact_result[1], label='exact')
    plt.legend()
    if filename is None:
        # (show on screen)
        plt.show()
    else:
        # (save to file)
        plt.savefig(filename)

def hoph_bifurcation(t, u0):
    sigma = -1 
    beta = 3

    u1, u2 = u0
    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    dUdt = np.array([du1dt, du2dt])

    return dUdt

def explicit_sol(t):
    phase = 0
    beta = 3

    u1 = np.sqrt(beta)*np.cos(t + phase)
    u2 = np.sqrt(beta)*np.sin(t + phase)
    U = np.array([u1, u2])
    return U

def phase_condition(ode, u0, T):  return np.array(ode(0, u0)[0])

if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    main(*args)