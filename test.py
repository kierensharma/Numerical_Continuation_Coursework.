import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def vdp(t, z):
    x, y = z
    return [y, mu*(1 - x**2)*y - x]

def predator_prey(t, u0, b):
    a = 1
    d = 0.1

    x, y = u0
    dxdt = x*(1 - x) - (a*x*y)/(d + x)
    dydt = (b*y)*(1 - y/x)
    dXdt = np.array([dxdt, dydt])

    return dXdt

t1, t2 = 0, 100
t = np.linspace(t1, t2, 500)
bs = [0.1, 0.26, 0.5]
styles = ["-", "--", ":"]

for b, style in zip(bs, styles):
    sol = solve_ivp(lambda t, u: predator_prey(t, u, b), (t1, t2), (0.5, 2))
    plt.plot(sol.y[0], sol.y[1], style)

plt.legend([f"$b={b}$" for b in bs])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()