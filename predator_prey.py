# %%
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from matplotlib import pyplot
from math import nan

# %%
def predator_prey(t, u, b):
    a = 1
    d = 0.1

    x, y = u
    dxdt = x*(1 - x) - (a*x*y)/(d + x)
    dydt = (b*y)*(1 - y/x)
    dXdt = np.array([dxdt, dydt])

    return dXdt


# %%

sol = solve_ivp(lambda t, u: predator_prey(t, u, 0.26), (0, 100), (1.5, 1.75))
pyplot.plot(sol.t, sol.y[0, :], sol.y[1, :])

# %%

sol = solve_ivp(lambda t, u: predator_prey(t, u, 0.26), (0, 100), (0.5, 0.75))
pyplot.plot(sol.y[0, :], sol.y[1, :])
sol = solve_ivp(lambda t, u: predator_prey(t, u, 0.26), (0, 100), (0.75, 1))
pyplot.plot(sol.y[0, :], sol.y[1, :])
sol = solve_ivp(lambda t, u: predator_prey(t, u, 0.26), (0, 100), (1, 1.5))
pyplot.plot(sol.y[0, :], sol.y[1, :])
sol = solve_ivp(lambda t, u: predator_prey(t, u, 0.26), (0, 100), (1.5, 1.75))
pyplot.plot(sol.y[0, :], sol.y[1, :])
sol = solve_ivp(lambda t, u: predator_prey(t, u, 0.26), (0, 100), (1.75, 2))
pyplot.plot(sol.y[0, :], sol.y[1, :])

# %%

# V nullcline
Vval = np.linspace(-60, 20, 81)
Nval = np.zeros(np.size(Vval))
for (i, V) in enumerate(Vval):
    result = root(lambda N: predator_prey(nan, (V, N), 0.26)[0], 0)
    if result.success:
        Nval[i] = result.x
    else:
        Nval[i] = nan
pyplot.plot(Vval, Nval)

# N nullcline
Vval = np.linspace(-60, 20, 81)
Nval = np.zeros(np.size(Vval))
for (i, V) in enumerate(Vval):
    result = root(lambda N: morris_lecar(nan, (V, N), 30)[1], 0)
    if result.success:
        Nval[i] = result.x
    else:
        Nval[i] = nan
pyplot.plot(Vval, Nval)

# %%
result = root(lambda u: predator_prey(nan, u, 0.26), (-40, 0))
if result.success:
    print("Equilibrium at {}".format(result.x))
else:
    print("Failed to converge")
result = root(lambda u: predator_prey(nan, u, 0.26), (-20, 0))
if result.success:
    print("Equilibrium at {}".format(result.x))
else:
    print("Failed to converge")
result = root(lambda u: predator_prey(nan, u, 0.26), (5, 0.3))
if result.success:
    print("Equilibrium at {}".format(result.x))
else:
    print("Failed to converge")

# %%
