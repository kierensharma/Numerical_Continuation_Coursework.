import numpy as np
import pylab as pl
from math import pi

def main():
    # Set problem parameters/functions
    kappa = 1.0   # diffusion constant
    L=1.0         # length of spatial domain
    T=0.5         # total time to solve for

    # Set numerical parameters
    mx = 10     # number of gridpoints in space
    mt = 1000   # number of gridpoints in time

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    print("deltax=",deltax)
    print("deltat=",deltat)
    print("lambda=",lmbda)
    if 0 < lmbda < 0.5:
        print("stability criterion= stable")
    else:
        print("stability criterion= unstable")

    # Set up the solution variables
    u_0 = np.zeros(x.size)        # u at current time step

    # Set initial condition
    for i in range(0, mx+1):
        u_0[i] = u_I(x[i], L)

    # result = forward_Euler(u_0, mx, mt, lmbda, [0, 0])
    # result = backward_Euler(u_0, mx, mt, lmbda, [0, 0])
    result = Crank_Nicholson(u_0, mx, mt, lmbda, [0, 0])

    # Plot the final result and exact solution
    pl.plot(x, result,'ro',label='num')
    xx = np.linspace(0,L,250)
    pl.plot(xx, u_exact(xx,T, L, kappa),'b-',label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()

def u_I(x, L):
    # initial temperature distribution
    y = (np.sin(pi*x/L))**2
    return y

def u_exact(x, t, L, kappa):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

# Solve the PDE: in matrix form
def forward_Euler(u_j, mx, mt, lmbda, boundaries):
    u_jp1 = np.zeros(u_j.size)    # u at next time step

    A_FE = np.zeros([mx-1,mx-1])
    np.fill_diagonal(A_FE, (1-2*lmbda))
    np.fill_diagonal(A_FE[1:], lmbda)
    np.fill_diagonal(A_FE[:,1:], lmbda)

    for j in range(0, mt):
        u_jp1[1:-1] = np.matmul(A_FE, u_j[1:-1].T)

        # Boundary conditions
        u_jp1[0] = boundaries[0]; u_jp1[mx] = boundaries[-1]
            
        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]

    return u_j

def backward_Euler(u_j, mx, mt, lmbda, boundaries):
    u_jp1 = np.zeros(u_j.size)    # u at next time step

    A_BE = np.zeros([mx-1,mx-1])
    np.fill_diagonal(A_BE, (1+2*lmbda))
    np.fill_diagonal(A_BE[1:], -lmbda)
    np.fill_diagonal(A_BE[:,1:], -lmbda)

    for j in range(0, mt):
        u_jp1[1:-1] = np.linalg.solve(A_BE, u_j[1:-1].T)

        # Boundary conditions
        u_jp1[0] = boundaries[0]; u_jp1[mx] = boundaries[-1]
            
        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]

    return u_j

def Crank_Nicholson(u_j, mx, mt, lmbda, boundaries):
    u_jp1 = np.zeros(u_j.size)    # u at next time step

    A_CN = np.zeros([mx-1,mx-1])
    np.fill_diagonal(A_CN, (1+lmbda))
    np.fill_diagonal(A_CN[1:], -(lmbda/2))
    np.fill_diagonal(A_CN[:,1:], -(lmbda/2))

    B_CN = np.zeros([mx-1,mx-1])
    np.fill_diagonal(B_CN, (1-lmbda))
    np.fill_diagonal(B_CN[1:], lmbda/2)
    np.fill_diagonal(B_CN[:,1:], lmbda/2)

    for j in range(0, mt):
        u_jp1[1:-1] = np.linalg.solve(A_CN, np.matmul(B_CN, u_j[1:-1].T))

        # Boundary conditions
        u_jp1[0] = boundaries[0]; u_jp1[mx] = boundaries[-1]
            
        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]

    return u_j

if __name__ == "__main__":
    main()