import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import matplotlib as mpl

# Parameters
alpha = 1/np.pi**2
L = 1.0
T = 1

dx = 1/160
dt = dx**2

Nx = int(L / dx)
Nt = int(T / dt)

x = np.linspace(0, L, Nx+1)
r = alpha * dt / (dx**2)

# Initial and exact solution
u = np.sin(np.pi * x)
u_exact = lambda x, t: np.exp(-t) * np.sin(np.pi * x)

# Prepare matrix A and B (tridiagonal for CN method)
lower = -r/2 * np.ones(Nx-2)
main  = (1 + r) * np.ones(Nx-1)
upper = -r/2 * np.ones(Nx-2)

A = np.zeros((3, Nx-1))
A[0,1:] = upper      # upper diag
A[1,:] = main        # main diag
A[2,:-1] = lower     # lower diag

# Time stepping
for n in range(Nt):
    b = np.zeros(Nx-1)
    for i in range(1, Nx):
        b[i-1] = (r/2)*u[i-1] + (1 - r)*u[i] + (r/2)*u[i+1]
    u[1:Nx] = solve_banded((1,1), A, b)

# Plotting
u_num = u
u_ex = u_exact(x, T)
error = np.abs(u_num - u_ex)
print("Max error:", max(error))

# mpl.rcParams['font.family'] = 'Times New Roman'
# # Plot 1: Numerical and Exact Solutions
# plt.figure(figsize=(8,5))
# plt.plot(x, u_num, 'bo-', label='Numerical', linewidth=1.5, markersize=6, markevery=10)
# plt.plot(x, u_ex, 'r--^', label='Exact', linewidth=1.5, markersize=5, markevery=10)
# plt.xlabel('$x$')
# plt.ylabel('$u(x,t)$')
# plt.legend()
# plt.grid(True)
# # plt.savefig("Solution Problem 1.1.png", dpi=500)
# plt.show()

# # Plot 2: Error
# plt.figure(figsize=(8,5))
# plt.plot(x, error, 'r', linewidth=2)
# plt.xlabel('$x$')
# plt.ylabel('$Error$')
# plt.grid(True)
# # plt.savefig("error_plot Problem 1.1 .png", dpi=500)
# plt.show()
