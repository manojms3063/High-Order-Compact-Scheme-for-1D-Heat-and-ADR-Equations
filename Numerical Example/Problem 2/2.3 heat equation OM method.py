import numpy as np
import sympy as smp
from sympy import symbols, sin, exp, diff, lambdify
from scipy.sparse import diags
import scipy
import matplotlib.pyplot as plt

# Parameters
# m = 2**5
# n = 2**2
h = 1/160   
k = h**2
x0 = 0
xm = 1

T = 1
m = int((xm - x0) / h)
n = int(T/k)
x = np.zeros(m+1)
x[0] = x0
x[m] = xm

for i in range(0, m+1):
    x[i] = x0 + i*h
    
t = np.zeros(n+1)
for j in range(0, n+1):
    t[j] = j*k

# alpha = 1/np.pi**2
alpha = 1/np.pi**2


R= 1 / (12 * k) - alpha / (2 * h**2)
Q= 10 / (12 * k) + alpha / h**2
P= 1 / (12 * k) - alpha / (2 * h**2)

# Fill RHS diagonals
Rb= 1 / (12 * k) + alpha / (2 * h**2)
Qb = 10 / (12 * k) - alpha / h**2
Pb = 1 / (12 * k) + alpha / (2 * h**2)


# Initial Condition
u = np.zeros([m+1, n+1])
for i in range(0, m+1):
    # u[i, 0] = np.sin(np.pi*x[i])
    u[i, 0] = np.sin(np.pi*x[i])
    
#Boundary Condition
for j in range(1, n):
    u[0, j] = 0
    u[m, j] = 0
    
li = np.zeros(m-2) 
di = np.zeros(m-1) 
ui = np.zeros(m-2) 
b = np.zeros(m-1) 
C = np.zeros(m-1)


for i in range(0, m-2):  
    li[i] = R
    ui[i] = P
for i in range(0, m-1):
    di[i] = Q
A = diags([li, di, ui], [-1, 0, 1], shape=(m-1, m-1)).toarray()

for j in range(0, n):

    C[0] = R*u[0,j]          
    C[-1] = P*u[m,j]
    # print("old",C)          
    for i in range(1, m):  
           # b[i-1] = Pb*u[i+1, j] + Qb*u[i, j] + Rb*u[i-1, j]
            b[i-1] = Pb*u[i+1, j] + Qb*u[i, j] + Rb*u[i-1, j]- C[i-1]
    # print("b",b)
    vi = np.linalg.solve(A, b)
    # print("V",vi)
    for i in range(1,m):
        u[i,j+1]=np.take(vi,i-1)
    #u = np.round(u, 4)
    # print("old",u)  
   

v = np.zeros([m+1, n+1])
err = np.zeros([m+1, n+1])

for i in range(m+1):
    for j in range(n+1):
        v[i, j] = np.e**(-t[j])*np.sin(np.pi * x[i])
        
        # v[i, j] = np.e**(-(np.pi**2 )* t[j])*np.sin(np.pi*x[i])
        err[i, j] = np.abs(u[i, j] - v[i, j])
        
error = err.max()
print(error)
# print(np.round(error,14))
     

# plt.rcParams["font.family"] = "Times New Roman"  # Set Times New Roman font
# # Create meshgrid for 3D plotting
# X, T = np.meshgrid(x, t)

# fig = plt.figure(figsize=(6,5))
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(X, T, u.T, cmap='hot', edgecolor='k', alpha= 0.7)

# ax.set_xlabel("Spatial Coordinate x", fontsize=12)
# ax.set_ylabel("Time t", fontsize=12)
# ax.set_zlabel("$u(x,t)$", fontsize=12)
# # ax.set_title("3D Surface Plot: Numerical Solution", fontsize=14)

# ax.view_init(elev=10 , azim= 310)  # Change viewpoint
# # plt.savefig("numerical p1", dpi=600)
# plt.show()

# # # Exact Solution with Different View
# # fig = plt.figure(figsize=(6, 5))
# # ax = fig.add_subplot(projection='3d')
# # ax.plot_surface(X, T, v.T, cmap='coolwarm', edgecolor='k', alpha=0.7)

# # ax.set_xlabel("Spatial Coordinate x", fontsize=12)
# # ax.set_ylabel("Time t", fontsize=12)
# # ax.set_zlabel(" $u(x,t)$", fontsize=12)
# # # ax.set_title("3D Surface Plot: Exact Solution", fontsize=14)

# # ax.view_init(elev=10, azim=310)  # Another viewpoint
# # # plt.savefig("exact p1", dpi=600)
# # plt.show()


