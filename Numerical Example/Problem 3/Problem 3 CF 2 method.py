import numpy as np
import sympy as smp
from sympy import symbols, sin, exp, diff, lambdify
from scipy.sparse import diags
import scipy
import matplotlib.pyplot as plt

# Parameters
# m = 2**5
# n = 2**2
h = 1/16
k = h**2
x0 = 0
xm = 1

T = 2
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

alpha = 0.1
beta = 1

# Define coefficients
S1 = -((alpha/h**2) + (beta**2 / (12 * alpha)))
S2 = beta / (2 * h)
S3 = 1 / 12
S4 = - (beta * h) / (24 * alpha)

a1 = S3 - S4
a2 = 1 - 2*S3
a3 = S3 + S4
a4 = -(S1 - S2)
a5 = 2*S1
a6 = -(S1 + S2)

P = -((S1+S2)/2)-((S3+S4)/k)
Pb = ((S1+S2)/2)-((S3+S4)/k)

Q = 2*S1/2 - ((1-2*S3)/k)
Qb = -2*S1/2 - ((1-2*S3)/k)

R = -(S1-S2)/2 - ((S3-S4)/k)
Rb = (S1-S2)/2 - ((S3-S4)/k)


# Initial Condition
u = np.zeros([m+1, n+1])
for i in range(0, m+1):
    u[i, 0] = np.e**(5*x[i])*(np.cos((np.pi/2)*x[i])+0.25*np.sin((np.pi/2)*x[i]))
    
#Boundary Condition
for j in range(1, n+1):
    u[0, j] = np.exp(5 * (0 - t[j] / 2)) * np.exp(- (np.pi**2 / 40) * t[j]) *(np.cos((np.pi / 2) * 0) + 0.25 * np.sin((np.pi / 2) * 0))
    u[m, j] = np.exp(5 * (1 - t[j] / 2)) * np.exp(- (np.pi**2 / 40) * t[j]) *(np.cos((np.pi / 2) * 1) + 0.25 * np.sin((np.pi / 2) * 1))
    
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
    C[0] = R*u[0,j+1]          
    C[-1] = P*u[m,j+1]
    # print(C)
    # print("old",C)          
    for i in range(1, m):  
           # b[i-1] = Pb*u[i+1, j] + Qb*u[i, j] + Rb*u[i-1, j]
            b[i-1] = Pb*u[i+1, j] + Qb*u[i, j] + Rb*u[i-1, j] - C[i-1]
           # b[i-1] = b[i-1] - C[i-1]
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
        v[i, j] = np.exp(5 * (x[i] - t[j] / 2)) * np.exp(- (np.pi**2 / 40) * t[j]) *(np.cos((np.pi / 2) * x[i]) + 0.25 * np.sin((np.pi / 2) * x[i]))
        err[i, j] = np.abs(u[i, j] - v[i, j])
        
error = err.max()
print(error)

# # print(np.round(u,4))
# Create meshgrid for 3D plotting
X, T = np.meshgrid(x, t)

# 3D Surface Plot of Numerical Solution
fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X, T, u.T, cmap='hot', edgecolor='k', alpha=0.7)
ax1.set_xlabel("Spatial Coordinate x")
ax1.set_ylabel("Time t")
ax1.set_zlabel("$u(x,t)$")

ax1.view_init(elev=30, azim= 100)  # Change viewpoint
# plt.savefig("numerical p5", dpi=600)
# ax1.set_title("Numerical Solution")
plt.show()

# 3D Surface Plot of Exact Solution
fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X, T, v.T, cmap='coolwarm', edgecolor='k', alpha=0.7)
ax2.set_xlabel("Spatial Coordinate x")
ax2.set_ylabel("Time t")
ax2.set_zlabel("$v(x,t)$")

ax2.view_init(elev=30, azim= 100)  # Change viewpoint
# plt.savefig("Exact p5", dpi=600)
# ax2.set_title("Exact Solution")
plt.show()









# # Create meshgrid for 3D plotting
# X, T = np.meshgrid(x, t)

# # 3D Surface Plot of Numerical Solution 
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, T, u.T, cmap='viridis', edgecolor='k')

# ax.set_xlabel("Spatial Coordinate x")
# ax.set_ylabel("Time t")
# ax.set_zlabel("Solution u(x,t)")
# ax.set_title("Numerical Solution of PDE")
# plt.show()


# # Comparison of Numerical vs Exact Solution 
# fig, ax = plt.subplots(figsize=(8, 5))
# for j in range(n+1):
#     ax.plot(x, u[:, j], 'o-', label=f"Numerical (t={t[j]:.2f})")
#     ax.plot(x, v[:, j], '--', label=f"Exact (t={t[j]:.2f})")

# ax.set_xlabel("Spatial Coordinate x")
# ax.set_ylabel("Solution u(x,t)")
# ax.set_title("Comparison of Numerical and Exact Solutions")
# # ax.legend()
# plt.show()


# # Error Heatmap 
# fig, ax = plt.subplots(figsize=(8, 5))
# c = ax.contourf(X, T, err.T, cmap="coolwarm", levels=50)
# plt.colorbar(c, label="Error Magnitude")

# ax.set_xlabel("Spatial Coordinate x")
# ax.set_ylabel("Time t")

# ax.set_title("Error Distribution (Numerical vs Exact)")
# plt.show()


