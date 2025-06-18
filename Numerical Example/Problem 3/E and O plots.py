# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl

# # Set font style
# mpl.rcParams['font.family'] = 'Times New Roman'

# # Grid size values from table (converted to decimals)
# h_vals = np.array([1/16, 1/32, 1/64, 1/128])
# h_labels = [r'$\frac{1}{16}$', r'$\frac{1}{32}$', r'$\frac{1}{64}$', r'$\frac{1}{128}$']

# # Error and Order values from Table 4.3
# errors = np.array([3.0308e-4, 1.9066e-5, 1.1941e-6, 7.4672e-8])
# orders = [None, 3.98, 4.00, 4.00]


# plt.figure(figsize=(8, 5))
# plt.loglog(h_vals, errors, 'ro--', label='Problem-3 Error', linewidth=1.5, markersize=8)
# plt.xticks(h_vals, h_labels)
# plt.xlabel('Decreasing $h$', fontsize=12)
# plt.ylabel('Max Absolute Error', fontsize=12)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.legend()

# plt.gca().invert_xaxis()
# plt.tight_layout()
# plt.title('Error vs Grid Size for Problem - 3')

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set font style
mpl.rcParams['font.family'] = 'Times New Roman'

# Grid size values from table
h_vals = np.array([1/16, 1/32, 1/64, 1/128])
h_labels = [r'$\frac{1}{16}$', r'$\frac{1}{32}$', r'$\frac{1}{64}$', r'$\frac{1}{128}$']

# Error values from Table 4.3
errors = np.array([3.0308e-4, 1.9066e-5, 1.1941e-6, 7.4672e-8])
orders = [None, 3.98, 4.00, 4.00]

# Plot Error
plt.figure(figsize=(8, 5))
plt.loglog(h_vals, errors, 'ro--', label='CF-2 Error', linewidth=1.5, markersize=8)

# Reference Line: 4th-order (pass through first point)
ref_line = errors[0] * (h_vals / h_vals[0])**4
plt.loglog(h_vals, ref_line, 'k:.', label=r'Ref: $\mathcal{O}(h^4)$')

# Annotate the slope
plt.text(h_vals[-1]*1.1, ref_line[-1]*2, r'$\mathcal{O}(h^4)$', fontsize=11)

# Formatting
plt.xticks(h_vals, h_labels)
plt.xlabel('Decreasing $h$', fontsize=12)
plt.ylabel('Max Absolute Error', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.gca().invert_xaxis()
# plt.title('Error vs Grid Size for Problem - 3', fontsize=13)
plt.tight_layout()

# Save and Show
plt.savefig("Error p3", dpi=700)
plt.show()


# # Plot 2: Convergence Order vs h
# plt.figure(figsize=(8, 5))
# plt.plot(h_vals[1:], orders[1:], 'bs--', label='Problem-3 Order', linewidth=1.5, markersize=8)
# plt.xticks(h_vals[1:], h_labels[1:])
# plt.xlabel('Decreasing $h$', fontsize=12)
# plt.ylabel('Convergence Order', fontsize=12)
# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.legend()
# plt.gca().invert_xaxis()
# plt.tight_layout()
# # plt.title('Convergence Order vs Grid Size for Problem - 3')
# plt.savefig("order p3", dpi=700)
# plt.show()
