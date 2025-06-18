# import numpy as np
# import matplotlib.pyplot as plt

# # Grid sizes and labels
# h_vals = np.array([1/8, 1/16, 1/32, 1/64, 1/128])
# h_labels = [r'$\frac{1}{8}$', r'$\frac{1}{16}$', r'$\frac{1}{32}$', r'$\frac{1}{64}$', r'$\frac{1}{128}$']

# # Errors and orders from the table (Problem-3)
# errors = np.array([3.0497e-4, 1.9066e-5, 1.1919e-6, 7.4495e-8, 4.6559e-9])
# orders = [None, 3.98, 4.00, 4.00, 4.00]

# # Create separate figures for log-log error plot and semi-log order plot

# # Log-Log Error Plot
# plt.figure(figsize=(8, 5))
# plt.loglog(h_vals, errors, 'ro--', linewidth=1.5, markersize=6)
# plt.xticks(h_vals, h_labels)
# plt.xlabel('Decreasing $h$', fontsize=12)
# plt.ylabel('Max Absolute Error', fontsize=12)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.gca().invert_xaxis()
# # plt.title('Log-Log Error Plot for Problem - 3')
# plt.tight_layout()

# plt.show()




import numpy as np
import matplotlib.pyplot as plt

# Grid sizes and labels
h_vals = np.array([1/8, 1/16, 1/32, 1/64, 1/128])
h_labels = [r'$\frac{1}{8}$', r'$\frac{1}{16}$', r'$\frac{1}{32}$', r'$\frac{1}{64}$', r'$\frac{1}{128}$']

# Errors and orders from the table (Problem-3)
errors = np.array([3.0497e-4, 1.9066e-5, 1.1919e-6, 7.4495e-8, 4.6559e-9])
orders = [None, 3.98, 4.00, 4.00, 4.00]

# Create log-log error plot
plt.figure(figsize=(8, 5))
plt.loglog(h_vals, errors, 'ro--', label='Numerical Error', linewidth=1.5, markersize=6)

# Reference line for 4th-order convergence (anchor at first point)
ref_line = errors[0] * (h_vals / h_vals[0])**4
plt.loglog(h_vals, ref_line, 'k:.', label=r'Ref: $\mathcal{O}(h^4)$')

# Annotate order line
plt.text(h_vals[-1]*1.1, ref_line[-1]*1.5, r'$\mathcal{O}(h^4)$', fontsize=11)

# Formatting
plt.xticks(h_vals, h_labels)
plt.xlabel('Decreasing $h$', fontsize=12)
plt.ylabel('Max Absolute Error', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig("error_p5_with_ref_line", dpi=600)
# plt.savefig("error p5", dpi=600)
plt.show()


# # Order Plot (Linear x, Linear y)
# plt.figure(figsize=(8, 5))
# plt.plot(h_vals[1:], orders[1:], 'bs--', linewidth=1.5, markersize=6)
# plt.xticks(h_vals[1:], h_labels[1:])
# plt.xlabel('Decreasing $h$', fontsize=12)
# plt.ylabel('Order of Convergence', fontsize=12)
# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.gca().invert_xaxis()
# plt.ylim(3.5, 4.1)
# # plt.title('Order of Convergence for Problem - 3')
# plt.tight_layout()
# # plt.savefig("order p5", dpi=600)
# plt.show()

