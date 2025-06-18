# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl

# # Use Times New Roman font if available
# mpl.rcParams['font.family'] = 'Times New Roman'

# # Grid sizes and LaTeX labels
# h_vals = np.array([1/5, 1/10, 1/20, 1/40, 1/80, 1/160])
# h_labels = [r'$\frac{1}{5}$', r'$\frac{1}{10}$', r'$\frac{1}{20}$',
#             r'$\frac{1}{40}$', r'$\frac{1}{80}$', r'$\frac{1}{160}$']

# # Errors and orders (from image)
# cn_errors = np.array([2.5764e-2, 6.7221e-3, 1.6772e-3, 4.1909e-4, 1.0476e-4, 2.6190e-5])
# cn_orders = [None, 1.94, 2.00, 2.00, 2.00, 2.00]

# cf1_errors = np.array([1.8398e-4, 1.1923e-5, 7.4250e-7, 4.6364e-8, 2.8970e-9, 1.8059e-10])
# cf1_orders = [None, 3.95, 4.00, 4.00, 4.00, 4.00]

# cf2_errors = np.array([1.8399e-4, 1.1923e-5, 7.4250e-7, 4.6364e-8, 2.8970e-9, 1.8059e-10])
# cf2_orders = [None, 3.95, 4.00, 4.00, 4.00, 4.00]

# # Plot 1: Error vs h (log-log plot)
# plt.figure(figsize=(8, 5))
# plt.loglog(h_vals, cn_errors, 'go-', label='CN Error', linewidth=1.5, markersize=6)
# plt.loglog(h_vals, cf1_errors, 'rs--', label='CF-1 Error', linewidth=3, markersize=9)
# plt.loglog(h_vals, cf2_errors, 'b^-.', label='CF-2 Error', linewidth=1.5, markersize=6,)
# plt.xticks(h_vals, h_labels)
# plt.xlabel('Decreasing $h$', fontsize=12)
# plt.ylabel('Max Absolute Error', fontsize=12)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.legend()
# plt.gca().invert_xaxis()
# plt.tight_layout()

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use Times New Roman font
mpl.rcParams['font.family'] = 'Times New Roman'

# Grid sizes and LaTeX labels
h_vals = np.array([1/5, 1/10, 1/20, 1/40, 1/80, 1/160])
h_labels = [r'$\frac{1}{5}$', r'$\frac{1}{10}$', r'$\frac{1}{20}$',
            r'$\frac{1}{40}$', r'$\frac{1}{80}$', r'$\frac{1}{160}$']

# Error arrays
cn_errors = np.array([2.5764e-2, 6.7221e-3, 1.6772e-3, 4.1909e-4, 1.0476e-4, 2.6190e-5])
cf1_errors = np.array([1.8398e-4, 1.1923e-5, 7.4250e-7, 4.6364e-8, 2.8970e-9, 1.8059e-10])
cf2_errors = np.array([1.8399e-4, 1.1923e-5, 7.4250e-7, 4.6364e-8, 2.8970e-9, 1.8059e-10])

# Start plot
plt.figure(figsize=(8, 5))

# Plot errors
plt.loglog(h_vals, cn_errors, 'go-', label='CN Error', linewidth=1.5, markersize=6)
plt.loglog(h_vals, cf1_errors, 'rs--', label='CF-1 Error', linewidth=3, markersize=9)
plt.loglog(h_vals, cf2_errors, 'b^-.', label='CF-2 Error', linewidth=1.5, markersize=6)

# Reference lines
ref_h = np.array([1/5, 1/160])
ref_cn = cn_errors[0] * (ref_h / h_vals[0])**2    # slope 2
ref_cf = cf1_errors[0] * (ref_h / h_vals[0])**4   # slope 4

plt.loglog(ref_h, ref_cn, 'k--', label=r'Ref: $\mathcal{O}(h^2)$')
plt.loglog(ref_h, ref_cf, 'k:', label=r'Ref: $\mathcal{O}(h^4)$')

# Annotate slopes
plt.text(ref_h[1], ref_cn[1]*1.5, r'$\mathcal{O}(h^2)$', fontsize=11)
plt.text(ref_h[1], ref_cf[1]*2, r'$\mathcal{O}(h^4)$', fontsize=11)

# Styling
plt.xticks(h_vals, h_labels)
plt.xlabel('Decreasing $h$', fontsize=12)
plt.ylabel('Max Absolute Error', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()

# Show or save
plt.savefig("Error P2", dpi=700)
plt.show()


# # Plot 2: Convergence Order vs h
# plt.figure(figsize=(8, 5))
# plt.plot(h_vals[1:], cn_orders[1:], 'go-', label='CN Order', linewidth=1.5, markersize=6)
# plt.plot(h_vals[1:], cf1_orders[1:], 'rs--', label='CF-1 Order', linewidth=3, markersize=8)
# plt.plot(h_vals[1:], cf2_orders[1:], 'b^-.', label='CF-2 Order', linewidth=1.5, markersize=6)
# plt.xticks(h_vals[1:], h_labels[1:])
# plt.xlabel('Decreasing $h$', fontsize=12)
# plt.ylabel('Convergence Order', fontsize=12)
# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.legend()
# plt.gca().invert_xaxis()
# plt.tight_layout()
# # plt.savefig("order P2", dpi=700)
# plt.show()
