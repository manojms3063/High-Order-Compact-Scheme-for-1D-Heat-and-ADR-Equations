import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use Times New Roman font
mpl.rcParams['font.family'] = 'Times New Roman'

# Grid sizes and labels (h = k^2)
h_vals = np.array([1/5, 1/10, 1/20, 1/40, 1/80, 1/160])
h_labels = [r'$\frac{1}{5}$', r'$\frac{1}{10}$', r'$\frac{1}{20}$',
            r'$\frac{1}{40}$', r'$\frac{1}{80}$', r'$\frac{1}{160}$']

# CN method errors and orders
cn_errors = np.array([3.9412e-5, 9.6487e-6, 2.3484e-6, 5.8290e-7, 1.4545e-7, 3.6347e-8])

# CF-1 method errors and orders
cf1_errors = np.array([4.3147e-3, 2.8390e-4, 1.7730e-5, 1.1082e-6, 6.9261e-8, 4.3289e-9])

# Create the plot
plt.figure(figsize=(8, 5))
plt.loglog(h_vals, cn_errors, 'b-o', label='CN Error', linewidth=1.5, markersize=6)
plt.loglog(h_vals, cf1_errors, 'rs--', label='CF-1 Error', linewidth=1.5, markersize=6)

# Reference lines
ref_h = np.array([1/5, 1/160])
ref_cn = cn_errors[0] * (ref_h / h_vals[0])**2   # Slope 2
ref_cf = cf1_errors[0] * (ref_h / h_vals[0])**4  # Slope 4

plt.loglog(ref_h, ref_cn, 'k--', label=r'Ref: $\mathcal{O}(h^2)$')
plt.loglog(ref_h, ref_cf, 'k:', label=r'Ref: $\mathcal{O}(h^4)$')

# Annotations
plt.text(ref_h[1], ref_cn[1]*2, r'$\mathcal{O}(h^2)$', fontsize=11)
plt.text(ref_h[1], ref_cf[1]*1.5, r'$\mathcal{O}(h^4)$', fontsize=11)

# Axes and labels
plt.xticks(h_vals, h_labels)
plt.xlabel('Decreasing $h$', fontsize=12)
plt.ylabel('Max Absolute Error', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()
# plt.savefig("error_comparison_with_refs.png", dpi=700)
plt.show()

# # Plot 2: Convergence Order vs h (reverse x-axis to show refinement left to right)
# plt.figure(figsize=(8, 5))
# plt.plot(h_vals[1:], cn_orders[1:], 'b-o', label='CN Order', linewidth=1.5, markersize=6)
# plt.plot(h_vals[1:], cf1_orders[1:], 'rs--', label='CF-1 Order', linewidth=1.5, markersize=6)
# plt.xticks(h_vals[1:], h_labels[1:])
# plt.gca().invert_xaxis()
# plt.xlabel('Decreasing $h$', fontsize=12)
# plt.ylabel('Convergence Order', fontsize=12)
# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.legend()
# plt.tight_layout()
# # plt.savefig("order_comparison.png", dpi=700)
# plt.show()
