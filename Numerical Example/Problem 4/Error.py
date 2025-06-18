import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set font style
mpl.rcParams['font.family'] = 'Times New Roman'

# Grid sizes (decimal) and labels
h_vals = np.array([1/16, 1/32, 1/64, 1/128])
h_labels = [r'$\frac{1}{16}$', r'$\frac{1}{32}$', r'$\frac{1}{64}$', r'$\frac{1}{128}$']

# Error data from Table 4.4 and 4.5
errors_data = {
    "Pe = 10":    [8.8239e-5, 5.5792e-6, 3.4715e-7, 2.1691e-8],
    "Pe = 20":    [1.4970e-3, 8.7430e-5, 5.5135e-6, 3.4308e-7],
    "Pe = 100":   [1.6741e-1, 3.3796e-2, 3.4567e-3, 2.1290e-4],
    "Pe = 1000":  [7.5202e-1, 5.1300e-1, 2.4073e-1, 6.1102e-2]
}

# Different markers and colors
markers = ['o', 's', 'D', '^']
colors = ['g', 'b', 'k', 'r']

# Plot
plt.figure(figsize=(10, 6))
for (label, errors), marker, color in zip(errors_data.items(), markers, colors):
    plt.plot(h_vals, errors, linestyle='--', marker=marker, color=color, label=label,
             linewidth=1.5, markersize=7)

# Reference line for 4th order (slope = 4)
ref_x = np.array([1/16, 1/128])
ref_y4 = 1e-3 * (ref_x / ref_x[0])**4
plt.plot(ref_x, ref_y4, 'm-', linewidth=1.5, label=r'$\mathcal{O}(h^4)$')
plt.text(ref_x[1], ref_y4[1]*1.5, r'$\mathcal{O}(h^4)$', fontsize=12, color='k')

# Reference line for 2nd order (slope = 2)
ref_y2 = 1e-2 * (ref_x / ref_x[0])**2
plt.plot(ref_x, ref_y2, 'm:', linewidth=1.5, label=r'$\mathcal{O}(h^2)$')
plt.text(ref_x[1], ref_y2[1]*1.5, r'$\mathcal{O}(h^2)$', fontsize=12, color='k')

# Plot formatting
plt.xticks(h_vals, h_labels, fontsize=11)
plt.yscale('log')
plt.xlabel('Decreasing $h$', fontsize=12)
plt.ylabel('Max Absolute Error', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title='$P_e$', fontsize=10)
plt.gca().invert_xaxis()
plt.tight_layout()
# plt.savefig("error_graph_with_ref_lines", dpi=700)
# plt.savefig("error graph p4", dpi=700)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set font style
mpl.rcParams['font.family'] = 'Times New Roman'

# Grid sizes starting from 1/32 for order plot
h_vals_order = np.array([1/32, 1/64, 1/128])
h_labels_order = [r'$\frac{1}{32}$', r'$\frac{1}{64}$', r'$\frac{1}{128}$']

# Order data from Table 4.4 and 4.5 (excluding first None)
orders_data = {
    "Pe = 20":    [4.00, 4.00, 4.00],
    "Pe = 10":    [3.98, 4.00, 4.00],
    "Pe = 100":   [2.38, 3.30, 4.00],
    "Pe = 1000":  [0.55, 1.09, 2.00]
}

# Different markers and colors (matching error plot)
markers = ['o', 's', 'D', '^']
colors = ['b', 'g', 'k', 'r']

plt.figure(figsize=(10, 6))

for (label, orders), marker, color in zip(orders_data.items(), markers, colors):
    plt.plot(h_vals_order, orders, linestyle='--', marker=marker, color=color,
             label=label, linewidth=1.5, markersize=7)

plt.xticks(h_vals_order, h_labels_order, fontsize=11)
plt.xlabel('Decreasing $h$', fontsize=12)
plt.ylabel('Order of Convergence', fontsize=12)
# plt.title('Order of Convergence vs Grid Size for Problem-4 (Different $P_e$)', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(title='$P_e$', fontsize=10)
plt.gca().invert_xaxis()
plt.tight_layout()
# plt.savefig("order graph p4", dpi=700)
plt.show()


