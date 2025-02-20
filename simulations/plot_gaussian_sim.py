# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# Load data from CSV file
file_path = "./results/gaussian_v1/sim/results_sim.csv"  

df = pd.read_csv(file_path)

# Set the plotting style
plt.style.use('science')
plt.rcParams['text.usetex'] = False

# Prepare figure and axis
fig, ax_left = plt.subplots(figsize=(4, 3))
ax_right = ax_left.twinx() 

colors = ['#FF9900', '#8CD282', '#FF7F50', '#8FBC8F']
dash_patterns = ['-', '-', '--', '--']
labels = ['Empirical (Single)', 'Empirical (Multi)',
          'Theoretical (Single)', 'Theoretical (Multi)']

empirical_terms = ["empirical TV of single", "empirical TV of multi"]
for i, term in enumerate(empirical_terms):
    ax_left.plot(
        df["sim"], df[term], label=labels[i], color=colors[i],
        linestyle=dash_patterns[i], linewidth=1.5, marker='s', markersize=6, alpha=0.6
    )

theoretical_terms = ["theoretical TV bound of single", "theoretical TV bound of multi"]
for i, term in enumerate(theoretical_terms, start=2):  
    ax_right.plot(
        df["sim"], df[term], label=labels[i], color=colors[i],
        linestyle=dash_patterns[i], linewidth=1.5, marker='d', markersize=7, alpha=0.7
    )

ax_left.set_xlabel("(c) Distribution Similarity", fontsize=14)
ax_left.set_ylabel("Empirical TV Error", fontsize=14, color='black')
ax_left.tick_params(axis='both', which='major', labelsize=12)
left_min = df[empirical_terms].min().min()
left_max = df[empirical_terms].max().max()
left_interval = left_max - left_min
ax_left.set_ylim(left_min - 0.1 * left_interval, left_max + 0.1 * left_interval)
ax_left.grid(False)

ax_right.set_ylabel("Theoretical Bound", fontsize=14, color='black')
ax_right.tick_params(axis='y', which='major', labelsize=12)
rigth_min = df[theoretical_terms].min().min()
rigth_max = df[theoretical_terms].max().max()
rigth_interval = rigth_max - rigth_min
ax_right.set_ylim(rigth_min - 0.1 * rigth_interval, rigth_max + 0.1 * rigth_interval)

ax_left.set_xticks(df["sim"])
ax_left.set_yticks([0.05, 0.07, 0.09, 0.11])
ax_right.set_yticks([0.15, 0.20, 0.25, 0.30])

ax_left.grid(False)
ax_right.grid(False)

lines_left, labels_left = ax_left.get_legend_handles_labels()
legend_left = ax_left.legend(
    lines_left, labels_left,
    loc="upper left", fontsize=14, 
    frameon=True,       
    facecolor="white",  
    edgecolor="none",   
    framealpha=0.6,    
    bbox_to_anchor=(-0.01, 0.92),
)
legend_left.set_zorder(10)

lines_right, labels_right = ax_right.get_legend_handles_labels()
legend_right = ax_right.legend(
    lines_right, labels_right,
    loc="lower right", fontsize=14,
    frameon=True,       
    facecolor="white",  
    edgecolor="none",   
    framealpha=0.6,     
)
legend_right.set_zorder(10)
ax_right.set_zorder(1) 


os.makedirs("./figures", exist_ok=True)
plt.savefig("./figures/gaussian_sim.png", bbox_inches='tight', pad_inches=0.03)
plt.savefig("./figures/gaussian_sim.svg", bbox_inches='tight', pad_inches=0.03)
fig.canvas.draw() 

plt.show()


# %%
