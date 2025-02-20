# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# Load data from CSV file
file_path = "./results/gaussian_v1/K/results_K.csv"  

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
        df["K"], df[term], label=labels[i], color=colors[i],
        linestyle=dash_patterns[i], linewidth=1.5, marker='s', markersize=6, alpha=0.6
    )

theoretical_terms = ["theoretical TV bound of single", "theoretical TV bound of multi"]
for i, term in enumerate(theoretical_terms, start=2):  
    ax_right.plot(
        df["K"], df[term], label=labels[i], color=colors[i],
        linestyle=dash_patterns[i], linewidth=1.5, marker='d', markersize=7, alpha=0.7
    )

ax_left.set_xlabel("(a) Number of Sources", fontsize=14)
ax_left.set_ylabel("Empirical TV Error", fontsize=14, color='black')
ax_left.tick_params(axis='both', which='major', labelsize=12)

left_min = df[empirical_terms].min().min()
left_max = df[empirical_terms].max().max()
left_interval = left_max - left_min
ax_left.set_ylim(left_min - 0.1 * left_interval, left_max + 0.1 * left_interval)
ax_left.set_yticks([0.05, 0.10, 0.15, 0.2])
ax_left.set_xticks(df["K"])
ax_left.grid(False)

ax_right.set_ylabel("Theoretical Bound", fontsize=14, color='black')
ax_right.tick_params(axis='y', which='major', labelsize=12)

rigth_min = df[theoretical_terms].min().min()
rigth_max = df[theoretical_terms].max().max()
rigth_interval = rigth_max - rigth_min
ax_right.set_ylim(rigth_min - 0.1 * rigth_interval, rigth_max + 0.1 * rigth_interval)
ax_right.set_yticks([0.15, 0.25, 0.35, 0.45, 0.55])


lines_left, labels_left = ax_left.get_legend_handles_labels()
ax_left.legend(
    lines_left, labels_left,
    loc="upper left", fontsize=14, 
    frameon=True,       
    facecolor="white",  
    edgecolor="none",   
    framealpha=0.6,   
)

lines_right, labels_right = ax_right.get_legend_handles_labels()
ax_right.legend(
    lines_right, labels_right,
    loc="lower right", fontsize=14,
    frameon=True,    
    facecolor="white",  
    edgecolor="none",   
    framealpha=0.6,   
)

ax_left.grid(False)
ax_right.grid(False)

os.makedirs("./figures", exist_ok=True)
plt.savefig("./figures/gaussian_K.png", bbox_inches='tight', pad_inches=0.03)
plt.savefig("./figures/gaussian_K.svg", bbox_inches='tight', pad_inches=0.03)
plt.show()



# %%
