import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("input.csv")

processes = sorted(df["P"].unique())
data_sizes = sorted(df["M"].unique())

fig, axes = plt.subplots(1, len(data_sizes), figsize=(12,6), sharey=True)

if len(data_sizes) == 1:
    axes = [axes]

for i, M in enumerate(data_sizes):
    ax = axes[i]
    box_data = []
    
    for P in processes:
        times = df[(df["P"] == P) & (df["M"] == M)]["time"]
        box_data.append(times)
    
    ax.boxplot(box_data)
    ax.set_xticklabels(processes)
    ax.set_xlabel("Processes (P)")
    ax.set_title(f"M = {M}")

axes[0].set_ylabel("Time (seconds)")
plt.suptitle("Execution Time vs Processes")
plt.tight_layout()
plt.savefig("boxplot.pdf")
plt.show()
