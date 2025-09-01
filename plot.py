import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.style.use("default")
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")


skyline = pd.read_csv("1_table", sep="\t")  # 替换为你的实际文件名

print(skyline.columns.tolist())


height = skyline["time"].values
median = skyline["median"].values
lower = skyline["lower"].values
upper = skyline["upper"].values


edges = [0, 2000, 2700, height.max()]
values = [10000, 500, 10000]
true_time = []
true_ne = []
for i in range(len(values)):
    true_time += [edges[i], edges[i+1]]
    true_ne  += [values[i], values[i]]

true_time = np.array(true_time)
true_ne = np.array(true_ne)


plt.figure(figsize=(9, 6))


plt.fill_between(height, lower, upper, color="skyblue", alpha=0.3, label="95% HPD (Posterior)")


plt.plot(height, median, color="black", linewidth=2, label="Posterior Median")

plt.plot(true_time, true_ne, color="red", linestyle="--", linewidth=2, label="True Ne (piecewise)")


plt.xlabel("Time (generations)")
plt.ylabel("Effective Population Size")
plt.yscale("log")
plt.title("Bayesian Skyline Plot with True Ne Overlay")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
