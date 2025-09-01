import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Rectangle
true_N0 = 10000
true_r = 0.1

df = pd.read_csv("rep32.log", sep="\t", comment="#")
posterior = df[int(len(df)*0.1):]  # burn-in 10%


N0 = posterior["ePopSize"].values / 2
r  = posterior["growthRate"].to_numpy(dtype=float)
alpha = np.log(N0)


if "Tree.height" in posterior.columns:
    H = np.nanpercentile(posterior["Tree.height"].to_numpy(), 95)  # 用树高 95% 分位数做上界
else:
    H = max(1.0, np.median(N0) / max(np.median(r[r>0]), 1e-6))
T = np.linspace(0.0, H, 400)
print(T)


TT = T[:, None]      # (400,1)
RR = r[None, :]      # (1, M)
Z  = alpha[None, :] - RR * TT   # log N(t)
var_z = np.var(Z, axis=1, ddof=1)


t_star = T[np.argmin(var_z)]
print("Global t* (min variance) =", t_star)


plt.figure(figsize=(7,4))
plt.plot(T, var_z, label="Var[log N(t)]")
plt.axvline(t_star, linestyle="--", color="red", label=f"t*={t_star:.2f}")
plt.xlabel("Time before present")
plt.ylabel("Var[ log N(t) ]")
plt.title("Variance of log N(t) over time")
plt.legend()
plt.show()




time_windows = [(0, 20), (20, 40), (40, 60),(60, 80)]
plt.figure(figsize=(6,5))
plt.scatter(alpha, r, s=4, alpha=0.2, label="posterior samples")

for (tmin, tmax) in time_windows:
    mask = (T >= tmin) & (T <= tmax)
    if not np.any(mask):
        continue
    t_loc = T[mask][np.argmin(var_z[mask])]
    c = np.median(alpha - r*t_loc)
    rr_line = np.linspace(np.min(r), np.max(r), 200)
    aa_line = c + rr_line*t_loc
    plt.plot(aa_line, rr_line, linewidth=2,
             label=f"ridge [{tmin},{tmax}] t*={t_loc:.1f}")

plt.xlabel("alpha = log N0")
plt.ylabel("r (growthRate)")
plt.title("Posterior ridge lines in (alpha,r)")
plt.legend()
plt.show()



