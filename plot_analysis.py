# ====== 依赖 ======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy.stats import gaussian_kde
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

plt.style.use("default")


true_N0 = 10000
true_r  = 0.1


log_df = pd.read_csv("rep32.log", sep="\t", comment="#")
posterior = log_df[int(len(log_df)*0.1):]      # 10% burn-in

# 取参数（按你的列名）
N0 = posterior["ePopSize"].to_numpy() / 2
r  = posterior["growthRate"].to_numpy()
tree_height = posterior["Tree.height"].to_numpy()


tmrca_lo = az.hdi(tree_height, hdi_prob=0.95)[0]
t_grid = np.linspace(0, tmrca_lo, 500)

# ====== 传播不确定性：Ne(t) = N0 * exp(-r t) ======
logN0 = np.log(N0)
logNe_mat = logN0[:, None] - r[:, None] * t_grid[None, :]
Ne_mat = np.exp(logNe_mat)

Ne_med = np.median(Ne_mat, axis=0)
Ne_lo, Ne_hi = np.quantile(Ne_mat, [0.025, 0.975], axis=0)
Ne_true = true_N0 * np.exp(-true_r * t_grid)


band_width = Ne_hi - Ne_lo
t_star_emp = t_grid[np.argmin(band_width)]


var_logN0 = np.var(logN0, ddof=1)
var_r     = np.var(r, ddof=1)
cov_logN0_r = np.cov(logN0, r, bias=False)[0, 1]
t_star_ana = cov_logN0_r / var_r if var_r > 0 else np.nan
t_star_ana = float(np.clip(t_star_ana, t_grid[0], t_grid[-1]))


def hpd_levels_from_kde(zz, probs=(0.95, 0.50)):
    z = zz.ravel()
    order = np.argsort(z)[::-1]
    z_sorted = z[order]
    csum = np.cumsum(z_sorted) / z_sorted.sum()
    levels = []
    for p in probs:
        idx = np.searchsorted(csum, p)
        levels.append(z_sorted[idx])
    return levels

x = np.linspace(N0.min()*0.8, N0.max()*1.2, 320)
y = np.linspace(r.min()*0.8,  r.max()*1.2,  320)
XX, YY = np.meshgrid(x, y)

kde = gaussian_kde(np.vstack([N0, r]))
ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

lev95, lev50 = hpd_levels_from_kde(ZZ, probs=(0.95, 0.50))

plt.figure(figsize=(8, 6))


plt.contourf(XX, YY, ZZ, levels=60, cmap="Blues", alpha=0.15, zorder=1)

plt.contour(XX, YY, ZZ, levels=[lev50],
            colors="gold", linewidths=2.5, zorder=4)

plt.contour(XX, YY, ZZ, levels=[lev95],
            colors="crimson", linewidths=2.5, linestyles="--", zorder=5)

plt.scatter(N0, r, alpha=0.05, s=8, color="black", zorder=2, label="Posterior samples")
med_N0, med_r = np.median(N0), np.median(r)

plt.scatter(med_N0, med_r, color="blue", s=70, edgecolor="k", zorder=6, label="Posterior median")


txt1 = plt.text(med_N0*1.05, med_r, f"({med_N0:.0f}, {med_r:.3f})",
                color="blue", fontsize=10, weight="bold",zorder=20)
txt1.set_path_effects([path_effects.Stroke(linewidth=2, foreground="white"),
                       path_effects.Normal()])


plt.scatter(true_N0, true_r, color="red", s=70, edgecolor="k", zorder=6, label="True value")

txt2 = plt.text(true_N0*1.05, true_r, f"({true_N0:.0f}, {true_r:.3f})",
                color="red", fontsize=10, weight="bold", zorder=20)
txt2.set_path_effects([path_effects.Stroke(linewidth=2, foreground="white"),
                       path_effects.Normal()])

plt.xlabel("Effective Population Size $N_0$")
plt.ylabel("Growth Rate $r$ (per generation)")
plt.title("Joint Posterior of $N_0$ and $r$")


from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], color="crimson", lw=2.5, ls="--", label="95% HPD region"),
    Line2D([0],[0], color="gold",    lw=2.5,       label="50% HPD region"),
    Line2D([0],[0], color="black", marker="o", ls="", markersize=6, alpha=0.35, label="Posterior samples"),
    Line2D([0],[0], color="blue",  marker="o", ls="", markersize=7, label="Posterior median"),
    Line2D([0],[0], color="red",   marker="o", ls="", markersize=7, label="True value"),
]
plt.legend(handles=handles, title="Joint uncertainty & points", loc="upper left", frameon=True)

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
