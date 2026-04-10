import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import norm

_cn_fonts = [
    '/System/Library/Fonts/STHeiti Light.ttc',
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
]
for _fp in _cn_fonts:
    try:
        fm.fontManager.addfont(_fp)
    except Exception:
        pass
_available = []
for _fp in _cn_fonts:
    try:
        _available.append(fm.FontProperties(fname=_fp).get_name())
    except Exception:
        pass
plt.rcParams['font.family'] = _available[0] if _available else 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

BG = '#f0f4f8'
BLUE_DARK  = '#4a90c4'
BLUE_MID   = '#a8c8e8'
BLUE_LIGHT = '#cce0f5'
ORANGE     = '#e07b39'

# ── 连续 quantile 曲线 ──
tau_cont = np.linspace(0.001, 0.999, 500)
theta_cont = norm.ppf(tau_cont)

# ── 离散拟合：10 根均匀柱 ──
z_vals = np.arange(-4, 5)
pmf = norm.cdf(z_vals + 0.5) - norm.cdf(z_vals - 0.5)
cdf = np.cumsum(pmf)

tau_pos = np.arange(0.1, 1.01, 0.1)
q_idx   = np.clip(np.searchsorted(cdf, tau_pos, side='left'), 0, len(z_vals) - 1)
theta_q = z_vals[q_idx]

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# 连续 quantile 曲线
ax.plot(tau_cont, theta_cont, color=BLUE_DARK, linewidth=2.2, zorder=2,
        label=r'$\theta = F^{-1}(\tau)$ (continuous)')

# 离散柱状图拟合
bar_colors = [BLUE_MID if i == 9 else BLUE_LIGHT for i in range(len(tau_pos))]
bar_edges  = [BLUE_DARK if i == 9 else '#9ab8d4' for i in range(len(tau_pos))]

ax.bar(
    tau_pos, theta_q,
    width=0.08,
    color=bar_colors,
    edgecolor=bar_edges,
    linewidth=1.2,
    alpha=0.8,
    zorder=3,
    label=r'Discrete approximation'
)

for tx, th in zip(tau_pos, theta_q):
    offset = 0.15 if th >= 0 else -0.15
    va = 'bottom' if th >= 0 else 'top'
    ax.text(tx, th + offset, str(th), ha='center', va=va,
            fontsize=8, color='#333333')

ax.axhline(y=0, color='#aaaaaa', linewidth=0.8, zorder=1)

ax.set_title(r'Quantile 视角: 给定 $\tau$, 求 $\theta=F^{-1}(\tau)$', fontsize=13, pad=10)
ax.set_xlabel(r'$\tau$', fontsize=12)
ax.set_ylabel(r'$\theta = F^{-1}(\tau)$', fontsize=11)
ax.set_xticks(tau_pos)
ax.set_xticklabels([f'{t:.1f}' for t in tau_pos], fontsize=8)
ax.set_xlim(0.0, 1.05)
ax.set_ylim(-4.7, 4.7)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper left', framealpha=0.7, edgecolor='#aaaaaa', fontsize=11)

fig.tight_layout()
fig.savefig('quantile_fit.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: quantile_fit.png")
