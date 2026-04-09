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
print("Available Chinese font names:", _available)
plt.rcParams['font.family'] = _available[0] if _available else 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 离散点：整数 z 值，从 -4 到 4
z_vals = np.arange(-4, 5)

# 用正态分布 CDF 差值计算每个整数区间的概率质量
# P(k - 0.5 < Z <= k + 0.5) ≈ Φ(k+0.5) - Φ(k-0.5)
pmf = norm.cdf(z_vals + 0.5) - norm.cdf(z_vals - 0.5)
cdf = np.cumsum(pmf)

# 累积概率：P(Z <= 1)
tau_cutoff = 1
tau = pmf[z_vals <= tau_cutoff].sum()
tau_cdf = cdf[z_vals == tau_cutoff][0]

BG = '#f0f4f8'
BLUE_DARK  = '#4a90c4'
BLUE_MID   = '#a8c8e8'
BLUE_LIGHT = '#cce0f5'
ORANGE     = '#e07b39'
RED_DARK   = '#8b1a1a'

def _make_fig():
    f, ax = plt.subplots(figsize=(9, 5))
    f.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return f, ax

# ── 图1：离散 PMF ──────────────────────────────────────────────
fig1, ax1 = _make_fig()
colors   = [BLUE_MID   if z <= tau_cutoff else BLUE_LIGHT for z in z_vals]
bar_edge = [BLUE_DARK  if z <= tau_cutoff else '#9ab8d4'  for z in z_vals]

ax1.bar(
    z_vals, pmf,
    width=0.75,
    color=colors,
    edgecolor=bar_edge,
    linewidth=1.2,
    zorder=3,
    label=rf'$\tau = P(Z \leq 1) = {tau:.4f}$'
)

ax1.axvline(x=tau_cutoff + 0.5, color=ORANGE, linestyle='--', linewidth=1.8, zorder=4)

for z, p in zip(z_vals, pmf):
    ax1.text(z, p + 0.004, f'{p:.3f}', ha='center', va='bottom',
             fontsize=7.5, color='#333333')

ax1.set_title(r'$p(z)$ Discretized Normal', fontsize=13, pad=10)
ax1.set_xlabel(r'$z$', fontsize=12)
ax1.set_ylabel('Probability Mass', fontsize=11)
ax1.set_xticks(z_vals)
ax1.set_xlim(-4.7, 4.7)
ax1.set_ylim(0, max(pmf) * 1.18)
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
ax1.spines[['top', 'right']].set_visible(False)
ax1.legend(loc='upper left', framealpha=0.7, edgecolor='#aaaaaa', fontsize=11)

fig1.tight_layout()
fig1.savefig('discrete_pmf.png', dpi=150, bbox_inches='tight')
print("Saved: discrete_pmf.png")

# ── 图2：离散 CDF（柱状图）──────────────────────────────────
fig2, ax2 = _make_fig()

cdf_colors   = [BLUE_MID  if z == tau_cutoff else BLUE_LIGHT for z in z_vals]
cdf_bar_edge = [BLUE_DARK if z == tau_cutoff else '#9ab8d4'  for z in z_vals]

ax2.bar(
    z_vals, cdf,
    width=0.75,
    color=cdf_colors,
    edgecolor=cdf_bar_edge,
    linewidth=1.2,
    zorder=3,
    label=rf'$\tau = F(1) = {tau_cdf:.4f}$'
)

for z, c in zip(z_vals, cdf):
    ax2.text(z, c + 0.01, f'{c:.3f}', ha='center', va='bottom',
             fontsize=7.5, color='#333333')

ax2.set_title(r'CDF 视角: 给定 $z$, 求 $\tau=F(z)$', fontsize=13, pad=10)
ax2.set_xlabel(r'$z$', fontsize=12)
ax2.set_ylabel(r'$\tau = F(z)$', fontsize=11)
ax2.set_xticks(z_vals)
ax2.set_xlim(-4.7, 4.7)
ax2.set_ylim(0, 1.18)
ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
ax2.spines[['top', 'right']].set_visible(False)
ax2.legend(loc='upper left', framealpha=0.7, edgecolor='#aaaaaa', fontsize=11)

fig2.tight_layout()
fig2.savefig('discrete_cdf.png', dpi=150, bbox_inches='tight')
print("Saved: discrete_cdf.png")

# ── 图3：离散 Quantile（竖向柱状图，10 根均匀柱）────────────
fig3, ax3 = _make_fig()

GREEN = '#2a7a2a'

tau_pos = np.arange(0.1, 1.01, 0.1)

q_idx    = np.clip(np.searchsorted(cdf, tau_pos, side='left'), 0, len(z_vals) - 1)
z_quant  = z_vals[q_idx]

hi = int(np.argmin(np.abs(tau_pos - tau_cdf)))
q_colors   = [BLUE_MID  if i == hi else BLUE_LIGHT for i in range(len(tau_pos))]
q_bar_edge = [BLUE_DARK if i == hi else '#9ab8d4'  for i in range(len(tau_pos))]

ax3.bar(
    tau_pos, z_quant,
    width=0.08,
    color=q_colors,
    edgecolor=q_bar_edge,
    linewidth=1.2,
    zorder=3,
    label=rf'$z = F^{{-1}}(\tau) = {z_quant[hi]}$  ($\tau={tau_pos[hi]:.2f}$)'
)

for tx, z in zip(tau_pos, z_quant):
    offset = 0.12 if z >= 0 else -0.12
    va = 'bottom' if z >= 0 else 'top'
    ax3.text(tx, z + offset, str(z), ha='center', va=va,
             fontsize=8, color='#333333')

ax3.axhline(y=0, color='#aaaaaa', linewidth=0.8, zorder=2)

ax3.set_title(r'Quantile 视角: 给定 $\tau$, 求 $z=F^{-1}(\tau)$', fontsize=13, pad=10)
ax3.set_xlabel(r'$\tau$', fontsize=12)
ax3.set_ylabel(r'$z = F^{-1}(\tau)$', fontsize=11)
ax3.set_xticks(tau_pos)
ax3.set_xticklabels([f'{t:.2f}' for t in tau_pos], fontsize=8)
ax3.set_xlim(0.02, 1.08)
ax3.set_ylim(-4.7, 4.7)
ax3.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
ax3.spines[['top', 'right']].set_visible(False)
ax3.legend(loc='upper left', framealpha=0.7, edgecolor='#aaaaaa', fontsize=11)

fig3.tight_layout()
fig3.savefig('discrete_quantile.png', dpi=150, bbox_inches='tight')
print("Saved: discrete_quantile.png")

plt.show()
print(f"tau = P(Z <= 1) = {tau:.4f}  |  F(1) = {tau_cdf:.4f}")
