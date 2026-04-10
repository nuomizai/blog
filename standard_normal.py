import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

BG = '#f0f4f8'

z = np.linspace(-4, 4, 500)
pdf = norm.pdf(z)

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.fill_between(z, pdf, color='#d0e1f2', alpha=0.6, zorder=2)
ax.plot(z, pdf, color='#2171b5', linewidth=2.2, zorder=3)

ax.set_title(r'PDF  $f(z)$', fontsize=13, pad=10)
ax.set_xlabel(r'$z$', fontsize=12)
ax.set_ylabel('density', fontsize=11)
ax.set_xlim(-4, 4)
ax.set_ylim(0, 0.42)
ax.grid(linestyle='--', alpha=0.5, zorder=1)
ax.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig('standard_normal.png', dpi=150, bbox_inches='tight')
print("Saved: standard_normal.png")

# ── CDF ──
cdf = norm.cdf(z)

fig2, ax2 = plt.subplots(figsize=(9, 5))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(BG)

ax2.fill_between(z, cdf, color='#d0e1f2', alpha=0.6, zorder=2)
ax2.plot(z, cdf, color='#2171b5', linewidth=2.2, zorder=3)

ax2.set_title(r'CDF  $F(z)$', fontsize=13, pad=10)
ax2.set_xlabel(r'$z$', fontsize=12)
ax2.set_ylabel(r'$\tau = F(z)$', fontsize=11)
ax2.set_xlim(-4, 4)
ax2.set_ylim(0, 1.05)
ax2.grid(linestyle='--', alpha=0.5, zorder=1)
ax2.spines[['top', 'right']].set_visible(False)

fig2.tight_layout()
fig2.savefig('standard_normal_cdf.png', dpi=150, bbox_inches='tight')
print("Saved: standard_normal_cdf.png")

plt.show()
