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

u = np.linspace(-3, 3, 1000)

# 不同的 τ 值
taus = [0.1, 0.3, 0.5, 0.7, 0.9]

plt.figure(figsize=(10, 6))
for tau in taus:
    rho = np.where(u >= 0, tau * u, -(1 - tau) * u)
    plt.plot(u, rho, label=f'τ = {tau}')

plt.xlabel('u (误差)')
plt.ylabel('ρ_τ(u) (损失)')
plt.title('分位数损失函数 ρ_τ(u)')
plt.legend()
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
plt.show()