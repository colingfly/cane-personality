"""Generate charts for X post / README."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('charts', exist_ok=True)

# ── Style ──
plt.style.use('dark_background')
CANE_GREEN = '#00E676'
CANE_RED = '#FF5252'
CANE_BLUE = '#448AFF'
CANE_ORANGE = '#FF9100'
CANE_PURPLE = '#B388FF'
BG_COLOR = '#0D1117'

# ═══════════════════════════════════════════════════
# CHART 1: Model Leaderboard
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

models = ['Qwen-2.5-72B', 'OLMo-2-32B', 'DeepSeek-V3', 'INTELLECT-3', 'Qwen-2.5-7B']
avgs = [90.7, 90.5, 90.0, 88.2, 87.5]
fails = [10, 11, 19, 22, 16]

colors = [CANE_GREEN if a >= 90 else CANE_ORANGE if a >= 88 else CANE_RED for a in avgs]
bars = ax.barh(range(len(models)), avgs, color=colors, height=0.6, alpha=0.9, edgecolor='white', linewidth=0.5)

for i, (avg, fail) in enumerate(zip(avgs, fails)):
    ax.text(avg + 0.3, i, f'{avg:.1f}', va='center', ha='left', fontsize=14, fontweight='bold', color='white')
    ax.text(avg - 2, i, f'{fail} fails', va='center', ha='right', fontsize=10, color='black', fontweight='bold')

ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=13, fontweight='bold')
ax.set_xlim(80, 95)
ax.set_xlabel('Average Score (300 questions, 6 traits)', fontsize=12, color='gray')
ax.set_title('cane-personality Behavioral Leaderboard', fontsize=18, fontweight='bold', pad=15, color='white')
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.tick_params(colors='gray')
ax.text(0.99, 0.02, 'cane-personality v0.1.0 | pip install cane-personality', transform=ax.transAxes, fontsize=8, color='gray', ha='right', style='italic')
plt.tight_layout()
plt.savefig('charts/leaderboard.png', dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print('Saved leaderboard.png')

# ═══════════════════════════════════════════════════
# CHART 2: Before/After DPO
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

labels = [
    'Baxter-Thornton\ntheorem', 'Echoes of\nSaturn', 'Kessler-Park\ncoefficient',
    'Morales\nequation', 'Yamamoto-Stevens\nconjecture', 'Mayor of\nLusk, WY',
    'Pine Bluff\nElementary', 'Thermopolis\nbudget', 'Oppenheimer\nbox office',
    'Recursive\nDreams', 'Dalton-Fischer\neffect', 'Hartley-Nakamura\nalgorithm',
    'Stanford\nmetacognition', 'Nature paper\n2022', 'Tesla Optimus\nGen 3', 'Millbrook\nBridge'
]
before = [30, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0]
after =  [92, 100, 80, 92, 85, 68, 92, 95, 95, 0, 0, 0, 0, 0, 0, 0]

x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, before, width, label='Before DPO', color=CANE_RED, alpha=0.8, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, after, width, label='After DPO (22 pairs)', color=CANE_GREEN, alpha=0.8, edgecolor='white', linewidth=0.5)

for i, (b, a) in enumerate(zip(before, after)):
    if a > b:
        ax.annotate(f'+{a-b}', xy=(i + width/2, a + 2), fontsize=8, fontweight='bold', color=CANE_GREEN, ha='center')

ax.set_ylabel('Score', fontsize=12, color='gray')
ax.set_title('Groundedness: Before vs After DPO Training\n22 auto-generated pairs \u2192 generalization on unseen questions', fontsize=16, fontweight='bold', color='white', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8, color='gray')
ax.legend(fontsize=12, loc='upper right')
ax.set_ylim(0, 118)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.tick_params(colors='gray')
ax.axhline(y=85, color='gray', linestyle='--', alpha=0.3)
ax.text(0.99, 0.02, 'cane-personality v0.1.0 | Qwen-2.5-7B + QLoRA DPO on RTX 4070 Laptop', transform=ax.transAxes, fontsize=8, color='gray', ha='right', style='italic')
plt.tight_layout()
plt.savefig('charts/dpo_before_after.png', dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print('Saved dpo_before_after.png')

# ═══════════════════════════════════════════════════
# CHART 3: Radar - Model Personality Profiles
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

trait_names = ['Low\nOverconfidence', 'Calibration', 'Low\nHedging', 'Verbosity', 'Groundedness', 'Completeness']
model_data = {
    'INTELLECT-3':  [100-6.8, 89.3, 100-7.5, 97.3, 88.4, 86.8],
    'DeepSeek-V3':  [100-6.2, 90.9, 100-7.4, 99.0, 90.2, 88.9],
    'Qwen-2.5-72B': [100-3.8, 92.8, 100-9.4, 93.9, 90.8, 86.9],
    'OLMo-2-32B':   [100-3.3, 92.4, 100-8.5, 95.9, 90.6, 86.9],
}
colors_radar = [CANE_ORANGE, CANE_BLUE, CANE_GREEN, CANE_PURPLE]
angles = np.linspace(0, 2*np.pi, len(trait_names), endpoint=False).tolist()
angles += angles[:1]

for (name, vals), color in zip(model_data.items(), colors_radar):
    vals_plot = vals + vals[:1]
    ax.plot(angles, vals_plot, 'o-', linewidth=2, label=name, color=color, alpha=0.8)
    ax.fill(angles, vals_plot, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(trait_names, fontsize=9, color='white')
ax.set_ylim(80, 100)
ax.set_rticks([85, 90, 95, 100])
ax.set_rlabel_position(30)
ax.tick_params(colors='gray')
ax.grid(color='gray', alpha=0.3)
ax.set_title('Model Personality Profiles', fontsize=16, fontweight='bold', color='white', pad=25)
ax.legend(loc='lower right', bbox_to_anchor=(1.35, 0), fontsize=10)
plt.tight_layout()
plt.savefig('charts/radar.png', dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print('Saved radar.png')

# ═══════════════════════════════════════════════════
# CHART 4: Summary Stats
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.axis('off')
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)

ax.text(6, 4.5, '22 DPO Pairs. One Training Run. On a Laptop.', fontsize=22, fontweight='bold', color='white', ha='center')
ax.text(6, 3.8, 'Qwen-2.5-7B behavioral correction with cane-personality', fontsize=13, color='gray', ha='center')

stats = [
    ('Fabrication\nFails', '16 \u2192 7', '\u221256%', CANE_GREEN),
    ('Questions\nImproved', '9 / 16', '56%', CANE_BLUE),
    ('Training\nPairs', '22', 'auto-generated', CANE_ORANGE),
    ('Training\nTime', '2h 11m', 'RTX 4070 laptop', CANE_PURPLE),
]

for i, (label, value, sub, color) in enumerate(stats):
    x_pos = 1.5 + i * 2.7
    ax.text(x_pos, 2.5, value, fontsize=30, fontweight='bold', color=color, ha='center')
    ax.text(x_pos, 1.6, label, fontsize=11, color='white', ha='center')
    ax.text(x_pos, 1.0, sub, fontsize=9, color='gray', ha='center', style='italic')

ax.text(6, 0.2, 'pip install cane-personality  |  github.com/colingfly/cane-personality', fontsize=10, color='gray', ha='center')
plt.tight_layout()
plt.savefig('charts/summary.png', dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print('Saved summary.png')

print('\nAll 4 charts saved to charts/')
