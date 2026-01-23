import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

# --- 1. Setup & Style ---
plt.rcParams['font.family'] = 'monospace'
bg_color = 'white'
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# --- 2. Data Anchors ---

# A. SHARED BLACK LINE (Capability)
x_black = [2025.5, 2026.5, 2027.0, 2027.29, 2027.54, 2027.79]
y_black = [50,     80,     120,    180,     250,     350] 

# B. RED LINE (Capability - Race)
x_red =   [2027.79, 2028.15, 2028.45]
y_red =   [350,     700,     1200]

# C. GREEN LINE (Capability - Slowdown)
x_green_crash = [2027.79, 2027.95] 
y_green_crash = [350,     100]      

x_green_rec = [2027.95, 2028.25, 2028.45] 
y_green_rec = [100,      250,     600]      

# D. EXPLAINABILITY LINES (The New Branching Logic)

# 1. Common History (Blue Dotted) -> Ends at Branch Point
x_ex_common = [2025.5, 2027.29, 2027.79]
y_ex_common = [10,     55,      85]

# 2. Green Branch (Green Dotted) -> Grows
x_ex_green = [2027.79, 2027.95, 2028.45]
y_ex_green = [85,      90,      450]

# 3. Red Branch (Red Dotted) -> Stays nearly constant
x_ex_red = [2027.79, 2028.15, 2028.45]
y_ex_red = [85,      88,      92] 

# --- 3. Interpolation ---

def make_smooth(x, y):
    interpolator = PchipInterpolator(x, y)
    x_new = np.linspace(x[0], x[-1], 100)
    return x_new, interpolator(x_new)

# Capability Interpolation
x_black_plot, y_black_plot = make_smooth(x_black, y_black)
x_red_plot, y_red_plot = make_smooth(x_red, y_red)
x_green_rec_plot, y_green_rec_plot = make_smooth(x_green_rec, y_green_rec)

# Explainability Interpolation
x_ex_common_plot, y_ex_common_plot = make_smooth(x_ex_common, y_ex_common)
x_ex_green_plot, y_ex_green_plot = make_smooth(x_ex_green, y_ex_green)
x_ex_red_plot, y_ex_red_plot = make_smooth(x_ex_red, y_ex_red)


# --- 4. Plotting ---

# --- GAP FILLING ---
ax.fill_between(
    x_black_plot, 
    y_black_plot, 
    y_ex_common_plot, 
    color='#212020',  # Dark Grey
    alpha=0.5,        # 50% transparency
    zorder=1,         # Render behind the lines
    label='The "Black Box" Gap' 
)


# --- Solid Capability Lines ---
ax.plot(x_black_plot, y_black_plot, color='black', linewidth=4, zorder=2, label='Main AI Dev Line')
ax.plot(x_red_plot, y_red_plot, color='#8B3A3A', linewidth=4, zorder=2, label='Race')
ax.plot(x_green_crash, y_green_crash, color='#6E8B5B', linewidth=4, zorder=2)      
ax.plot(x_green_rec_plot, y_green_rec_plot, color='#6E8B5B', linewidth=4, zorder=2, label='Slowdown') 

# --- Dotted Explainability Lines ---
# Blue (History)
ax.plot(x_ex_common_plot, y_ex_common_plot, color='blue', linewidth=3, linestyle=':', zorder=5, label='Explained AI Capability')
# Green (Future)
ax.plot(x_ex_green_plot, y_ex_green_plot, color='#6E8B5B', linewidth=3, linestyle=':', zorder=5)
# Red (Future)
ax.plot(x_ex_red_plot, y_ex_red_plot, color='#8B3A3A', linewidth=3, linestyle=':', zorder=5)


# --- 5. Visual Elements ---

# Legend (Large)
ax.legend(loc='upper left', fontsize=20, framealpha=1, facecolor=bg_color, edgecolor='gray')

# Dots (Large)
ax.scatter([2027.29, 2027.79], [180, 350], color='black', s=180, zorder=3) 
ax.scatter([2028.15], [700], color='#8B3A3A', s=180, zorder=3)

# Threshold Lines 
thresholds = [
    (180, "Superhuman Coder"),
    (350, "Superhuman Researcher"),
    (700, "Superintelligence (ASI)")
]

for y_val, label in thresholds:
    ax.axhline(y=y_val, color='#E0E0E0', linestyle='--', linewidth=2, zorder=0)
    # Threshold Text (INCREASED FONT SIZE HERE)
    ax.text(2025.5, y_val + 15, label, fontsize=22, color='gray', va='bottom', weight='bold')

# Text Helper
def add_text(x, y, text, color='black', align='center', weight='normal', size=20):
    ax.text(x, y, text, fontsize=size, color=color, ha=align, weight=weight, 
            bbox=dict(facecolor=bg_color, edgecolor='none', alpha=0.8, pad=4))

# Main Annotations
add_text(2027.29, 210, "Automated\nCoding", align='right')
add_text(2027.79, 390, "Branch Point", weight='bold')

# Red Annotations
add_text(2028.15, 760, "AI Takeover", color='#8B3A3A', align='right', weight='bold')

# Green Annotations 
add_text(2027.87, 140, "Government\nAlignment", color='#6E8B5B', align='left', weight='bold') 

# Axis Cleanup & Labels
ax.set_xticks([2026, 2027, 2027.29, 2027.54, 2027.79, 2028])
ax.set_xticklabels(["2026", "2027", "Apr", "Jul", "Oct", "2028"])

# Tick Params
ax.tick_params(axis='x', colors='black', labelsize=18, width=2, length=6)
ax.set_yticks([]) 

for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# AXIS TITLES
ax.set_xlabel("TIME", loc='right', fontsize=24, weight='bold')
ax.set_ylabel("AI Capability", fontsize=24, weight='bold')

plt.xlim(2025.5, 2028.5)
plt.ylim(0, 1300) 
plt.tight_layout()
plt.savefig("ai_capability_plot_presentation.png")
#plt.show()