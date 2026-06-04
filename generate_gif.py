import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Ensure assets dir exists
os.makedirs('assets', exist_ok=True)

# Generate synthetic bot path (linear)
t = np.linspace(0, 1, 100)
bot_x = np.linspace(100, 800, 100)
bot_y = np.linspace(100, 600, 100)

# Generate synthetic human path (Bezier-like with acceleration/deceleration and jitter)
control_x, control_y = 400, 800
human_x = (1-t)**2 * 100 + 2*(1-t)*t * control_x + t**2 * 800
human_y = (1-t)**2 * 100 + 2*(1-t)*t * control_y + t**2 * 600
# Add slight natural jitter
human_x += np.random.normal(0, 1.5, 100)
human_y += np.random.normal(0, 1.5, 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.patch.set_facecolor('#0d1117') # GitHub Dark Mode Background

for ax in (ax1, ax2):
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#c9d1d9')
    for spine in ax.spines.values():
        spine.set_color('#30363d')

ax1.set_xlim(0, 1000)
ax1.set_ylim(0, 1000)
ax1.set_title("Standard Robotic Bot", color='white', pad=20, fontsize=14, weight='bold')
ax1.invert_yaxis()

ax2.set_xlim(0, 1000)
ax2.set_ylim(0, 1000)
ax2.set_title("Neural ODE Human Model", color='white', pad=20, fontsize=14, weight='bold')
ax2.invert_yaxis()

# Bot line (red, straight)
line1, = ax1.plot([], [], color='#ff7b72', lw=2, alpha=0.8, linestyle='--')
point1, = ax1.plot([], [], 'o', color='#ff7b72', markersize=8)

# Human line (green, organic)
line2, = ax2.plot([], [], color='#3fb950', lw=2, alpha=0.8)
point2, = ax2.plot([], [], 'o', color='#3fb950', markersize=8)

def init():
    line1.set_data([], [])
    point1.set_data([], [])
    line2.set_data([], [])
    point2.set_data([], [])
    return line1, point1, line2, point2

def animate(i):
    line1.set_data(bot_x[:i], bot_y[:i])
    if i>0: point1.set_data([bot_x[i-1]], [bot_y[i-1]])
    
    line2.set_data(human_x[:i], human_y[:i])
    if i>0: point2.set_data([human_x[i-1]], [human_y[i-1]])
    
    return line1, point1, line2, point2

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=30, blit=True)

print("Generating trajectory_comparison.gif...")
ani.save('assets/trajectory_comparison.gif', writer='pillow', fps=30)
print("GIF successfully created!")
