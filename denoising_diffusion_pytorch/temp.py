import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Read two GIF images
gif1_path = 'path/to/gif1.gif'
gif2_path = 'path/to/gif2.gif'

# Create a subplot showing two images per row
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Disable axis display
for ax in axes:
    ax.axis('off')

# Show first GIF
img1 = plt.imread(gif1_path)
im1 = axes[0].imshow(img1)

# Show second GIF
img2 = plt.imread(gif2_path)
im2 = axes[1].imshow(img2)

# Update function, used for animation effects
def update(frame):
    pass

# Create animated objects
ani = animation.FuncAnimation(fig, update, frames=range(10), interval=200, blit=False)

# show animation
plt.show()
