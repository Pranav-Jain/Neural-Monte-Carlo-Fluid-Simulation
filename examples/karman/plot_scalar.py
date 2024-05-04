import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import imageio
def draw_scalar_field2D(arr, vmin=None, vmax=None, cmap='bwr', colorbar=False):
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(5000*px, 2000*px))

    ax.pcolormesh(arr, shading='gouraud', vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_axis_off()
    ax.set_aspect('equal')
    plt.axis('equal')
    return fig

def save_figure(fig, save_path, close=True):
    plt.savefig(save_path, bbox_inches='tight', dpi=1200)
    if close:
        plt.close("all")

dir = sys.argv[1]
res = int(sys.argv[2])+2
print("\nCreating vorticity images...")
for i in range(1000):
    file_s = os.path.join(dir, f'vorticity_samples_t{i:03d}.txt')
    file_v = os.path.join(dir, f'vorticity_values_t{i:03d}.txt')
    try:
        samples = np.loadtxt(file_s)
        values = np.loadtxt(file_v)
    except:
        break
    
    v = values.reshape((int(values.shape[0]/res), res))
    v[np.abs(v)<0.3] = 0.0
    fig = draw_scalar_field2D(v, vmin=-5, vmax=5)
    # plt.gca().invert_yaxis()
    plt.savefig(os.path.join(dir, os.pardir, f'vorticity/vorticity_t{i:03d}.png'), bbox_inches='tight')
    plt.close()

print("Done")
