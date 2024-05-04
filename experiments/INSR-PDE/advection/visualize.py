import matplotlib.pyplot as plt


def draw_signal1D(x, y, y_max=None, y_gt=None):
    fig, ax = plt.subplots()
    if y_gt is not None:
        ax.plot(x, y_gt, color='red', alpha=0.2)
    ax.plot(x, y)
    if y_max is not None:
        ax.set_ylim(0, y_max)
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig


def save_figure(fig, save_path, close=True):
    plt.savefig(save_path, bbox_inches='tight')
    if close:
        plt.close(fig)
