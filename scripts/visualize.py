"""Visualization scripts for OrcaQD archive analysis and paper figures."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys

plt.rcParams.update({
    'font.size': 11,
    'axes.linewidth': 1.0,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUT_DIR = Path('outputs/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_archive(path: str) -> dict:
    """Load a MAP-Elites archive from .npz file."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def plot_archive_heatmap(
    repertoire: dict,
    grid_shape: tuple = (25, 25),
    min_bd: tuple = (-1.134, -1.0),
    max_bd: tuple = (0.611, 1.0),
    title: str = 'Archive Heatmap',
    xlabel: str = 'Wrist Yaw (rad)',
    ylabel: str = 'Mean PIP Flexion (tanh)',
    save_path: str = None,
):
    """Figure 1: Archive heatmap colored by fitness."""
    fitnesses = repertoire.get('fitnesses', repertoire.get('QD_score', None))
    if fitnesses is None:
        # Try to reconstruct from repertoire structure
        print("Warning: Could not find fitnesses in archive")
        return

    # Reshape to grid
    grid = np.full(grid_shape, np.nan)
    descriptors = repertoire.get('descriptors', None)
    
    if descriptors is not None and descriptors.shape[0] == np.prod(grid_shape):
        for i in range(descriptors.shape[0]):
            d = descriptors[i]
            if fitnesses[i] > -np.inf:  # Filled cell
                ix = int((d[0] - min_bd[0]) / (max_bd[0] - min_bd[0]) * (grid_shape[0] - 1))
                iy = int((d[1] - min_bd[1]) / (max_bd[1] - min_bd[1]) * (grid_shape[1] - 1))
                ix = np.clip(ix, 0, grid_shape[0] - 1)
                iy = np.clip(iy, 0, grid_shape[1] - 1)
                grid[ix, iy] = fitnesses[i]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid.T, origin='lower', cmap='viridis', aspect='auto',
                   extent=[min_bd[0], max_bd[0], min_bd[1], max_bd[1]])
    
    filled = np.sum(~np.isnan(grid))
    total = np.prod(grid_shape)
    ax.set_title(f'{title}\nCoverage: {filled}/{total} ({filled/total*100:.1f}%)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label='Fitness (QD-Score contribution)')
    
    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_convergence_curve(
    qd_scores: list,
    coverages: list,
    save_path: str = None,
    title: str = 'Training Convergence',
):
    """Figure 2: QD-Score and Coverage over iterations."""
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    iterations = range(1, len(qd_scores) + 1)
    color1 = '#2563eb'
    ax1.plot(iterations, qd_scores, color=color1, linewidth=2, label='QD-Score')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('QD-Score', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = '#dc2626'
    ax2.plot(iterations, coverages, color=color2, linewidth=2, label='Coverage')
    ax2.set_ylabel('Coverage (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_comparison(
    methods: dict,
    metric: str = 'qd_score',
    save_path: str = None,
    title: str = 'Method Comparison',
):
    """Figure 3: Compare QD-Score across methods."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    names = list(methods.keys())
    values = [methods[n][metric] for n in names]
    colors = ['#2563eb', '#dc2626', '#059669', '#d97706']
    
    bars = ax.bar(names, values, color=colors[:len(names)], edgecolor='black', linewidth=0.5)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(values),
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_bd_distribution(
    descriptors: np.ndarray,
    save_path: str = None,
    title: str = 'Behavior Descriptor Distribution',
):
    """Figure 4: 2D scatter of BD values."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.scatter(descriptors[:, 0], descriptors[:, 1], alpha=0.3, s=10, c='steelblue')
    ax.set_xlabel('BD1: Wrist Yaw (rad)')
    ax.set_ylabel('BD2: Mean PIP Flexion (tanh)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def generate_all_figures(archive_path: str, log_path: str = None):
    """Generate all paper figures from archive and log files."""
    print(f"Loading archive: {archive_path}")
    repertoire = load_archive(archive_path)
    
    # Parse log file for convergence data
    qd_scores = []
    coverages = []
    if log_path and Path(log_path).exists():
        with open(log_path) as f:
            for line in f:
                if 'QD=' in line and 'Cov=' in line:
                    try:
                        qd = float(line.split('QD=')[1].split()[0])
                        cov = float(line.split('Cov=')[1].split()[0])
                        qd_scores.append(qd)
                        coverages.append(cov)
                    except (ValueError, IndexError):
                        pass
    
    # Generate figures
    plot_archive_heatmap(
        repertoire,
        save_path=str(OUT_DIR / 'fig1_archive_heatmap.png'),
        title='OrcaQD Archive: PGA-MAP-Elites',
    )
    
    if qd_scores:
        plot_convergence_curve(
            qd_scores, coverages,
            save_path=str(OUT_DIR / 'fig2_convergence.png'),
        )
    
    # BD distribution
    descriptors = repertoire.get('descriptors', None)
    if descriptors is not None:
        filled = descriptors[descriptors[:, 0] != -np.inf]
        if len(filled) > 0:
            plot_bd_distribution(
                filled,
                save_path=str(OUT_DIR / 'fig4_bd_distribution.png'),
            )
    
    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True, help='Path to archive .npz file')
    parser.add_argument('--log', help='Path to training log file')
    args = parser.parse_args()
    
    generate_all_figures(args.archive, args.log)
