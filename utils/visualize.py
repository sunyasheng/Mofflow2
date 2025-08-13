import io
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D, art3d
from utils import so3 as su
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info


def visualize_torsion(gt_torsion, pred_torsion, global_step, logger, tag='torsion_unit_circle', max_samples=100, is_global_zero=True):
    """
    Visualize torsion predictions on the unit circle and log to TensorBoard (only from rank 0 / global zero).

    Args:
        gt_torsion (torch.Tensor): Ground truth torsions [N, 2] (cos, sin)
        pred_torsion (torch.Tensor): Predicted torsions [N, 2]
        global_step (int): Current training step
        logger: PyTorch Lightning logger (e.g., self.logger)
        tag (str): Tag name for TensorBoard logging
        max_samples (int): Max number of (gt, pred) pairs to plot
        is_global_zero (bool): Whether current process is global rank 0
    """
    if not is_global_zero:
        return

    gt_np = gt_torsion.detach().cpu().numpy()[:max_samples]
    pred_np = pred_torsion.detach().cpu().numpy()[:max_samples]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')

    # Draw unit circle
    circle = plt.Circle((0, 0), 1.0, color='lightgray', fill=False, linestyle='--')
    ax.add_artist(circle)

    # Colormap
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i / max_samples) for i in range(max_samples)]

    # Plot torsion pairs
    for i in range(len(gt_np)):
        gt = gt_np[i]
        pred = pred_np[i]
        color = colors[i]
        ax.plot(*gt, 'x', color=color, markersize=4)     # ground truth
        ax.plot(*pred, 'o', color=color, markersize=4)   # prediction
        ax.plot([gt[0], pred[0]], [gt[1], pred[1]], '-', color=color, alpha=0.4, linewidth=1)

    ax.set_xlim(-1.5, 3.0)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('cos(θ)')
    ax.set_ylabel('sin(θ)')
    ax.set_title(f'Torsion Prediction (step {global_step})')

    # Convert plot to image and upload to TensorBoard
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf).convert('RGB')

    if isinstance(logger, TensorBoardLogger):
        image_np = np.array(image).transpose(2, 0, 1) / 255.0  # [C, H, W]
        logger.experiment.add_image(tag, image_np, global_step=global_step)
    elif isinstance(logger, WandbLogger):
        image_np = np.array(image) / 255.0  # [H, W, C]
        logger.log_image(key=tag, images=[image_np], step=global_step)
    else:
        rank_zero_info("WARNING:: Logger not recognized. Skipping logging.")
    plt.close(fig)

def visualize_rotations(gt_rotmats, pred_rotmats, global_step, logger, tag="rotation", max_samples=100, is_global_zero=True):
    """
    Plot GT and predicted rotations in angle-axis space (3D) with color-matched pairs.
    Logs to TensorBoard as a figure.

    Args:
        gt_rotmats (torch.Tensor): Ground truth rotation matrices [N, 3, 3]
        pred_rotmats (torch.Tensor): Predicted rotation matrices [N, 3, 3]
        global_step (int): Training step
        logger: Lightning logger with .experiment.add_figure()
        tag (str): TensorBoard tag
        max_samples (int): Number of rotation pairs to plot
        is_global_zero (bool): If True, log only from rank 0
    """
    if not is_global_zero:
        return

    gt_rotvecs = su.rotmat_to_rotvec(gt_rotmats[:max_samples])
    pred_rotvecs = su.rotmat_to_rotvec(pred_rotmats[:max_samples])
    gt_rotvecs = gt_rotvecs.detach().cpu().numpy()
    pred_rotvects = pred_rotvecs.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    cmap = cm.get_cmap('hsv', max_samples)
    for i in range(len(gt_rotvecs)):
        color = cmap(i)
        ax.scatter(*gt_rotvecs[i], color=color, label=f"GT {i}", marker='x', s=30)
        ax.scatter(*pred_rotvects[i], color=color, label=f"Pred {i}", marker='o', s=30, alpha=0.7)
        # Optional: draw a line between GT and pred
        ax.plot(
            [gt_rotvecs[i, 0], pred_rotvects[i, 0]],
            [gt_rotvecs[i, 1], pred_rotvects[i, 1]],
            [gt_rotvecs[i, 2], pred_rotvects[i, 2]],
            color=color, alpha=0.4, linewidth=1,
        )

    ax.scatter(0, 0, 0, color='black', s=50, label='Identity')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"GT vs Pred Rotations (Step {global_step})")
    plt.tight_layout()

    if isinstance(logger, TensorBoardLogger):
        logger.experiment.add_figure(tag, fig, global_step)
    elif isinstance(logger, WandbLogger):
        logger.log_image(key=tag, images=[fig], step=global_step)
    else:
        rank_zero_info("WARNING:: Logger not recognized. Skipping logging.")
    plt.close(fig)