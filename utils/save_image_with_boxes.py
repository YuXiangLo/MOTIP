import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch

def save_image_with_boxes(images_tensor, boxes, labels, filepath, scores=None, title=None):
    """
    images_tensor: (C, H, W) tensor in [0, 1]
    boxes: (N, 4) cx,cy,w,h in [0,1]
    labels: (N,) long
    scores: (N,) float in [0,1], optional
    """
    # --- move to CPU + numpy ---
    if isinstance(images_tensor, torch.Tensor):
        images_tensor = images_tensor.detach().cpu()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu()

    img = images_tensor.permute(1, 2, 0).numpy()  # HWC
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(img)
    if title:
        ax.set_title(title)

    # cxcywh -> xyxy (pixel space)
    h, w = img.shape[:2]
    if len(boxes) > 0:
        boxes_xyxy = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        boxes_xyxy[:, [0, 2]] *= w
        boxes_xyxy[:, [1, 3]] *= h

        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy.tolist()):
            rect = plt.Rectangle((x1, y1), max(0.0, x2 - x1), max(0.0, y2 - y1),
                                 linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            txt = str(int(labels[i]))
            if scores is not None:
                txt += f" {scores[i]:.2f}"
            ax.text(x1, y1, txt, color='white', fontsize=8,
                    bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)

