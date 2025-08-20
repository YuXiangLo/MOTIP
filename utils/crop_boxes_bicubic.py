import torch
import torch.nn.functional as F

def crop_boxes_bicubic(
    orig_images: torch.Tensor,          # [BT, C, H, W]
    pred_boxes: torch.Tensor,           # [BT, Q, 4]  (cx,cy,w,h) in [0,1]
    out_h: int = 384,
    out_w: int = 128,
    batch_rois: int = 512,              # tune this for your GPU
    min_side: float = 1.0,              # minimum side (pixels) after clamping
    align_corners: bool = True,
    no_grad: bool = True,               # set False if you need gradients
) -> torch.Tensor:
    """
    Returns:
        crops: [BT, Q, C, out_h, out_w], bicubic-resized per-query crops.
    Notes:
        - Uses F.grid_sample(mode='bicubic') ~ INTER_CUBIC.
        - Batched over flattened ROIs to control memory usage.
    """
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        device = orig_images.device
        dtype  = orig_images.dtype
        BT, C, H, W = orig_images.shape
        BT2, Q, four = pred_boxes.shape
        assert BT == BT2 and four == 4, "Shapes must be [BT,C,H,W] and [BT,Q,4]."

        # -- convert normalized cxcywh -> pixel xyxy (and clamp) --
        boxes = pred_boxes.to(device=device, dtype=dtype).clone()
        boxes[..., 0] *= W  # cx
        boxes[..., 1] *= H  # cy
        boxes[..., 2] *= W  # w
        boxes[..., 3] *= H  # h

        cx, cy, bw, bh = boxes.unbind(-1)
        x1 = (cx - 0.5 * bw).clamp(0, W - 1)
        y1 = (cy - 0.5 * bh).clamp(0, H - 1)
        x2 = (cx + 0.5 * bw).clamp(0, W - 1)
        y2 = (cy + 0.5 * bh).clamp(0, H - 1)
        x2 = torch.maximum(x2, x1 + min_side)
        y2 = torch.maximum(y2, y1 + min_side)

        # -- flatten to N = BT*Q --
        N = BT * Q
        x1f = x1.reshape(N, 1, 1)
        y1f = y1.reshape(N, 1, 1)
        x2f = x2.reshape(N, 1, 1)
        y2f = y2.reshape(N, 1, 1)

        # which image each ROI comes from
        b_idx = torch.arange(BT, device=device).repeat_interleave(Q)  # [N]

        # ramps shared across chunks (0..1)
        xs01 = torch.linspace(0, 1, out_w, device=device, dtype=dtype).reshape(1, 1, out_w)
        ys01 = torch.linspace(0, 1, out_h, device=device, dtype=dtype).reshape(1, out_h, 1)

        # output buffer
        crops = torch.empty((N, C, out_h, out_w), device=device, dtype=dtype)

        for s in range(0, N, batch_rois):
            e = min(s + batch_rois, N)
            M = e - s

            # pixel coords inside each ROI -> [M, out_h, out_w]
            X = x1f[s:e] + (x2f[s:e] - x1f[s:e]) * xs01      # [M, 1, out_w] broadcast on H
            Y = y1f[s:e] + (y2f[s:e] - y1f[s:e]) * ys01      # [M, out_h, 1] broadcast on W
            X = X.expand(M, out_h, out_w)
            Y = Y.expand(M, out_h, out_w)

            # normalize to [-1, 1] for grid_sample
            if align_corners:
                Xn = (X / (W - 1)) * 2 - 1
                Yn = (Y / (H - 1)) * 2 - 1
            else:
                Xn = ((X + 0.5) / W) * 2 - 1
                Yn = ((Y + 0.5) / H) * 2 - 1

            grid = torch.stack([Xn, Yn], dim=-1)             # [M, out_h, out_w, 2]

            # gather only the needed source images for this chunk
            src = orig_images[b_idx[s:e]]                    # [M, C, H, W]

            # bicubic sampling (INTER_CUBIC)
            crops[s:e] = F.grid_sample(
                src, grid, mode='bicubic', padding_mode='zeros', align_corners=align_corners
            )

        return crops.view(BT, Q, C, out_h, out_w).contiguous()
