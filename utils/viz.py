"""Drawing helpers shared by the inference scripts."""
import cv2
import torch


def draw_dashed_rect(img, x1, y1, x2, y2, color, thickness=1, dash=5, gap=4):
    """dashed rectangle (no cv2 builtin)"""
    def hline(y, xs, xe):
        x = xs
        while x < xe:
            cv2.line(img, (x, y), (min(x + dash, xe), y), color, thickness)
            x += dash + gap

    def vline(x, ys, ye):
        y = ys
        while y < ye:
            cv2.line(img, (x, y), (x, min(y + dash, ye)), color, thickness)
            y += dash + gap

    hline(y1, x1, x2)
    hline(y2, x1, x2)
    vline(x1, y1, y2)
    vline(x2, y1, y2)


def draw_bw_dashed_rect(img, x1, y1, x2, y2, thickness=1, dash=5, gap=4):
    """gt-box style: white underlay + black dashes"""
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness + 1)
    draw_dashed_rect(img, x1, y1, x2, y2, (0, 0, 0), thickness=thickness, dash=dash, gap=gap)


def xyxy_int(b, w, h):
    """clamp an xyxy box to the image, return ints"""
    return (
        int(min(max(float(b[0]), 0), w)),
        int(min(max(float(b[1]), 0), h)),
        int(min(max(float(b[2]), 0), w)),
        int(min(max(float(b[3]), 0), h)),
    )


def draw_label(img, segments, org=(10, 30), font_scale=1.0, thickness=2, gap=14):
    """draw (text, color) segments left to right, outlined so they stay readable"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = org
    for text, color in segments:
        cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness + 4, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x += tw + gap


def save_depth_visual(model_module, img_1chw, pad_wh, orig_w, orig_h, out_path):
    """save the 1-ch depth map the model consumed (turbo colormap, near = warm)"""
    depth_model = getattr(model_module, 'depth_model', None)
    if depth_model is None:
        return False
    with torch.no_grad():
        # valid-region depth, same input the fusion sees
        d = model_module.predict_depth_map(img_1chw.unsqueeze(0))[1][0, 0]  # [1] = 1-ch disparity; (Hd, Wd)
    hd, wd = d.shape[-2], d.shape[-1]
    in_h, in_w = img_1chw.shape[-2], img_1chw.shape[-1]
    # crop the zero padding (pad_wh in model-frame px)
    vw = max(1, int(round((in_w - float(pad_wh[0])) / in_w * wd)))
    vh = max(1, int(round((in_h - float(pad_wh[1])) / in_h * hd)))
    d = d[:vh, :vw].float()
    # clip outliers before min-max
    lo, hi = torch.quantile(d, 0.02), torch.quantile(d, 0.98)
    d = d.clamp(lo, hi)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    # near = warm: invert depth-like outputs
    if getattr(model_module.depth_model, 'cfg', {}).get('output', 'depth') == 'depth':
        d = 1.0 - d
    d8 = (d * 255.0).to(torch.uint8).cpu().numpy()
    dvis = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    dvis = cv2.resize(dvis, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_path, dvis)
    # pca view of the decoder features when they feed the adapter
    if getattr(model_module, 'depth_source', 'scalar') == 'decoder':
        alt = out_path.replace('_depth.png', '_depth_path1.png')
        if alt != out_path:
            model_module.save_depth_feature_visual(img_1chw, pad_wh, orig_w, orig_h, alt)
    return True


def save_density_visual(dmap, pad_wh, in_h, in_w, orig_w, orig_h, out_path):
    """save the predicted density map (jet), pad-cropped and resized like the count"""
    d = dmap.detach().float()
    d = d.reshape(d.shape[-2], d.shape[-1])
    hd, wd = d.shape
    # crop the zero padding (pad_wh in model-frame px)
    vw = max(1, int(round((in_w - float(pad_wh[0])) / in_w * wd)))
    vh = max(1, int(round((in_h - float(pad_wh[1])) / in_h * hd)))
    d = d[:vh, :vw].clamp(min=0)
    d8 = (d / (d.max() + 1e-8) * 255.0).to(torch.uint8).cpu().numpy()
    dvis = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    dvis = cv2.resize(dvis, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_path, dvis)
    return True


def save_density_canvas(dens_map, orig_w, orig_h, out_path):
    """save the stitched per-tile density canvas (jet)"""
    d = dens_map.detach().float().clamp(min=0)
    d8 = (d / (d.max() + 1e-8) * 255.0).to(torch.uint8).cpu().numpy()
    dvis = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    if dvis.shape[:2] != (orig_h, orig_w):
        dvis = cv2.resize(dvis, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_path, dvis)
    return True


def blend_window(th, tw, rect, orig_w, orig_h, margin, device):
    """(th, tw) blend weights: ramp over margin px on internal tile edges, flat 1
    on image borders, so two tiles sum to ~1 across a seam"""
    x0, y0, x1, y1 = rect
    m = max(1, min(int(margin), th // 2, tw // 2))
    ramp = torch.linspace(1.0 / (m + 1), 1.0, m, device=device)
    wy = torch.ones(th, device=device)
    wx = torch.ones(tw, device=device)
    if y0 > 0:
        wy[:m] = torch.minimum(wy[:m], ramp)
    if y1 < orig_h:
        wy[-m:] = torch.minimum(wy[-m:], ramp.flip(0))
    if x0 > 0:
        wx[:m] = torch.minimum(wx[:m], ramp)
    if x1 < orig_w:
        wx[-m:] = torch.minimum(wx[-m:], ramp.flip(0))
    return torch.outer(wy, wx)
