# transform_psd_assets.py
import os
import math
import argparse
import numpy as np
from PIL import Image
import cv2
from psd_tools import PSDImage
from skimage import img_as_float
from skimage.filters import gaussian

# ---------------------------
# Utility: PSD rasterization
# ---------------------------
def rasterize_psd(psd_path):
    psd = PSDImage.open(psd_path)
    comp = psd.composite()
    img = np.array(comp.convert("RGB"))
    return img  # HxWx3 uint8

# ---------------------------
# Saliency: Spectral residual
# ---------------------------
def spectral_saliency(img_rgb):
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img_f = img.astype(np.float32)

    fft = np.fft.fft2(img_f)
    log_amp = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)

    avg = cv2.blur(log_amp, (3, 3))
    residual = log_amp - avg
    saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase)))
    sal = np.real(saliency)
    sal = cv2.GaussianBlur(sal, (5, 5), 0)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    sal = cv2.resize(sal, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return sal  # HxW float in [0,1]

# ----------------------------------------
# Text detection: MSER + stroke heuristics
# ----------------------------------------
def detect_text_regions(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    mser = cv2.MSER_create(delta=5, min_area=60, max_area=int(0.01 * H * W))
    regions, _ = mser.detectRegions(gray)

    boxes = []
    for p in regions:
        x, y, w, h = cv2.boundingRect(p.reshape(-1, 1, 2))
        if w < 8 or h < 8:
            continue
        aspect = w / float(h)
        if aspect < 0.2 or aspect > 20:
            continue
        roi = gray[y:y+h, x:x+w]
        edges = cv2.Canny(roi, 50, 150)
        edge_density = edges.sum() / (255.0 * max(1, w*h))
        if 0.02 < edge_density < 0.35:
            boxes.append((x, y, w, h))
    boxes = non_max_suppression(boxes, 0.3)
    return boxes

# -------------------------------------
# Logo/CTA detection: shape + color cue
# -------------------------------------
def detect_logo_and_cta(img_rgb):
    # Heuristic: CTAs often high-contrast rounded rectangles with short text
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    edges = cv2.Canny(v, 60, 120)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logos = []
    ctas = []
    H, W = v.shape
    area_thresh = max(100, int(0.0005 * H * W))
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < area_thresh:
            continue
        aspect = w / float(h)
        rect_like = 0.3 < aspect < 5.0
        # Roundness via contour approximation
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        rect_corners = len(approx) in (4, 5, 6)

        roi = img_rgb[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        # Uniformity: logos/CTAs often have strong color uniformity
        roi_lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
        l_std = roi_lab[..., 0].std()
        a_std = roi_lab[..., 1].std()
        b_std = roi_lab[..., 2].std()
        color_uniform = (l_std + a_std + b_std) / 3.0 < 18.0

        # CTA heuristic: medium-sized rect with color uniformity and short text overlap later
        if rect_like and rect_corners and color_uniform:
            ctas.append((x, y, w, h))
        # Logo heuristic: compact shape, possibly nearly square or emblem
        if 0.6 < aspect < 1.8 and area < 0.1 * H * W and color_uniform:
            logos.append((x, y, w, h))

    return non_max_suppression(logos, 0.2), non_max_suppression(ctas, 0.2)

# ---------------------------
# NMS for bounding boxes
# ---------------------------
def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(y2)
    pick = []

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(int(last))
        suppress = [len(idxs) - 1]
        for pos in range(len(idxs) - 1):
            i = idxs[pos]
            xx1 = max(x1[last], x1[i])
            yy1 = max(y1[last], y1[i])
            xx2 = min(x2[last], x2[i])
            yy2 = min(y2[last], y2[i])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            overlap = (w * h) / (areas[i] + areas[last] - w * h + 1e-8)
            if overlap > overlap_thresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return [tuple(map(int, boxes[i])) for i in pick]

# -------------------------------------
# Importance map combining detections
# -------------------------------------
def build_importance_map(img_rgb, saliency, text_boxes, logo_boxes, cta_boxes):
    H, W = img_rgb.shape[:2]
    imp = saliency.copy()
    # Weight detections
    def add_box_weight(imp, boxes, weight=2.0, feather=15):
        for (x, y, w, h) in boxes:
            mask = np.zeros((H, W), dtype=np.float32)
            mask[y:y+h, x:x+w] = 1.0
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather, sigmaY=feather)
            imp += weight * mask
        return imp

    imp = add_box_weight(imp, text_boxes, weight=3.0)
    imp = add_box_weight(imp, logo_boxes, weight=2.5)
    imp = add_box_weight(imp, cta_boxes, weight=3.5)
    # Normalize
    imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)
    return imp

# ------------------------------------------
# Smart crop: maximize importance coverage
# ------------------------------------------
def smart_crop(img_rgb, imp_map, target_w, target_h):
    H, W = imp_map.shape
    target_ratio = target_w / float(target_h)

    # Search window sizes around ratio with scale
    scales = np.linspace(0.4, 1.0, 15)
    best_score = -1
    best_rect = (0, 0, W, H)

    for s in scales:
        win_w = int(W * s)
        win_h = int(win_w / target_ratio)
        if win_h > H:
            win_h = H
            win_w = int(win_h * target_ratio)
        if win_w < 10 or win_h < 10:
            continue
        step_x = max(8, win_w // 20)
        step_y = max(8, win_h // 20)
        for y in range(0, H - win_h + 1, step_y):
            for x in range(0, W - win_w + 1, step_x):
                sub = imp_map[y:y+win_h, x:x+win_w]
                # coverage + compactness bonus
                score = sub.mean() + 0.05 * (win_w * win_h) / (W * H)
                if score > best_score:
                    best_score = score
                    best_rect = (x, y, win_w, win_h)
    x, y, win_w, win_h = best_rect
    crop = img_rgb[y:y+win_h, x:x+win_w]
    resized = resize_with_letterbox(crop, target_w, target_h, bg_mode='blur')
    return resized, best_rect

# ---------------------------------------------------------
# Reposition: constraint-based layout for narrow canvases
# ---------------------------------------------------------
def reposition_elements(base_img, detections, target_w, target_h):
    # detections: dict with keys 'text', 'logos', 'ctas' each list of (x,y,w,h)
    H, W = base_img.shape[:2]
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Background: scaled down blurred background to avoid hard padding
    bg = cv2.resize(base_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    bg_blur = cv2.GaussianBlur(bg, (31, 31), 0)
    canvas[:] = bg_blur

    # Sort by priority: CTA > Text > Logo
    items = []
    for b in detections.get('ctas', []):
        items.append(('cta', b))
    for b in detections.get('text', []):
        items.append(('text', b))
    for b in detections.get('logos', []):
        items.append(('logo', b))

    # Target anchor regions: left, center, right bands for extremely wide formats
    # and top-middle-bottom bands for tall formats
    wide = target_w / float(target_h) > 2.5
    anchors = []
    if wide:
        band_w = target_w // 3
        anchors = [(0, 0, band_w, target_h),
                   (band_w, 0, band_w, target_h),
                   (2*band_w, 0, target_w - 2*band_w, target_h)]
    else:
        band_h = target_h // 3
        anchors = [(0, 0, target_w, band_h),
                   (0, band_h, target_w, band_h),
                   (0, 2*band_h, target_w, target_h - 2*band_h)]

    placed = []
    for i, (kind, (x, y, w, h)) in enumerate(items):
        roi = base_img[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        # Scale element to fit within anchor with margin
        ax, ay, aw, ah = anchors[min(i, len(anchors)-1)]
        margin = max(6, int(0.03 * min(target_w, target_h)))
        max_w = max(10, aw - 2*margin)
        max_h = max(10, ah - 2*margin)
        scale = min(max_w / float(w), max_h / float(h), 1.2)  # cap enlargement
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)

        # Place with alignment: CTA center, text top-left, logo bottom-left
        if kind == 'cta':
            px = ax + (aw - nw)//2
            py = ay + (ah - nh)//2
        elif kind == 'text':
            px = ax + margin
            py = ay + margin
        else:
            px = ax + margin
            py = ay + ah - nh - margin

        # Avoid overlaps with already placed
        px, py = avoid_overlap((px, py, nw, nh), placed, target_w, target_h)
        # Composite with slight shadow for readability
        shadow = cv2.copyMakeBorder(resized, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0,0,0))
        s_h, s_w = shadow.shape[:2]
        sx = max(0, min(target_w - s_w, px - 3))
        sy = max(0, min(target_h - s_h, py - 3))
        blend_patch(canvas, shadow, sx, sy, alpha=0.35)

        # Place element over shadow
        blend_patch(canvas, resized, px, py, alpha=1.0)
        placed.append((px, py, nw, nh))

    # Content-aware improve: light inpaint around edges of placed elements to soften seams
    return canvas

def avoid_overlap(candidate, placed, W, H):
    px, py, nw, nh = candidate
    for (ox, oy, ow, oh) in placed:
        if rect_overlap((px, py, nw, nh), (ox, oy, ow, oh)):
            # push down or right
            px = min(W - nw, ox + ow + 6)
            py = min(H - nh, oy + oh + 6)
    px = max(0, min(W - nw, px))
    py = max(0, min(H - nh, py))
    return px, py

def rect_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax+aw <= by or ay+ah <= by or ax >= bx+bw or ay >= by+bh)

def blend_patch(canvas, patch, px, py, alpha=1.0):
    h, w = patch.shape[:2]
    x2 = min(canvas.shape[1], px + w)
    y2 = min(canvas.shape[0], py + h)
    w = x2 - px
    h = y2 - py
    if w <= 0 or h <= 0:
        return
    patch = patch[:h, :w]
    roi = canvas[py:y2, px:x2].astype(np.float32)
    patchf = patch.astype(np.float32)
    blended = (alpha * patchf + (1 - alpha) * roi).astype(np.uint8)
    canvas[py:y2, px:x2] = blended

# ------------------------------------------------
# Pipeline to generate each target specification
# ------------------------------------------------
def process_target(img_rgb, target_w, target_h):
    sal = spectral_saliency(img_rgb)
    text_boxes = detect_text_regions(img_rgb)
    logos, ctas = detect_logo_and_cta(img_rgb)

    imp = build_importance_map(img_rgb, sal, text_boxes, logos, ctas)

    # Try smart crop first
    cropped, rect = smart_crop(img_rgb, imp, target_w, target_h)

    # If too much important content lies outside crop, do reposition
    importance_full = imp.mean()
    mask = np.zeros_like(imp)
    x, y, w, h = rect
    mask[y:y+h, x:x+w] = 1.0
    coverage = (imp * mask).sum() / (imp.sum() + 1e-8)

    if coverage < 0.70:
        detections = {'text': text_boxes, 'logos': logos, 'ctas': ctas}
        repositioned = reposition_elements(img_rgb, detections, target_w, target_h)
        final_img = repositioned
    else:
        final_img = cropped

    return final_img

def resize_with_letterbox(img, target_w, target_h, bg_mode='blur'):
    """
    Resize image to target size while preserving aspect ratio.
    Fills remaining space with blurred or averaged background.
    """
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    if bg_mode == 'blur':
        bg = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        canvas[:] = cv2.GaussianBlur(bg, (31, 31), 0)
    elif bg_mode == 'average':
        avg_color = img.mean(axis=(0, 1)).astype(np.uint8)
        canvas[:] = avg_color
    else:
        canvas[:] = (0, 0, 0)  # fallback to black

    x_offset = (target_w - nw) // 2
    y_offset = (target_h - nh) // 2
    canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
    return canvas

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--psd", required=True, help="Path to 1080x1080 PSD")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    base = rasterize_psd(args.psd)

    targets = [
        ("image_970x90", 970, 90),
        ("image_728x90", 728, 90),
        ("image_160x600", 160, 600),
        ("image_468x60", 468, 60),
        ("image_200x200", 200, 200),
        ("image_300x250", 300, 250),
    ]

    for name, w, h in targets:
        out = process_target(base, w, h)
        Image.fromarray(out).save(os.path.join(args.outdir, f"{name}.png"))
        print(f"Saved {name}.png ({w}x{h})")

if __name__ == "__main__":
    main()
