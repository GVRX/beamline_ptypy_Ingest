import numpy as np
from typing import Optional, Tuple, Literal

Grouping = Literal['sum','mean','first','best','none']
FlipOpt = Literal['none','h','v','hv','vh']

def orient(img: np.ndarray,
           rotate: int = 0,
           flip: str | None = None) -> np.ndarray:
    """
    Re-orient a diffraction image.

    Parameters
    ----------
    img : ndarray
        2D image.
    rotate : {0, 90, 180, 270}
        Degrees of **clockwise** rotation to apply.
    flip : {'h', 'horizontal', 'v', 'vertical', None}
        Optional flip applied *after* rotation.

    Returns
    -------
    ndarray
        Reoriented image (shape will swap for 90/270 if img is non-square).
    """
    if rotate not in (0, 90, 180, 270):
        raise ValueError(f"rotate must be one of 0, 90, 180, 270 (got {rotate})")

    out = img
    # numpy.rot90 uses CCW for positive k; we want CW => use negative k
    k = (rotate // 90) % 4
    if k:
        out = np.rot90(out, k=-k)  # negative = clockwise

    if flip in ('h', 'horizontal', True):
        out = np.fliplr(out)
    elif flip in ('v', 'vertical'):
        out = np.flipud(out)
    elif flip is None:
        pass
    else:
        raise ValueError(f"flip must be 'h'/'horizontal', 'v'/'vertical', or None (got {flip})")

    return out

def apply_dark_flat(imgs: np.ndarray,
                    dark: Optional[np.ndarray] = None,
                    flat: Optional[np.ndarray] = None) -> np.ndarray:
    arr = imgs.astype(np.float32, copy=True)
    if dark is not None:
        arr = arr - (dark if arr.ndim == 2 else dark[np.newaxis, ...])
    if flat is not None:
        eps = 1e-8
        arr = arr / ((flat if arr.ndim == 2 else flat[np.newaxis, ...]) + eps)
    return arr

def build_mask(imgs: np.ndarray,
               sat_level: Optional[int] = None,
               hot_sigma: float = 6.0,
               cold_sigma: float = 6.0) -> np.ndarray:
    ref = imgs.mean(axis=0) if imgs.ndim == 3 else imgs
    m = np.zeros_like(ref, dtype=bool)
    if sat_level is not None:
        m |= (ref >= sat_level)
    mu, sig = float(np.mean(ref)), float(np.std(ref))
    if hot_sigma is not None:
        m |= (ref > mu + hot_sigma * sig)
    if cold_sigma is not None:
        m |= (ref < max(0.0, mu - cold_sigma * sig))
    return m

def _resolve_center(center: Optional[Tuple[float, float]], shape: Tuple[int, int], ref: np.ndarray) -> Tuple[int, int]:
    ny, nx = shape
    if center is None:
        Y, X = np.indices(shape)
        tot = ref.sum() + 1e-12
        cy = int((Y * ref).sum() / tot)
        cx = int((X * ref).sum() / tot)
        return cy, cx
    cy, cx = center
    if 0.0 <= cy < 1.0 and 0.0 <= cx < 1.0:
        return int(round(cy * (ny - 1))), int(round(cx * (nx - 1)))
    return int(round(cy)), int(round(cx))

def crop_to_roi(imgs: np.ndarray,
                size: Optional[Tuple[int, int]] = None,
                center: Optional[Tuple[float, float]] = None):
    if size is None:
        return imgs, (0, imgs.shape[-2], 0, imgs.shape[-1])
    ref = imgs if imgs.ndim == 2 else imgs.mean(axis=0)
    ny, nx = ref.shape
    ty, tx = size
    cy, cx = _resolve_center(center, (ny, nx), ref)
    y0 = max(0, min(ny - ty, cy - ty // 2))
    x0 = max(0, min(nx - tx, cx - tx // 2))
    y1 = min(ny, y0 + ty)
    x1 = min(nx, x0 + tx)
    if imgs.ndim == 2:
        return imgs[y0:y1, x0:x1], (y0, y1, x0, x1)
    return imgs[:, y0:y1, x0:x1], (y0, y1, x0, x1)

def group_frames(frames: np.ndarray,
                 method: Grouping = 'sum',
                 exposure_s: Optional[np.ndarray] = None,
                 sat_level: Optional[int] = None) -> np.ndarray:
    if method == 'none':
        return frames
    if method == 'sum':
        return frames.sum(axis=0)
    if method == 'mean':
        return frames.mean(axis=0)
    if method == 'first':
        return frames[0]
    if method == 'best':
        if sat_level is None:
            idx = int(np.argmax(np.median(frames, axis=(1,2))))
            return frames[idx]
        scores = [np.median(np.clip(f, 0, sat_level)) for f in frames]
        return frames[int(np.argmax(scores))]
    raise ValueError('Unknown grouping method: %r' % (method,))

def select_by_position(pos: np.ndarray,
                       shape: str = 'none',
                       center: Optional[Tuple[float, float]] = None,
                       radius: Optional[float] = None,
                       size: Optional[Tuple[float, float]] = None) -> np.ndarray:
    if shape == 'none' or pos is None:
        return np.ones(len(pos), dtype=bool)
    if center is None:
        raise ValueError("select_by_position: center is required for shape != 'none'")
    cy, cx = center
    dy = pos[:,0] - cy
    dx = pos[:,1] - cx
    if shape == 'circle':
        if radius is None:
            raise ValueError("select_by_position: radius required for circle")
        return (dy*dy + dx*dx) <= (radius*radius)
    if shape == 'rect':
        if size is None:
            raise ValueError("select_by_position: size (dy,dx) required for rect")
        sy, sx = size
        return (np.abs(dy) <= sy/2.0) & (np.abs(dx) <= sx/2.0)
    raise ValueError(f"Unknown shape {shape!r}")

def center_of_mass(img: np.ndarray) -> Tuple[float, float]:
    ref = np.asarray(img, dtype=np.float64)
    ref = np.clip(ref, 0, None)
    tot = ref.sum()
    if tot <= 0:
        ny, nx = ref.shape
        return (ny/2.0, nx/2.0)
    ny, nx = ref.shape
    y = np.arange(ny, dtype=np.float64)
    x = np.arange(nx, dtype=np.float64)
    cy = (ref.sum(axis=1) * y).sum() / tot
    cx = (ref.sum(axis=0) * x).sum() / tot
    return (float(cy), float(cx))

def mean_center_of_mass(stack: np.ndarray, weights: Optional[np.ndarray]=None) -> Tuple[float, float]:
    if stack.ndim != 3:
        raise ValueError("mean_center_of_mass expects a stack (N,Y,X)")
    N = stack.shape[0]
    cys, cxs, w = [], [], []
    for i in range(N):
        cy, cx = center_of_mass(stack[i])
        cys.append(cy); cxs.append(cx)
        w.append(stack[i].sum() if weights is None else weights[i])
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, 0, None)
    if w.sum() == 0:
        return (float(np.mean(cys)), float(np.mean(cxs)))
    return (float(np.average(cys, weights=w)), float(np.average(cxs, weights=w)))
