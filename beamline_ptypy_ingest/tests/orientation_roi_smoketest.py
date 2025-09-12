#!/usr/bin/env python3
import numpy as np
from beamline_ptypy_ingest.preprocess import orient, crop_to_roi
def assert_eq(a, b, msg):
    if a != b: raise AssertionError(f"{msg}: {a} != {b}")
def test_orient():
    img = np.zeros((7,9), dtype=np.float32); img[2,3] = 1.0
    stack = np.stack([img, img], axis=0)
    r90 = orient(img, rotate=90); y, x = np.argwhere(r90==1.0)[0]; assert_eq((y,x), (3, 7-1-2), "rotate=90 position")
    r180 = orient(img, rotate=180); y, x = np.argwhere(r180==1.0)[0]; assert_eq((y,x), (7-1-2, 9-1-3), "rotate=180 position")
    r270 = orient(img, rotate=270); y, x = np.argwhere(r270==1.0)[0]; assert_eq((y,x), (9-1-3, 2), "rotate=270 position")
    fh = orient(img, flip='h'); y, x = np.argwhere(fh==1.0)[0]; assert_eq((y,x), (2, 9-1-3), "flip=h position")
    fv = orient(img, flip='v'); y, x = np.argwhere(fv==1.0)[0]; assert_eq((y,x), (7-1-2, 3), "flip=v position")
    fvh = orient(img, flip='hv'); y, x = np.argwhere(fvh==1.0)[0]; assert_eq((y,x), (7-1-2, 9-1-3), "flip=hv position")
    r90s = orient(stack, rotate=90); ys, xs = np.argwhere(r90s[0]==1.0)[0]; assert_eq((ys,xs), (3, 7-1-2), "stack rotate=90 position")
def test_roi_center():
    ny, nx = 64, 96; img = np.zeros((ny, nx), dtype=np.float32); cy, cx = 20, 30
    img[cy,:]=1; img[:,cx]=1; stack = np.stack([img, img*2], axis=0)
    cropped, (y0,y1,x0,x1) = crop_to_roi(stack, size=(32,32), center=(cy, cx))
    assert_eq((y1-y0, x1-x0), (32,32), "ROI size"); assert_eq((y0, x0), (cy-16, cx-16), "ROI origin from center")
    cropped2, (y0b,y1b,x0b,x1b) = crop_to_roi(img, size=(32,32), center=(0.5, 0.5))
    assert_eq((y1b-y0b, x1b-x0b), (32,32), "ROI size (frac)"); assert_eq((y0b, x0b), (ny//2-16, nx//2-16), "ROI origin (frac center)")
if __name__ == "__main__":
    test_orient(); test_roi_center(); print("OK: orientation & ROI center tests passed")
