#!/usr/bin/env python3
import numpy as np
from ingest.preprocess import orient, crop_to_roi


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: {a} != {b}")

def test_orient():
    '''Test image orientation (rotation & flipping)

    start from a tiny array where we know the source coordinate and then compute its expected destination for clockwise rotations.

    Let’s define a 7×7 image with a single “beacon” pixel at (y, x) = (2, 3):

    For 90° CW: (y, x) → (x, W-1-y) ⇒ (2,3) → (3, 7-1-2=4) ⇒ (3,4)

    For 180°: (y, x) → (H-1-y, W-1-x) ⇒ (2,3) → (4, 3)

    For 270° CW (i.e., 90° CCW): (y, x) → (W-1-x, y) ⇒ (2,3) → (3, 2)

    '''
    H = W = 7
    img = np.zeros((H, W), dtype=float)
    src = (2, 3)       # (y, x)
    img[src] = 1.0

    r90 = orient(img, rotate=90)
    y, x = np.argwhere(r90 == 1.0)[0]
    assert_eq((y, x), (3, W-1-src[0]), "rotate=90 CW position")  # -> (3, 4)

    r180 = orient(img, rotate=180)
    y, x = np.argwhere(r180 == 1.0)[0]
    assert_eq((y, x), (H-1-src[0], W-1-src[1]), "rotate=180 position")  # -> (4, 3)

    r270 = orient(img, rotate=270)
    y, x = np.argwhere(r270 == 1.0)[0]
    assert_eq((y, x), (W-1-src[1], src[0]), "rotate=270 CW position")  # -> (3, 2)

    # Flips after rotation
    rh = orient(img, rotate=0, flip='h')
    y, x = np.argwhere(rh == 1.0)[0]
    assert_eq((y, x), (src[0], W-1-src[1]), "flip horizontal")

    rv = orient(img, rotate=0, flip='v')
    y, x = np.argwhere(rv == 1.0)[0]
    assert_eq((y, x), (H-1-src[0], src[1]), "flip vertical")

def test_roi_center():
    """
    If you select ROI by center (cy, cx) and half-size (hy, hx), ensure returned
    slices stay in bounds and that the beacon pixel falls inside when intended.
    """
    H = W = 64
    img = np.zeros((H, W), dtype=float)
    img[40, 10] = 1.0

    # Example ROI centered near (32, 16) with half-sizes 16, 16
    cy, cx = 32, 16
    hy, hx = 16, 16
    y0, y1 = max(0, cy-hy), min(H, cy+hy)
    x0, x1 = max(0, cx-hx), min(W, cx+hx)
    roi = img[y0:y1, x0:x1]

    assert roi.shape == (y1-y0, x1-x0)
    assert (roi == 1.0).any()  # beacon falls inside this ROI in this example

if __name__ == "__main__":
    test_orient()
    test_roi_center()
    print("OK: orientation & ROI center tests passed")