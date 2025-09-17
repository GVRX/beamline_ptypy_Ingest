#!/usr/bin/env python3
import numpy as np
from ingest.preprocess import select_by_position, mean_center_of_mass
def assert_true(cond, msg):
    if not cond: raise AssertionError(msg)
def test_selection():
    y = np.linspace(-1e-3, 1e-3, 11); x = np.linspace(-2e-3, 2e-3, 21)
    Y, X = np.meshgrid(y, x, indexing='ij'); pos = np.stack([Y.ravel(), X.ravel()], axis=1)
    keep_c = select_by_position(pos, shape='circle', center=(0.0, 0.0), radius=0.5e-3); assert_true(keep_c.sum() > 0, "circle selection yields some points")
    keep_r = select_by_position(pos, shape='rect', center=(0.0, 0.0), size=(1e-3, 1e-3)); assert_true(keep_r.sum() > 0, "rect selection yields some points")
def test_mean_com():
    img1 = np.zeros((64,64), dtype=np.float32); img1[20, 30] = 100
    img2 = np.zeros((64,64), dtype=np.float32); img2[40, 50] = 300
    stack = np.stack([img1, img2], axis=0); cy, cx = mean_center_of_mass(stack)
    assert_true(30 < cy < 45 and 40 < cx < 55, f"mean center in expected range, got {(cy,cx)}")
if __name__ == "__main__":
    test_selection(); test_mean_com(); print("OK: selection & COM tests passed")
