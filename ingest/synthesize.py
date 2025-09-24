import numpy as np, h5py


def make_synthetic_cxi(path, N=256, ny=256, nx=256, seed=0):
    # Generate a tiny synthetic dataset ----

    rng = np.random.default_rng(seed)
    gy = int(np.sqrt(N)); gx = gy
    posy, posx = np.meshgrid(np.linspace(-50e-6, 50e-6, gy), np.linspace(-50e-6, 50e-6, gx), indexing='ij')
    pos = np.stack([posx.ravel(), posy.ravel()], axis=1)[:N]

    yy, xx = np.mgrid[:ny, :nx]
    cy, cx = ny//2, nx//2
    rr2 = (yy-cy)**2 + (xx-cx)**2
    base = np.exp(-(rr2)/(2*(0.07*ny)**2)) * 3e4
    data = np.empty((N, ny, nx), np.float32)
    for i in range(N):
        noise = rng.normal(0, 1200, (ny, nx)).astype(np.float32)
        dy = int(np.round((pos[i,1]/pos[:,1].max())*3)) if pos[:,1].max() != 0 else 0
        dx = int(np.round((pos[i,0]/pos[:,0].max())*3)) if pos[:,0].max() != 0 else 0
        shifted = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
        data[i] = np.clip(shifted + noise, 0, 65535)

    # a few outliers
    if N >= 50:
        bad_idx = rng.choice(N, size=max(2, N//40), replace=False)
        for bad in bad_idx:
            data[bad] *= rng.uniform(0.1, 0.3)

    with h5py.File(path, "w") as h5:
        e = h5.create_group("entry")
        e.create_dataset("data", data=data.astype(np.uint32))
        e.create_dataset("pos", data=pos.astype(np.float64))
        e.create_dataset("energy_eV", data=np.array(930.0, dtype=np.float64))
        e.create_dataset("det_dist_m", data=np.array(1.2, dtype=np.float64))
        e.create_dataset("pixel_m", data=np.array(55e-6, dtype=np.float64))
    return path