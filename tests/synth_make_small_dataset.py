#!/usr/bin/env python3
import h5py, numpy as np
def make_synth(path='synth.h5', N=200, ny=512, nx=512):
    rs = np.random.RandomState(0)
    data = rs.poisson(10.0, size=(N, ny, nx)).astype('uint16')
    pos = np.stack([np.linspace(-2e-6, 2e-6, N), np.linspace(-2e-6, 2e-6, N)], axis=1)
    with h5py.File(path, 'w') as h5:
        e = h5.create_group('/entry')
        e.create_dataset('data', data=data, compression='gzip', compression_opts=4)
        e.create_dataset('pos', data=pos)
        e.create_dataset('energy_eV', data=800.0)
        e.create_dataset('det_dist_m', data=0.9)
        e.create_dataset('pixel_m', data=1.3e-6)
    print('Wrote', path)
if __name__ == '__main__':
    make_synth()
