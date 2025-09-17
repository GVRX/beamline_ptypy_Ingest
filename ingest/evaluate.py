#!/usr/bin/env python3
"""
Ptycho Outlier & Contrast Maps — lightweight, platform‑independent tooling for ptychography diffraction stacks (CXI/HDF5).

Additions
---------
- Reject list (top-K, top-p, threshold)
- Scan-space mask (rect/circle)
- STXM and DPC maps
- 4-panel comparison figure
- PtyPy JSON emitter: --emit-ptypy-json path.json
- Thumbnail gallery: --thumbs-topk N [--thumbs-size 256]

Dependencies: numpy, h5py, matplotlib (optional for plotting)
"""
from __future__ import annotations
import argparse, os, sys, math, json, time
from typing import Tuple, Optional, Sequence
import numpy as np, h5py

try:
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False

EPS = 1e-12

def robust_scale(x: np.ndarray, axis: int = 0):
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    mad = np.where(mad < EPS, EPS, mad)
    z = (x - med) / (1.4826 * mad)
    return z, med.squeeze(), mad.squeeze()

def pca_recon_error(X: np.ndarray, k: int = 10) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = max(1, min(k, min(Xc.shape)-1))
    Uk = U[:, :k]; Sk = S[:k]; Vk = Vt[:k, :]
    Xk = (Uk * Sk) @ Vk
    resid = Xc - Xk
    return np.sqrt(np.sum(resid * resid, axis=1)) / math.sqrt(X.shape[1])

def bin_ndarray(np_array: np.ndarray, new_shape: Tuple[int, int], op=np.mean) -> np.ndarray:
    M, N = np_array.shape
    m, n = new_shape
    if M % m != 0 or N % n != 0:
        M2 = (M // m) * m; N2 = (N // n) * n
        A = np_array[:M2, :N2]
    else:
        A = np_array; M2, N2 = M, N
    A = A.reshape(m, M2 // m, n, N2 // n).swapaxes(1, 2)
    return op(A, axis=(2, 3))

def radial_energy_fractions(arr: np.ndarray, rings=((0,0.1),(0.1,0.25),(0.25,0.5),(0.5,1.0))) -> np.ndarray:
    H, W = arr.shape; cy = (H-1)/2.0; cx = (W-1)/2.0
    yy, xx = np.ogrid[:H, :W]
    rr = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    rnorm = rr / (rr.max()+EPS); total = arr.sum()+EPS
    feats = []
    for (r0,r1) in rings:
        mask = (rnorm >= r0) & (rnorm < r1)
        feats.append(arr[mask].sum()/total)
    return np.array(feats, dtype=np.float64)

def center_of_mass(img: np.ndarray):
    s = img.sum()+EPS
    yy, xx = np.indices(img.shape)
    cx = (img*xx).sum()/s; cy = (img*yy).sum()/s
    return float(cx), float(cy)

def normalized_xcorr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()-a.mean(); b = b.ravel()-b.mean()
    sa = np.linalg.norm(a)+EPS; sb = np.linalg.norm(b)+EPS
    return float((a@b)/(sa*sb))

COMMON_DATA_CANDIDATES = ["/entry_1/data/data","/entry_0/data_0/data","/entry/data/data"]
COMMON_POSX_CANDIDATES = ["/entry_1/instrument/positioners/x","/entry_1/instrument/positioners/pos_x","/entry_1/data/scan/x","/entry_1/data/scan/translation_x","/entry_1/data/positions_x"]
COMMON_POSY_CANDIDATES = ["/entry_1/instrument/positioners/y","/entry_1/instrument/positioners/pos_y","/entry_1/data/scan/y","/entry_1/data/scan/translation_y","/entry_1/data/positions_y"]

def auto_find_path(h, candidates):
    for p in candidates:
        if p in h: return p
    return None

def read_positions(h, posx_path, posy_path, n):
    if posx_path is None: posx_path = auto_find_path(h, COMMON_POSX_CANDIDATES)
    if posy_path is None: posy_path = auto_find_path(h, COMMON_POSY_CANDIDATES)
    if posx_path is None or posy_path is None:
        for p in ["/entry_1/data/positions","/entry/data/positions","/positions"]:
            if p in h:
                ds = h[p][...]
                if ds.shape[0] >= n and ds.shape[1] >= 2:
                    return ds[:n,0].astype(float), ds[:n,1].astype(float)
        raise KeyError("Could not find scan position datasets. Provide --pos-x-path and --pos-y-path.")
    x = np.asarray(h[posx_path][...], dtype=np.float64).reshape(-1)[:n]
    y = np.asarray(h[posy_path][...], dtype=np.float64).reshape(-1)[:n]
    return x, y

def iter_frames(ds, batch=32):
    n = ds.shape[0]
    for i in range(0,n,batch):
        j = min(i+batch,n)
        yield i,j,ds[i:j]

def preprocess_frame(img, roi, log=True):
    A = np.asarray(img, dtype=np.float64)
    if roi is not None:
        h,w = A.shape; m = min(h,w,roi)
        y0 = (h-m)//2; x0 = (w-m)//2
        A = A[y0:y0+m, x0:x0+m]
    if log: A = np.log1p(A)
    return A

def downsample(img, target):
    h,w = img.shape
    if h==target and w==target: return img
    return bin_ndarray(img,(target,target),op=np.mean)

def compute_features_and_embeddings(h, data_path, roi, bin_factor, embed_side, batch=32, reference_mode="median"):
    ds = h[data_path]; n = ds.shape[0]
    feats=[]; embeds=[]; sums = np.zeros(n); dpcx=np.zeros(n); dpcy=np.zeros(n)
    ref_accum=[]; ref=None
    if reference_mode in {"mean","median"}:
        sample_count=0
        for i,j,chunk in iter_frames(ds,batch=batch):
            for frame in chunk:
                A = preprocess_frame(frame, roi=roi, log=True); ref_accum.append(A); sample_count+=1
                if sample_count>=min(256,n): break
            if sample_count>=min(256,n): break
        ref = np.mean(ref_accum,axis=0) if reference_mode=="mean" else np.median(ref_accum,axis=0)
    k=0
    for i,j,chunk in iter_frames(ds,batch=batch):
        for frame in chunk:
            A = preprocess_frame(frame, roi=roi, log=True)
            total=float(A.sum()); vmax=float(A.max())
            cx,cy = center_of_mass(A); h0,w0 = A.shape
            dx=(cx-(w0-1)/2.0)/max(1.0,w0-1); dy=(cy-(h0-1)/2.0)/max(1.0,h0-1)
            varx=float(((A*((np.arange(w0)-cx)**2)).sum()/(A.sum()+EPS)))
            vary=float(((A.T*((np.arange(h0)-cy)**2)).sum()/(A.sum()+EPS)))
            rad=radial_energy_fractions(A)
            q=max(4, min(h0,w0)//4); y0=(h0-q)//2; x0=(w0-q)//2
            central=A[y0:y0+q, x0:x0+q]; sc=float(central.std()/(central.mean()+EPS))
            sim=normalized_xcorr(A,ref) if ref is not None else 0.0
            feats.append(np.concatenate([[total,vmax,dx,dy,varx,vary,sc,sim],rad]))
            A2 = A[::bin_factor,::bin_factor] if bin_factor>1 else A
            A3 = downsample(A2, embed_side); embeds.append(A3.ravel())
            sums[k]=total; dpcx[k]=dx; dpcy[k]=dy; k+=1
    return np.vstack(feats), np.vstack(embeds), sums, dpcx, dpcy

def build_scan_mask(px, py, rect=None, circ=None):
    mask = np.ones_like(px, dtype=bool)
    if rect is not None:
        cx,cy,w,h = rect
        mask &= (px>=cx-w/2) & (px<=cx+w/2) & (py>=cy-h/2) & (py<=cy+h/2)
    if circ is not None:
        cx,cy,r = circ
        mask &= ((px-cx)**2 + (py-cy)**2) <= r*r
    return mask

def save_thumbnail_gallery(cxi_path, data_path, indices, outdir, roi, cmap="viridis", thumb_size=256, scores=None, pos=None, title="Top outliers"):
    os.makedirs(outdir, exist_ok=True)
    thumbs=[]
    with h5py.File(cxi_path,"r") as h:
        ds=h[data_path]
        for idx in indices:
            A = preprocess_frame(ds[idx], roi=roi, log=True)
            H,W = A.shape; scale = max(1, int(max(H,W)/thumb_size)); A2 = A[::scale, ::scale]
            png = os.path.join(outdir, f"idx_{int(idx):06d}.png")
            if _HAVE_PLT:
                plt.figure(figsize=(3,3), dpi=120); plt.imshow(A2, cmap=cmap, origin="lower"); plt.axis("off")
                plt.tight_layout(pad=0); plt.savefig(png, bbox_inches="tight", pad_inches=0); plt.close()
            thumbs.append(png)
    html=[
        "<html><head><meta charset='utf-8'><title>Outlier Thumbnails</title>",
        "<style>body{font-family:sans-serif} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px} .card{border:1px solid #ddd;padding:8px;border-radius:8px}</style>",
        "</head><body>", f"<h2>{title}</h2>", "<div class='grid'>"
    ]
    px,py = pos if pos is not None else (None,None)
    for k, idx in enumerate(indices):
        s = float(scores[k]) if scores is not None else None
        pos_text = f"<div>pos=({px[int(idx)]:.4g}, {py[int(idx)]:.4g})</div>" if px is not None else ""
        score_text = f"<div>score={s:.4f}</div>" if s is not None else ""
        html.append(f"<div class='card'><img src='{os.path.basename(thumbs[k])}' width='200'/><div>idx={int(idx)}</div>{score_text}{pos_text}</div>")
    html += ["</div>", "</body></html>"]
    with open(os.path.join(outdir,"index.html"),"w") as f: f.write("\n".join(html))

def main():
    ap = argparse.ArgumentParser(description="Outlier + STXM/DPC map builder for ptychography CXI/HDF5 stacks")
    ap.add_argument("cxi"); ap.add_argument("--data-path",default=None)
    ap.add_argument("--pos-x-path",default=None); ap.add_argument("--pos-y-path",default=None)
    ap.add_argument("--roi",type=int,default=None); ap.add_argument("--bin",dest="bin_factor",type=int,default=1)
    ap.add_argument("--embed-side",type=int,default=32); ap.add_argument("--pca-k",type=int,default=12)
    ap.add_argument("--w-feat",type=float,default=0.5); ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--out",default="ptycho_maps_out"); ap.add_argument("--no-plot",action="store_true")
    ap.add_argument("--save-embeddings",action="store_true")
    ap.add_argument("--reject-topk",type=int,default=0); ap.add_argument("--reject-topp",type=float,default=0.0)
    ap.add_argument("--reject-thresh",type=float,default=None)
    ap.add_argument("--rect",nargs=4,type=float,metavar=("CX","CY","W","H"))
    ap.add_argument("--circ",nargs=3,type=float,metavar=("CX","CY","R"))
    ap.add_argument("--emit-ptypy-json",default=None)
    ap.add_argument("--thumbs-topk",type=int,default=0)
    ap.add_argument("--thumbs-size",type=int,default=256)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    with h5py.File(args.cxi,"r") as h:
        data_path = args.data_path or auto_find_path(h, COMMON_DATA_CANDIDATES)
        if data_path is None: raise KeyError("Could not auto-detect diffraction stack path. Provide --data-path.")
        ds = h[data_path]
        if ds.ndim != 3: raise ValueError(f"Expected stack shape (N,H,W); got {ds.shape}")
        N = ds.shape[0]; px,py = read_positions(h, args.pos_x_path, args.pos_y_path, N)
        mask = build_scan_mask(px, py, rect=args.rect, circ=args.circ)
        feats, embeds, stxm, dpcx, dpcy = compute_features_and_embeddings(h, data_path, roi=args.roi, bin_factor=args.bin_factor, embed_side=args.embed_side, batch=args.batch, reference_mode="median")
    z, med, mad = robust_scale(feats, axis=0)
    feat_dev = np.sqrt((z*z).sum(axis=1)/z.shape[1])
    recon_err = pca_recon_error(embeds, k=args.pca_k)
    def rank_norm(v):
        order = np.argsort(v); ranks = np.empty_like(order); ranks[order] = np.arange(len(v))
        return ranks.astype(np.float64)/max(1,len(v)-1)
    feat_dev_n = rank_norm(feat_dev); recon_err_n = rank_norm(recon_err)
    w = np.clip(args.w_feat,0.0,1.0); score = w*feat_dev_n + (1-w)*recon_err_n
    decision_idx = np.where(mask)[0]; decision_scores = score[decision_idx]
    reject_by = np.zeros(0,dtype=int)
    if args.reject_topk and args.reject_topk>0:
        order = np.argsort(-decision_scores); reject_by = decision_idx[order[:min(args.reject_topk, order.size)]]
    if args.reject_topp and 0<args.reject_topp<1:
        k = int(math.ceil(args.reject_topp*decision_idx.size))
        order = np.argsort(-decision_scores)
        reject_by = np.unique(np.concatenate([reject_by, decision_idx[order[:k]]]))
    if args.reject_thresh is not None:
        reject_by = np.unique(np.concatenate([reject_by, decision_idx[decision_scores>=args.reject_thresh]]))
    accept_by = np.setdiff1d(decision_idx, reject_by, assume_unique=True)
    np.save(os.path.join(args.out,"outlier_score.npy"), score)
    np.save(os.path.join(args.out,"stxm.npy"), stxm)
    np.save(os.path.join(args.out,"dpcx.npy"), dpcx)
    np.save(os.path.join(args.out,"dpcy.npy"), dpcy)
    np.save(os.path.join(args.out,"mask_indices.npy"), decision_idx)
    np.save(os.path.join(args.out,"reject_indices.npy"), reject_by)
    np.save(os.path.join(args.out,"accept_indices.npy"), accept_by)
    import csv
    with open(os.path.join(args.out,"qc_summary.csv"),"w",newline="") as f:
        wtr = csv.writer(f); wtr.writerow(["index","pos_x","pos_y","score","stxm","dpcx","dpcy","masked","rejected"])
        reject_set = set(reject_by.tolist())
        for i in range(len(score)):
            wtr.writerow([i, float(px[i]), float(py[i]), float(score[i]), float(stxm[i]), float(dpcx[i]), float(dpcy[i]), bool(mask[i]), bool(i in reject_set)])
    meta = {"cxi": os.path.abspath(args.cxi), "data_path": data_path, "pos_x_path": args.pos_x_path, "pos_y_path": args.pos_y_path}
    with open(os.path.join(args.out,"qc_meta.json"),"w") as f: json.dump(meta,f,indent=2)
    if args.emit_pty_py_json:
        pjson = {
            "cxi_path": os.path.abspath(args.cxi),
            "data_path": data_path,
            "pos_x_path": args.pos_x_path,
            "pos_y_path": args.pos_y_path,
            "accepted_indices_path": os.path.abspath(os.path.join(args.out, "accept_indices.npy")),
            "rejected_indices_path": os.path.abspath(os.path.join(args.out, "reject_indices.npy")),
            "mask_indices_path": os.path.abspath(os.path.join(args.out, "mask_indices.npy")),
        }
        with open(args.emit_ptypy_json, "w") as f: json.dump(pjson, f, indent=2)
    if _HAVE_PLT and not args.no_plot:
        fig,axs = plt.subplots(2,2, figsize=(9,8), dpi=120, constrained_layout=True)
        ax=axs[0,0]; sc0=ax.scatter(px,py,c=score,s=10,cmap="viridis"); ax.set_title("Deviation score"); ax.set_aspect("equal","box"); plt.colorbar(sc0,ax=ax)
        ax=axs[0,1]; sc1=ax.scatter(px,py,c=stxm,s=10,cmap="viridis"); ax.set_title("STXM (log1p sum)"); ax.set_aspect("equal","box"); plt.colorbar(sc1,ax=ax)
        ax=axs[1,0]; sc2=ax.scatter(px,py,c=dpcx,s=10,cmap="viridis"); ax.set_title("DPC-X (COM shift x)"); ax.set_aspect("equal","box"); plt.colorbar(sc2,ax=ax)
        ax=axs[1,1]; sc3=ax.scatter(px,py,c=dpcy,s=10,cmap="viridis"); ax.set_title("DPC-Y (COM shift y)"); ax.set_aspect("equal","box"); plt.colorbar(sc3,ax=ax)
        plt.suptitle("Ptychography QC Maps", y=1.02); plt.savefig(os.path.join(args.out,"qc_maps.png")); plt.close(fig)
    if args.thumbs_topk and args.thumbs_topk>0:
        order = np.argsort(-decision_scores); top_idx = decision_idx[order[:min(args.thumbs_topk, order.size)]]
        top_scores = score[top_idx]; thumbs_dir = os.path.join(args.out, "thumbs")
        try:
            save_thumbnail_gallery(args.cxi, data_path, top_idx, thumbs_dir, args.roi, thumb_size=args.thumbs_size, scores=top_scores, pos=(px,py), title="Top outliers (masked subset)")
        except Exception as e:
            print(f"[warn] thumbnail gallery failed: {e}")
    print(f"Outputs in: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
