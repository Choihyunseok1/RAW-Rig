"""Numerical CFA covariance decomposition under an explicit additive model."""
from pathlib import Path
import json,csv,sys
import numpy as np
from rawrig.noise import channel_slices
import argparse
parser=argparse.ArgumentParser(description="CFA additive covariance components from RAW frame pairs")
parser.add_argument("session", type=Path)
parser.add_argument("--bright-points", nargs="+", type=int, default=[8,19,32])
args=parser.parse_args()
root=args.session.resolve()
from rawrig.noise import verify_phase
for phase in ('dark','bright'):
    _,manifest=verify_phase(root,phase)
    if len({(r['serial'],r['point']['gain_index']) for r in manifest['stacks']})!=1:
        raise ValueError("Use a single-camera, single-gain session.")
out=root/'modeling/current_noise_covariance';out.mkdir(exist_ok=True)
rows=[];arrays={}
for phase in ('dark','bright'):
    for record in json.loads((root/f'{phase}.json').read_text())['stacks']:
        point=record['point']['index']
        if point not in ([0] if phase=='dark' else args.bright_points):continue
        meta=json.loads((root/record['metadata']).read_text());n=meta['count']
        raw=np.memmap(root/record['raw'],dtype='<u2',mode='r',shape=(n,*meta['stored_shape']))
        planes=channel_slices(meta['settings']['pixel_format']);P=np.zeros((4,4));R=P.copy();C=P.copy();G=P.copy()
        for start in range(0,n,20):
            a=np.asarray(raw[start:start+20],dtype=float);d=(a[::2]-a[1::2])/np.sqrt(2)
            v=np.stack([d[:,z[0],z[1]] for z in planes.values()],axis=-1)
            nr,nc=v.shape[1:3];rm=v.mean(axis=2);cm=v.mean(axis=1);gm=v.mean(axis=(1,2))
            P+=np.einsum('frci,frcj->ij',v,v)/(nr*nc)
            R+=np.einsum('fri,frj->ij',rm,rm)/nr
            C+=np.einsum('fci,fcj->ij',cm,cm)/nc
            G+=np.einsum('fi,fj->ij',gm,gm)
        P/=n//2;R/=n//2;C/=n//2;G/=n//2
        residual=(P-R-C+G)/((1-1/nr)*(1-1/nc))
        row=(R-G)/(1-1/nr)-residual/nc
        column=(C-G)/(1-1/nc)-residual/nr
        common=G-row/nr-column/nc-residual/(nr*nc)
        components={'observed_pixel_covariance':P,'observed_row_mean_covariance':R,'observed_column_mean_covariance':C,'observed_frame_mean_covariance':G,'model_pixel_residual_covariance':residual,'model_row_pair_covariance':row,'model_column_pair_covariance':column,'model_frame_common_covariance':common}
        for name,cov in components.items():
            arrays[f'{phase}_point_{point:03d}_{name}_dn2']=cov
            eigen=np.linalg.eigvalsh(cov)
            for i,ch in enumerate(planes):
                for j,ch2 in enumerate(planes):rows.append(dict(phase=phase,point=point,exposure_us=meta['settings']['exposure_us'],component=name,cfa_a=ch,cfa_b=ch2,covariance_dn2=float(cov[i,j]),minimum_eigenvalue_dn2=float(eigen.min()),positive_semidefinite_with_tolerance=bool(eigen.min()>=-1e-8)))
        del raw
with (out/'additive_noise_component_covariances.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
np.savez_compressed(out/'additive_noise_component_covariances.npz',**arrays)
(out/'noise_component_model.json').write_text(json.dumps(dict(source=str(root),CFA_order=['R','G1','G2','B'],units='stored DN^2',model='epsilon[row_pair,column_pair,CFA] = frame_common_CFA + row_pair_CFA + column_pair_CFA + pixel_residual_CFA',method='Method of moments from unchanged adjacent disjoint RAW differences / sqrt(2). CFA cell grid 128x128. Correct finite spatial averaging analytically. Each component is a 4x4 CFA covariance matrix.',cautions=['Additive separable covariance is a modeling approximation, not identification of physical circuit sources.','Row_pair shares two full Bayer sensor rows; column_pair shares two full sensor columns.','Bright components include shot noise and any remaining LED fluctuation; use dark components for correlated read-noise modeling.','Negative eigenvalues are reported unchanged: a component may be unresolved or the separable assumption may be invalid. Do not simulate an indefinite covariance matrix.','No covariance shrinkage, clipping or forced independence is applied.']),indent=2))
print('Component covariance rows',len(rows))
