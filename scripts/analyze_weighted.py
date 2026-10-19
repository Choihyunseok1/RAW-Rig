"""Read-only weighted numerical model and independent split validation."""
from pathlib import Path
import csv,json,sys
import numpy as np
from rawrig.noise import verify_phase,channel_slices
from rawrig.noise_analysis import _load
from rawrig.noise_modeling import weighted_gain,weighted_raw_noise
from rawrig.storage import write_json
from rawrig.signal_coverage import signal_coverage

import argparse
parser=argparse.ArgumentParser(description="Weighted conversion gain and split-sample uncertainty (one camera, one gain)")
parser.add_argument("session", type=Path)
args=parser.parse_args()
root=args.session.resolve()
if not root.exists():root=Path(str(root).removesuffix('.partial'))
_,bm=verify_phase(root,'bright');_,dm=verify_phase(root,'dark')
identities={(r['serial'],r['point']['gain_index']) for r in bm['stacks']}
if len(identities)!=1:
    raise ValueError("Use a single-camera, single-gain session; do not combine camera/gain conditions.")
dark={ (r['point']['index'],r['serial']):r for r in dm['stacks'] }
sets={k:dict(s=[],b=[],d=[],nb=[],nd=[],raw=[]) for k in ('full','half0','half1')}
levels=[]
clipping=[]
observed_variance=[]
for br in bm['stacks']:
    bs=_load(root,br); ds=_load(root,dark[br['point']['index'],br['serial']])
    if not levels:
        reference_offset=ds['mean'].copy();reference_read_variance=ds['pair_variance'].copy()
    clipping.append(bs['near_white']/int(bs['count']))
    observed_variance.append(bs['pair_variance'])
    for key,x in sets.items():
        if key=='full':
            raw_mean=bs['mean'];s=raw_mean-ds['mean']; b=bs['pair_variance'];d=ds['pair_variance'];nb=int(bs['count'])//2;nd=int(ds['count'])//2
        else:
            h=int(key[-1]);raw_mean=bs['pair_half_mean'][h];s=raw_mean-ds['pair_half_mean'][h];b=bs['pair_half_variance'][h];d=ds['pair_half_variance'][h];nb=int(bs['pair_half_count'][h]);nd=int(ds['pair_half_count'][h])
        b=np.where(clipping[-1]<.001,b,np.nan)
        for k,v in [('s',s),('b',b),('d',d),('nb',nb),('nd',nd)]:x[k].append(v)
        x['raw'].append(raw_mean)
    levels.append(dict(index=br['point']['index'],exposure_us=bs['metadata']['settings']['exposure_us'],bright_frames=int(bs['count']),dark_frames=int(ds['count'])))
    print('LOADED',len(levels),flush=True)
fits={k:weighted_gain(np.stack(x['s']),np.stack(x['b']),np.stack(x['d']),x['nb'],x['nd']) for k,x in sets.items()}
raw_fits={k:weighted_raw_noise(np.stack(x['raw']),np.stack(x['b']),x['nb'],reference_offset=reference_offset,reference_read_variance=reference_read_variance) for k,x in sets.items()}
maps={f'{key}_{k}':v for key,fit in fits.items() for k,v in fit.items()}
maps.update({f'raw_model_{key}_{k}':v for key,fit in raw_fits.items() for k,v in fit.items()})
channels={}; csv_rows=[]; coverage_rows=[]
for ch,z in channel_slices(bs['metadata']['settings']['pixel_format']).items():
    valid=np.logical_and.reduce([fit['valid'][z] for fit in fits.values()])
    g=fits['full']['gain_dn_per_e'][z][valid];se=fits['full']['gain_standard_error_dn_per_e'][z][valid]
    diff=(fits['half0']['gain_dn_per_e'][z]-fits['half1']['gain_dn_per_e'][z])[valid]
    empirical=float(np.std(diff,ddof=1)/2)
    robust=float(1.4826*np.median(np.abs(diff-np.median(diff)))/2)
    relative=empirical/np.mean(g)
    # Residual-to-model curve independently audited, no forced shape matching.
    x=sets['full'];curve=[]
    for j,level in enumerate(levels):
        signal=float(x['s'][j][z].mean());bv=float(observed_variance[j][z].mean());dv=float(x['d'][j][z].mean())
        clip_fraction=float(clipping[j][z].mean())
        usable=bool(clip_fraction<.001 and np.isfinite(bv) and signal>0)
        row=dict(cfa=ch,**level,signal_dn=signal,signal_e=signal/float(g.mean()),near_white_fraction=clip_fraction,valid_unsaturated=usable,bright_pair_variance_dn2=bv,dark_pair_variance_dn2=dv,
            alpha_mean_dn_per_e=float(g.mean()),predicted_variance_dn2=float(g.mean()*signal+dv),
            variance_model_relative_error=float((bv-dv-g.mean()*signal)/(g.mean()*signal+dv)))
        csv_rows.append(row);curve.append(row)
    channels[ch]=dict(pixels=int(valid.sum()),gain_mean_dn_per_e=float(g.mean()),gain_median_dn_per_e=float(np.median(g)),
        gain_map_sd_dn_per_e=float(np.std(g,ddof=1)),theoretical_gain_map_se_rms_dn_per_e=float(np.sqrt(np.mean(se**2))),
        empirical_full_map_se_dn_per_e=empirical,robust_empirical_full_map_se_dn_per_e=robust,
        empirical_gain_map_relative_se=relative,meets_two_percent_target=bool(relative<=.02),
        half_mean_relative_difference=float((fits['half0']['gain_dn_per_e'][z][valid].mean()-fits['half1']['gain_dn_per_e'][z][valid].mean())/g.mean()),
        fit_reduced_chi_squared_median=float(np.median(fits['full']['reduced_chi_squared'][z][valid])),
        max_curve_variance_model_relative_error=max((abs(row['variance_model_relative_error']) for row in curve if row['valid_unsaturated']),default=None))
    coverage_rows.extend(dict(cfa=ch,**r) for r in signal_coverage([r['exposure_us'] for r in curve],[r['signal_e'] for r in curve],[r['valid_unsaturated'] for r in curve]))
    good=np.logical_and.reduce([fit['valid'][z] for fit in raw_fits.values()])
    alpha=raw_fits['full']['alpha_dn_per_e'][z][good]
    alpha_difference=(raw_fits['half0']['alpha_dn_per_e'][z]-raw_fits['half1']['alpha_dn_per_e'][z])[good]
    beta_difference=(raw_fits['half0']['variance_at_reference_dn2'][z]-raw_fits['half1']['variance_at_reference_dn2'][z])[good]
    channels[ch]['raw_observation_model']=dict(alpha_mean_dn_per_e=float(alpha.mean()),
        alpha_empirical_full_map_se_dn_per_e=float(np.std(alpha_difference,ddof=1)/2),
        alpha_relative_empirical_se=float(np.std(alpha_difference,ddof=1)/2/alpha.mean()),
        conditional_variance_at_reference_mean_dn2=float(raw_fits['full']['variance_at_reference_dn2'][z][good].mean()),
        conditional_variance_at_reference_empirical_se_dn2=float(np.std(beta_difference,ddof=1)/2),
        physical_read_variance_identified=False,formula='variance = alpha * raw_mean + intercept_raw_dn2; reference variance is conditional at saved dark reference mean, not separately identified physical read variance')
maps['gain_empirical_half_difference_dn_per_e']=fits['half0']['gain_dn_per_e']-fits['half1']['gain_dn_per_e']
out=root/'modeling';out.mkdir(exist_ok=True)
np.savez_compressed(out/'weighted_gain_and_uncertainty.npz',**maps)
with (out/'weighted_noise_curves.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
write_json(out/'signal_coverage.json',coverage_rows)
with (out/'signal_coverage.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(coverage_rows[0]));w.writeheader();w.writerows(coverage_rows)
write_json(out/'precision_validation.json',dict(source=str(root),method='Independent whole disjoint frame pairs split by pair index parity, separately for bright and dark. Full-map empirical SD=SD(half0-half1)/2. Heteroscedastic inverse variance through-origin fit, independent pilot reference gain 5.9 DN/e. No RAW normalization or smoothing.',
    uncertainty_scope='Sampling uncertainty; does not bound unmeasured temperature drift, optical PRNU or QE. Two-percent target is empirical per-pixel conversion gain standard error, not all noise parameters.',channels=channels))
print(json.dumps(channels,indent=2),flush=True)
