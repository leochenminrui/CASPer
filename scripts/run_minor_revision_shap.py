#!/usr/bin/env python3
"""Five-seed SHAP analysis using the selected 50-trial primary XGBoost models."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from src.data.serialization import load_samples
from src.benchmark.featurizers import FEATURIZER_REGISTRY
from xgboost import XGBRegressor
import xgboost as xgb

OUT=ROOT/'results/minor_revision_experiments/shap'; OUT.mkdir(parents=True,exist_ok=True)
MODELS={
 'A':('chem_A_xgb','anchor_aware',{'descriptor_set':'basic','ablation_mode':'chemistry_only'}),
 'B':('site_B_xgb','site_only',{}),
 'A+B':('chem_site_AB_xgb','anchor_aware',{'descriptor_set':'basic','ablation_mode':'chemistry_anchors'}),
 'A+B+C':('full_ABC_xgb','anchor_aware',{'descriptor_set':'basic','ablation_mode':'full'}),
}

def group(name):
    if name in {'mol_weight','logp','tpsa','num_h_acceptors','num_h_donors','num_rotatable_bonds','num_aromatic_rings','num_aliphatic_rings','num_edits','num_edit_families'}: return 'A'
    if name.startswith('anchor_count') or name.startswith('anchor_density') or name.startswith('anchor_pos'): return 'B1'
    if name.startswith('anchor_res_'): return 'B2'
    if name.startswith('anchor_'): return 'B3'
    return 'C'

def ci(a):
    a=np.asarray(a,float);m=a.mean();sd=a.std(ddof=1);q=stats.t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a));return m,sd,m-q,m+q

def main():
    split=ROOT/'data/splits/CycPeptMPDB_PAMPA/random'
    train=load_samples(split/'train.jsonl');test=load_samples(split/'test.jsonl')
    y=np.array([s.label for s in train]); idx=np.random.default_rng(20260717).choice(len(test),min(500,len(test)),replace=False)
    group_rows=[];feat_rows=[]
    for rep,(mid,fkey,kwargs) in MODELS.items():
      fz=FEATURIZER_REGISTRY[fkey](**kwargs);fz.fit(train)
      Xtr=np.nan_to_num(fz.transform(train));Xte=np.nan_to_num(fz.transform(test))[idx];names=fz.get_feature_names()
      for seed in range(5):
        p=ROOT/f'results/minor_revision_experiments/raw_runs/primary_ablation/random/seed_{seed}/{mid}/best_params.json'
        params=json.loads(p.read_text())
        model=XGBRegressor(**params,tree_method='hist',verbosity=0,n_jobs=8,random_state=seed)
        model.fit(Xtr,y,verbose=False)
        # Native XGBoost TreeSHAP avoids SHAP-reader incompatibility with the
        # vector-form base_score emitted by XGBoost 3.x.  Final column is bias.
        sv=np.asarray(model.get_booster().predict(xgb.DMatrix(Xte),pred_contribs=True))[:,:-1]
        ma=np.abs(sv).mean(axis=0); total=ma.sum()
        raw={g:0.0 for g in ['A','B1','B2','B3','C']}
        for n,v in zip(names,ma):raw[group(n)]+=float(v)
        raw={k:v/total for k,v in raw.items()}
        raw['B']=raw['B1']+raw['B2']+raw['B3']
        for g,v in raw.items():group_rows.append({'representation_id':rep,'representation':rep.replace('A','Chemistry').replace('B','Site').replace('C','Context'),'seed':seed,'group':g,'attribution_proportion':v,'n_explained':len(idx),'evaluation_population':'fixed random-split test subset','shap_method':'XGBoost native TreeSHAP pred_contribs'})
        for n,v in zip(names,ma):feat_rows.append({'representation_id':rep,'seed':seed,'feature':n,'subblock':group(n),'mean_absolute_shap':float(v),'proportion':float(v/total)})
    gd=pd.DataFrame(group_rows);gd.to_csv(OUT/'seed_level_group_attribution.csv',index=False)
    sums=[]
    for (rep,g),z in gd.groupby(['representation_id','group']):
      m,sd,lo,hi=ci(z.attribution_proportion);sums.append({'representation_id':rep,'group':g,'mean':m,'sample_sd':sd,'ci95_low':lo,'ci95_high':hi,'n_seeds':5})
    sd=pd.DataFrame(sums);sd[sd.group.isin(['A','B','C'])].to_csv(OUT/'group_summary_with_ci.csv',index=False);sd[sd.group.isin(['B1','B2','B3'])].to_csv(OUT/'subgroup_summary_with_ci.csv',index=False)
    fd=pd.DataFrame(feat_rows); top=[]
    for rep,z in fd.groupby('representation_id'):
      q=z.groupby(['feature','subblock'],as_index=False).proportion.mean()
      selections=[('overall',q.nlargest(15,'proportion'))]
      for g in ['A','B','C']:
        if g=='B': u=q[q.subblock.isin(['B1','B2','B3'])]
        else:u=q[q.subblock==g]
        selections.append((g,u.nlargest(10,'proportion')))
      for scope,u in selections:
        for rank,(_,r) in enumerate(u.iterrows(),1):top.append({'representation_id':rep,'scope':scope,'rank':rank,**r.to_dict()})
    pd.DataFrame(top).to_csv(OUT/'top_features.csv',index=False)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    full=sd[sd.representation_id=='A+B+C']
    for groups,file,title in [(['A','B','C'],'conceptual_group_attribution.png','Conceptual groups'),(['A','B1','B2','B3','C'],'subblock_attribution.png','Descriptor subblocks')]:
      q=full.set_index('group').loc[groups];fig,ax=plt.subplots(figsize=(7,4));ax.bar(groups,q['mean'],yerr=[q['mean']-q.ci95_low,q.ci95_high-q['mean']],capsize=4);ax.set_ylabel('Absolute SHAP attribution proportion');ax.set_title(title);fig.tight_layout();fig.savefig(OUT/'plots'/file,dpi=300);plt.close(fig)
    print('five-seed SHAP complete')
if __name__=='__main__':main()
