#!/usr/bin/env python3
"""Recreate predictions for the reusable random-split 50-trial A+C runs."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from src.data.serialization import load_samples
from src.benchmark.featurizers import FEATURIZER_REGISTRY
from src.benchmark.evaluation import compute_all_metrics
from xgboost import XGBRegressor
OUT=ROOT/'results/minor_revision_experiments/primary_ablation/reused_ac_predictions';OUT.mkdir(parents=True,exist_ok=True)
def main():
 d=ROOT/'data/splits/CycPeptMPDB_PAMPA/random';tr=load_samples(d/'train.jsonl');te=load_samples(d/'test.jsonl');fz=FEATURIZER_REGISTRY['anchor_aware'](descriptor_set='basic',ablation_mode='chemistry_attachment');fz.fit(tr);X=np.nan_to_num(fz.transform(tr));Xt=np.nan_to_num(fz.transform(te));y=np.array([s.label for s in tr]);yt=np.array([s.label for s in te]);ref=pd.read_csv(ROOT/'results/supplement_5seed/all_per_seed.csv')
 for seed in range(5):
  params=json.loads((ROOT/f'results/supplement_5seed/tuning_A+C_seed{seed}/best_params.json').read_text());m=XGBRegressor(**params,tree_method='hist',verbosity=0,n_jobs=8,random_state=seed);m.fit(X,y,verbose=False);pr=m.predict(Xt);met=compute_all_metrics(yt,pr);z=ref[(ref.setting=='A+C')&(ref.seed==seed)].iloc[0]
  for k in ['r2','rmse','mae','spearman']:
   if not np.isclose(met[k],z[k],rtol=1e-7,atol=1e-7):raise RuntimeError(f'seed {seed} {k} mismatch: {met[k]} vs {z[k]}')
  pd.DataFrame({'sample_id':[s.sample_id for s in te],'y_true':yt,'y_pred':pr,'split':'test','seed':seed,'feature_set':'A+C','estimator':'XGBoost','sequence':[s.sequence for s in te]}).to_csv(OUT/f'seed_{seed}.csv',index=False)
 print('A+C predictions reproduced exactly')
if __name__=='__main__':main()
