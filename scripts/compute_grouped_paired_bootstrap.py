#!/usr/bin/env python3
"""Joint grouped bootstrap for primary model-to-model differences."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/minor_revision_experiments';P=pd.read_csv(OUT/'primary_ablation/seed_level_results.csv');CL=dict(pd.read_csv(OUT/'manifest/sequence_cluster_assignments.csv').set_index('sample_id').sequence_cluster_id)
PAIRS={'random':[('A+B+C','A'),('A+B','A'),('A+B+C','A+B'),('A+C','A'),('B+C','B'),('B+C','C')],'sequence_cluster':[('A+B','A'),('A+B','A+B+C'),('A+B+C','A'),('A+C','A'),('B+C','B'),('B+C','C')]}
def met(y,p):return {'r2':r2_score(y,p),'rmse':mean_squared_error(y,p)**.5,'mae':mean_absolute_error(y,p),'spearman':stats.spearmanr(y,p).statistic}
def job(t):
 sp,x,y,seed=t;a=P[(P.split==sp)&(P.feature_set_id==x)&(P.seed==seed)].iloc[0];b=P[(P.split==sp)&(P.feature_set_id==y)&(P.seed==seed)].iloc[0];pa=pd.read_csv(ROOT/a.prediction_file);pb=pd.read_csv(ROOT/b.prediction_file);q=pa[['sample_id','y_true','y_pred']].merge(pb[['sample_id','y_true','y_pred']],on=['sample_id','y_true'],suffixes=('_x','_y'));q['g']=q.sample_id.map(CL) if sp=='sequence_cluster' else q.sample_id.map(dict(zip(q.sample_id,q.sample_id))).astype(str)
 if sp=='random':
  from src.data.serialization import load_samples
  sm={s.sample_id:s.sequence for s in load_samples(ROOT/'data/splits/CycPeptMPDB_PAMPA/random/test.jsonl')};q['g']=q.sample_id.map(sm)
 groups=q.g.unique();rng=np.random.default_rng(20260717+seed);boots={m:[] for m in ['r2','rmse','mae','spearman']}
 for _ in range(2000):
  pick=rng.choice(groups,len(groups),replace=True);ix=np.concatenate([np.flatnonzero(q.g.to_numpy()==g) for g in pick]);yt=q.y_true.to_numpy()[ix];m1=met(yt,q.y_pred_x.to_numpy()[ix]);m0=met(yt,q.y_pred_y.to_numpy()[ix])
  for m in boots:boots[m].append(m1[m]-m0[m])
 return [{'split':sp,'comparison':f'{x} minus {y}','seed':seed,'metric':m,'observed_difference':met(q.y_true,q.y_pred_x)[m]-met(q.y_true,q.y_pred_y)[m],'ci95_low':np.nanquantile(v,.025),'ci95_high':np.nanquantile(v,.975),'bootstrap_unit':'unique peptide sequence' if sp=='random' else 'held-out 70% sequence cluster','replicates':2000,'bootstrap_seed':20260717+seed,'interval':'percentile'} for m,v in boots.items()]
def main():
 tasks=[(sp,x,y,s) for sp,ps in PAIRS.items() for x,y in ps for s in range(5)];rows=[]
 with ProcessPoolExecutor(max_workers=8) as ex:
  fs={ex.submit(job,t):t for t in tasks}
  for f in as_completed(fs):rows+=f.result();print(fs[f],'complete',flush=True)
 pd.DataFrame(rows).to_csv(OUT/'paired_statistics/bootstrap_differences.csv',index=False);print('paired grouped bootstrap complete')
if __name__=='__main__':
 import sys;sys.path[:0]=[str(ROOT),str(ROOT/'src')];main()
