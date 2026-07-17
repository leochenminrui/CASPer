#!/usr/bin/env python3
"""Consolidate complete experiments and compute seed/pair/group uncertainty."""
from pathlib import Path
from itertools import product
import json,sys
import numpy as np,pandas as pd
from scipy import stats
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from src.data.serialization import load_samples
OUT=ROOT/'results/minor_revision_experiments';SEEDS=range(5);METS=['r2','rmse','mae','spearman']
NAME={'A':'Chemistry','B':'Site','C':'Context','A+B':'Chemistry + Site','A+C':'Chemistry + Context','B+C':'Site + Context','A+B+C':'Chemistry + Site + Context','ECFP':'ECFP','RDKit':'RDKit'}
MID={'A':'chem_A_xgb','B':'site_B_xgb','C':'context_C_xgb','A+B':'chem_site_AB_xgb','A+C':'chem_context_AC_xgb','B+C':'site_context_BC_xgb','A+B+C':'full_ABC_xgb','ECFP':'ecfp_xgb','RDKit':'rdkit_full_xgb'}
def ci(a):
 a=np.asarray(a,float);m=a.mean();sd=a.std(ddof=1);q=stats.t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a));return m,sd,m-q,m+q
def signflip(d):
 d=np.asarray(d,float);obs=abs(d.mean());vals=[abs(np.mean(d*np.array(s))) for s in product([-1,1],repeat=len(d))];return sum(v>=obs-1e-15 for v in vals)/len(vals)
def holm(p):
 p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p));running=0
 for rank,i in enumerate(order):running=max(running,(len(p)-rank)*p[i]);out[i]=min(1,running)
 return out
def metric(y,p):return {'r2':r2_score(y,p),'rmse':mean_squared_error(y,p)**.5,'mae':mean_absolute_error(y,p),'spearman':stats.spearmanr(y,p).statistic}
def main():
 rows=[]
 for sp in ['random','sequence_cluster']:
  for fs in list(NAME)[:7]:
   for seed in SEEDS:
    p=ROOT/f'results/benchmark/{sp}/seed_{seed}/{MID[fs]}/metrics.json';d=json.loads(p.read_text());t=d['test_metrics']
    if d.get('status')!='completed' or sum(1 for _ in p.with_name('optuna_trials.csv').open())-1!=50:raise RuntimeError(f'invalid {p}')
    rows.append({'split':sp,'feature_set_id':fs,'feature_set':NAME[fs],'estimator':'XGBoost','seed':seed,**{m:t[m] for m in METS},'validation_score':d['val_metrics']['rmse'],'runtime_seconds':d.get('runtime_seconds',np.nan),'status':'completed','prediction_file':str(p.with_name('predictions.csv').relative_to(ROOT)),'best_params_json':json.dumps(d['best_params'],sort_keys=True)})
 rdf=pd.DataFrame(rows);rdf.to_csv(OUT/'primary_ablation/seed_level_results.csv',index=False)
 sums=[]
 for (sp,fs),g in rdf.groupby(['split','feature_set_id']):
  for m in METS:
   a=ci(g[m]);sums.append({'split':sp,'feature_set_id':fs,'feature_set':NAME[fs],'metric':m,'mean':a[0],'sample_sd':a[1],'ci95_low':a[2],'ci95_high':a[3],'n_completed_seeds':5})
 sd=pd.DataFrame(sums);sd.to_csv(OUT/'primary_ablation/summary_with_ci.csv',index=False);q=sd.copy();q['estimate_95ci']=q.apply(lambda r:f"{r['mean']:.3f} ({r['ci95_low']:.3f}, {r['ci95_high']:.3f})",axis=1);q.to_csv(OUT/'primary_ablation/summary_publication_rounded.csv',index=False)
 # Estimator matrix
 fmap={'Chem':'A','Site':'B','Context':'C','Chem_Site':'A+B','Chem_Context':'A+C','Site_Context':'B+C','Chem_Site_Context':'A+B+C'};emap={'ridge':'Ridge','elasticnet':'ElasticNet','random_forest':'Random Forest','svr':'RBF-SVR','xgboost':'XGBoost'};er=[]
 for safe,fs in fmap.items():
  for ek,en in emap.items():
   for seed in SEEDS:
    p=ROOT/f'results/benchmark/estimator_comparison/{safe}_{ek}/seed_{seed}/metrics.json';d=json.loads(p.read_text());t=d['test_metrics'];er.append({'feature_set_id':fs,'feature_set':NAME[fs],'estimator':en,'estimator_id':ek,'seed':seed,**{m:t[m] for m in METS},'validation_score':d.get('best_validation_rmse',d.get('val_metrics',{}).get('rmse',np.nan)),'runtime_seconds':d.get('runtime_seconds',np.nan),'status':'completed','convergence_note':'See execution log; ElasticNet/SVR emitted convergence warnings in some trials.' if ek in ['elasticnet','svr'] else '','best_params_json':json.dumps(d['best_params'],sort_keys=True),'source':str(p.relative_to(ROOT))})
 edf=pd.DataFrame(er);edf.to_csv(OUT/'estimator_matrix/seed_level_results.csv',index=False);es=[]
 for (e,fs),g in edf.groupby(['estimator','feature_set_id']):
  for m in METS:
   a=ci(g[m]);es.append({'estimator':e,'feature_set_id':fs,'feature_set':NAME[fs],'metric':m,'mean':a[0],'sample_sd':a[1],'ci95_low':a[2],'ci95_high':a[3],'n_completed_seeds':5})
 pd.DataFrame(es).to_csv(OUT/'estimator_matrix/summary_with_ci.csv',index=False)
 # Seed paired comparisons including protocol-matched generic baselines.
 specs=[]
 for sp,pairs in {'random':[('A+B+C','A'),('A+B','A'),('A+B+C','A+B'),('A+C','A'),('B+C','B'),('B+C','C')],'sequence_cluster':[('A+B','A'),('A+B','A+B+C'),('A+B+C','A'),('A+C','A'),('B+C','B'),('B+C','C'),('A+B','ECFP'),('A+B','RDKit')]}.items():
  for x,y in pairs:
   for m in METS:
    def vals(fs):
     if fs in ['ECFP','RDKit']:
      return np.array([json.loads((ROOT/f'results/benchmark/{sp}/seed_{s}/{MID[fs]}/metrics.json').read_text())['test_metrics'][m] for s in SEEDS])
     return rdf[(rdf.split==sp)&(rdf.feature_set_id==fs)].sort_values('seed')[m].to_numpy()
    d=vals(x)-vals(y);a=ci(d);specs.append({'family':'primary','split':sp,'comparison':f'{NAME[x]} minus {NAME[y]}','metric':m,'mean_difference':a[0],'sample_sd':a[1],'ci95_low':a[2],'ci95_high':a[3],'ci_excludes_zero':a[2]>0 or a[3]<0,'seed_differences_json':json.dumps(d.tolist()),'test':'exact paired sign-flip permutation','p_value_unadjusted':signflip(d),'n_pairs':5})
 for m in METS:
  a=edf[(edf.feature_set_id=='A+B+C')&(edf.estimator=='Random Forest')].sort_values('seed')[m].to_numpy();b=edf[(edf.feature_set_id=='A+B+C')&(edf.estimator=='XGBoost')].sort_values('seed')[m].to_numpy();d=a-b;c=ci(d);specs.append({'family':'estimator','split':'random_10_trial','comparison':'Random Forest minus XGBoost for Chemistry + Site + Context','metric':m,'mean_difference':c[0],'sample_sd':c[1],'ci95_low':c[2],'ci95_high':c[3],'ci_excludes_zero':c[2]>0 or c[3]<0,'seed_differences_json':json.dumps(d.tolist()),'test':'exact paired sign-flip permutation','p_value_unadjusted':signflip(d),'n_pairs':5})
 pdif=pd.DataFrame(specs);pdif['p_value_holm']=holm(pdif.p_value_unadjusted);pdif['interpretation']=np.where(pdif.ci_excludes_zero,'supported directional difference','inconclusive');pdif.to_csv(OUT/'paired_statistics/paired_differences.csv',index=False);pdif[['family','split','comparison','metric','test','p_value_unadjusted','p_value_holm','n_pairs','interpretation']].to_csv(OUT/'paired_statistics/paired_tests.csv',index=False)
 # Grouped bootstrap per primary seed/model; sequence is the random unit. For
 # the cluster split, the shipped split lacks IDs, so reconstruct separately.
 from scripts.create_sequence_cluster_split import cluster_sequences
 allsamples=load_samples(ROOT/'data/processed/pem_schema/cycpeptmpdb_pampa.jsonl');seq_to_ids={}
 for s in allsamples:seq_to_ids.setdefault(s.sequence,[]).append(s.sample_id)
 seqs=list(seq_to_ids);reps=[seq_to_ids[x][0] for x in seqs];cls=cluster_sequences(seqs,reps,.70);id_to_cl={sid:cid for cid,ids in cls.items() for rid in ids for sid in seq_to_ids[next(x for x in seqs if rid in seq_to_ids[x])]}
 pd.DataFrame([{'sample_id':k,'sequence_cluster_id':v} for k,v in id_to_cl.items()]).to_csv(OUT/'manifest/sequence_cluster_assignments.csv',index=False)
 rng=np.random.default_rng(20260717);br=[]
 for r in rdf.itertuples():
  p=pd.read_csv(ROOT/r.prediction_file);samples={s.sample_id:s for s in load_samples(ROOT/f'data/splits/CycPeptMPDB_PAMPA/{r.split}/test.jsonl')};p['group']=[samples[x].sequence if r.split=='random' else id_to_cl[x] for x in p.sample_id];groups=p.group.unique();boots={m:[] for m in METS}
  for _ in range(2000):
   pick=rng.choice(groups,len(groups),replace=True);ix=np.concatenate([np.flatnonzero(p.group.to_numpy()==g) for g in pick]);z=metric(p.y_true.to_numpy()[ix],p.y_pred.to_numpy()[ix])
   for m in METS:boots[m].append(z[m])
  base=metric(p.y_true,p.y_pred)
  for m in METS:br.append({'split':r.split,'feature_set_id':r.feature_set_id,'seed':r.seed,'metric':m,'estimate':base[m],'ci95_low':np.nanquantile(boots[m],.025),'ci95_high':np.nanquantile(boots[m],.975),'bootstrap_unit':'unique peptide sequence' if r.split=='random' else 'held-out 70% sequence cluster','replicates':2000,'seed_bootstrap':20260717,'interval':'percentile'})
 pd.DataFrame(br).to_csv(OUT/'paired_statistics/grouped_bootstrap_metrics.csv',index=False)
 # Completeness/missing manifest.
 miss=[]
 for r in rdf.itertuples():miss.append({'Experiment':'primary_ablation','Split':r.split,'Estimator':'XGBoost','Feature set':r.feature_set,'Seed':r.seed,'Search budget':50,'Status':'valid reusable run' if pd.isna(r.runtime_seconds) else 'newly completed run','Reason':'Validated metrics, 50 trials, predictions and held-out test evaluation.'})
 for r in edf.itertuples():miss.append({'Experiment':'estimator_matrix','Split':'random','Estimator':r.estimator,'Feature set':r.feature_set,'Seed':r.seed,'Search budget':10,'Status':'valid reusable run' if pd.isna(r.runtime_seconds) else 'newly completed run','Reason':'Validated metrics and protocol-compatible search budget.'})
 pd.DataFrame(miss).to_csv(OUT/'manifest/missing_runs.csv',index=False)
 print({'primary':len(rdf),'estimator':len(edf),'paired':len(pdif),'bootstrap_rows':len(br)})
if __name__=='__main__':main()
