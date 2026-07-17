#!/usr/bin/env python3
"""Build revision audit/statistical artifacts without altering manuscript files."""
from __future__ import annotations
import ast, csv, hashlib, json, os, platform, subprocess, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results/minor_revision_experiments"
SEEDS = [0, 1, 2, 3, 4]
NAME = {"A":"Chemistry","B":"Site","C":"Context","A+B":"Chemistry + Site",
        "A+C":"Chemistry + Context","B+C":"Site + Context","A+B+C":"Chemistry + Site + Context"}
MID = {"A":"chem_A_xgb","B":"site_B_xgb","C":"context_C_xgb","A+B":"chem_site_AB_xgb",
       "A+C":"chem_context_AC_xgb","B+C":"site_context_BC_xgb","A+B+C":"full_ABC_xgb"}
METRICS = ["r2","rmse","mae","spearman"]

def mkdirs():
    for d in ["manifest","primary_ablation/predictions","primary_ablation/best_params",
              "estimator_matrix/best_params","paired_statistics","scaffold_ranking","time_forward/plots",
              "shap/plots","feature_inventory"]:
        (OUT/d).mkdir(parents=True, exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def ci_t(a):
    a=np.asarray(a,float); n=len(a); m=float(np.mean(a))
    if n<2:return m,np.nan,np.nan,np.nan
    sd=float(np.std(a,ddof=1)); q=stats.t.ppf(.975,n-1)*sd/np.sqrt(n)
    return m,sd,m-q,m+q

def manifest():
    files=[ROOT/'data/processed/pem_schema/cycpeptmpdb_pampa.jsonl']
    files += sorted((ROOT/'data/splits/CycPeptMPDB_PAMPA').rglob('*.jsonl'))
    checks={str(p.relative_to(ROOT)):sha(p) for p in files}
    (OUT/'manifest/dataset_checksums.json').write_text(json.dumps(checks,indent=2)+"\n")
    freeze=subprocess.check_output(['uv','pip','freeze','--python',str(ROOT/'.venv/bin/python')],text=True)
    (OUT/'manifest/environment.txt').write_text(freeze)
    cfg=(ROOT/'configs/benchmark/benchmark_full.yaml').read_text()
    man={"git_commit":subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         "timestamp_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,
         "os":platform.platform(),"cpu":platform.processor(),"gpu":"none detected / CPU execution",
         "dataset_checksum":checks['data/processed/pem_schema/cycpeptmpdb_pampa.jsonl'],
         "split_checksums":{k:v for k,v in checks.items() if '/splits/' in k},"seeds":SEEDS,
         "primary_optuna_trials":50,"estimator_matrix_optuna_trials":10,
         "optuna_and_estimator_config":"configs/benchmark/benchmark_full.yaml","config_snapshot":cfg,
         "dependency_change":"biopython==1.87 added to runtime environment for existing split tests; requirements.txt unchanged",
         "output_root":str(OUT.relative_to(ROOT))}
    (OUT/'manifest/run_manifest.json').write_text(json.dumps(man,indent=2)+"\n")

def audit_and_primary():
    audit=[]; rows=[]; reused=0
    supp=pd.read_csv(ROOT/'results/supplement_5seed/all_per_seed.csv')
    for split in ['random','sequence_cluster']:
      for fs in NAME:
       for seed in SEEDS:
        p=ROOT/f'results/benchmark/{split}/seed_{seed}/{MID[fs]}/metrics.json'
        source=None; d=None; trials=50
        if p.exists():
            x=json.loads(p.read_text())
            if x.get('status')=='completed' and (p.with_name('optuna_trials.csv').exists()) and sum(1 for _ in open(p.with_name('optuna_trials.csv')))-1==50:
                d=x; source=str(p.relative_to(ROOT))
        if d is None and split=='random' and fs=='A+C':
            q=supp[(supp.setting=='A+C')&(supp.seed==seed)]
            tp=ROOT/f'results/supplement_5seed/tuning_A+C_seed{seed}/optuna_trials.csv'
            if len(q)==1 and tp.exists() and sum(1 for _ in open(tp))-1==50:
                z=q.iloc[0]; d={'test_metrics':{m:float(z[m]) for m in METRICS},'best_params':json.loads(tp.with_name('best_params.json').read_text()),'val_metrics':{}}
                source=str(tp.parent.relative_to(ROOT));
        status='reusable' if d else 'BLOCKED'
        reason='' if d else 'Missing 50-trial protocol-matched result; measured XGBoost trial runtime (~31 s) makes remaining full run infeasible in this session.'
        audit.append({'Experiment':'primary_ablation','Split':split,'Feature set':NAME[fs],'Estimator':'XGBoost','Seeds':seed,'Optuna trials':50,'Existing status':status,'Reusable':bool(d),'Source or blocker':source or reason})
        if d:
            reused+=1; tm=d['test_metrics']; rows.append({'split':split,'feature_set_id':fs,'feature_set':NAME[fs],'estimator':'XGBoost','seed':seed,**{m:tm[m] for m in METRICS},'validation_score':d.get('val_metrics',{}).get('rmse',np.nan),'runtime_seconds':np.nan,'status':'completed_reused','prediction_file':str(p.with_name('predictions.csv').relative_to(ROOT)) if p.exists() else '', 'source':source,'best_params_json':json.dumps(d.get('best_params',{}),sort_keys=True)})
    pd.DataFrame(audit).to_csv(OUT/'manifest/protocol_audit.csv',index=False)
    rdf=pd.DataFrame(rows); rdf.to_csv(OUT/'primary_ablation/seed_level_results.csv',index=False)
    sums=[]
    for (sp,fs),g in rdf.groupby(['split','feature_set_id']):
      for m in METRICS:
        mean,sd,lo,hi=ci_t(g[m]); sums.append({'split':sp,'feature_set_id':fs,'feature_set':NAME[fs],'metric':m,'mean':mean,'sample_sd':sd,'ci95_low':lo,'ci95_high':hi,'n_completed_seeds':len(g)})
    sdf=pd.DataFrame(sums); sdf.to_csv(OUT/'primary_ablation/summary_with_ci.csv',index=False)
    pub=sdf.copy(); pub['estimate_95ci']=pub.apply(lambda r:f"{r['mean']:.3f} ({r['ci95_low']:.3f}, {r['ci95_high']:.3f})",axis=1)
    pub.to_csv(OUT/'primary_ablation/summary_publication_rounded.csv',index=False)
    return rdf,reused

def estimator_matrix():
    rows=[]
    fmap={'Chem':'A','Site':'B','Context':'C','Chem_Site':'A+B','Chem_Context':'A+C','Site_Context':'B+C','Chem_Site_Context':'A+B+C'}
    emap={'ridge':'Ridge','elasticnet':'ElasticNet','random_forest':'Random Forest','svr':'RBF-SVR','xgboost':'XGBoost'}
    for safe,fs in fmap.items():
      for ek,en in emap.items():
       for seed in SEEDS:
        p=ROOT/f'results/benchmark/estimator_comparison/{safe}_{ek}/seed_{seed}/metrics.json'
        if p.exists():
          d=json.loads(p.read_text()); tm=d.get('test_metrics',{})
          rows.append({'feature_set_id':fs,'feature_set':NAME[fs],'estimator':en,'estimator_id':ek,'seed':seed,**{m:tm.get(m,np.nan) for m in METRICS},'validation_score':d.get('best_validation_rmse',np.nan),'runtime_seconds':d.get('runtime_seconds',np.nan),'status':'completed_reused','best_params_json':json.dumps(d.get('best_params',{}),sort_keys=True),'source':str(p.relative_to(ROOT))})
        else: rows.append({'feature_set_id':fs,'feature_set':NAME[fs],'estimator':en,'estimator_id':ek,'seed':seed,'status':'BLOCKED','source':'Missing 10-trial cell; not mixed with 50-trial primary results.'})
    df=pd.DataFrame(rows); df.to_csv(OUT/'estimator_matrix/seed_level_results.csv',index=False)
    ss=[]
    for (e,fs),g in df[df.status=='completed_reused'].groupby(['estimator','feature_set_id']):
      for m in METRICS:
        mean,sd,lo,hi=ci_t(g[m]);ss.append({'estimator':e,'feature_set_id':fs,'feature_set':NAME[fs],'metric':m,'mean':mean,'sample_sd':sd,'ci95_low':lo,'ci95_high':hi,'n_completed_seeds':len(g)})
    pd.DataFrame(ss).to_csv(OUT/'estimator_matrix/summary_with_ci.csv',index=False)
    return df

def paired(rdf,edf):
    comps={'random':[('A+B+C','A'),('A+B','A'),('A+B+C','A+B'),('A+C','A'),('B+C','B'),('B+C','C')],
           'sequence_cluster':[('A+B','A'),('A+B','A+B+C'),('A+B+C','A'),('A+C','A')]}
    out=[]; tests=[]
    for sp,pairs in comps.items():
      for x,y in pairs:
       for m in METRICS:
        a=rdf[(rdf.split==sp)&(rdf.feature_set_id==x)][['seed',m]];b=rdf[(rdf.split==sp)&(rdf.feature_set_id==y)][['seed',m]]
        q=a.merge(b,on='seed',suffixes=('_x','_y'))
        if len(q)==5:
          ds=(q[m+'_x']-q[m+'_y']).to_numpy();mean,sd,lo,hi=ci_t(ds); t,p=stats.ttest_rel(q[m+'_x'],q[m+'_y'])
          out.append({'split':sp,'comparison':f'{NAME[x]} minus {NAME[y]}','metric':m,'mean_difference':mean,'ci95_low':lo,'ci95_high':hi,'ci_excludes_zero':lo>0 or hi<0,'seed_differences_json':json.dumps(ds.tolist())})
          tests.append({'split':sp,'comparison':f'{NAME[x]} minus {NAME[y]}','metric':m,'test':'two-sided paired t-test','statistic':t,'p_value':p,'n_pairs':5,'evidence':'directional' if (lo>0 or hi<0) else 'inconclusive'})
    pd.DataFrame(out).to_csv(OUT/'paired_statistics/paired_differences.csv',index=False);pd.DataFrame(tests).to_csv(OUT/'paired_statistics/paired_tests.csv',index=False)

def scaffold():
    p=ROOT/'results/benchmark/scaffold_ranking/family_level_results.csv'; d=pd.read_csv(p)
    d.to_csv(OUT/'scaffold_ranking/family_level_results.csv',index=False)
    rng=np.random.default_rng(20260717); res=[]
    for mid,g in d.groupby('model_id'):
      vals=g.pairwise_ranking_accuracy.dropna().to_numpy(); boots=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(5000)])
      res.append({'model_id':mid,'n_families':len(vals),'pairwise_accuracy':vals.mean(),'ci95_low':np.quantile(boots,.025),'ci95_high':np.quantile(boots,.975),'bootstrap_unit':'peptide family','bootstrap_replicates':5000})
    pd.DataFrame(res).to_csv(OUT/'scaffold_ranking/summary_with_ci.csv',index=False)
    a=d[d.model_id=='chem_A_xgb'][['family_id','pairwise_ranking_accuracy']];b=d[d.model_id=='chem_site_AB_xgb'][['family_id','pairwise_ranking_accuracy']];q=b.merge(a,on='family_id',suffixes=('_AB','_A'));ds=q.iloc[:,1]-q.iloc[:,2]
    boots=np.array([rng.choice(ds.to_numpy(),len(ds),replace=True).mean() for _ in range(5000)]); pos=(ds>0).sum();neg=(ds<0).sum()
    pd.DataFrame([{'comparison':'Chemistry + Site minus Chemistry','mean_difference':ds.mean(),'ci95_low':np.quantile(boots,.025),'ci95_high':np.quantile(boots,.975),'proportion_improved':(ds>0).mean(),'proportion_tied':(ds==0).mean(),'proportion_worsened':(ds<0).mean(),'sign_test_p_value':stats.binomtest(pos,pos+neg,.5).pvalue if pos+neg else np.nan,'n_families':len(ds)}]).to_csv(OUT/'scaffold_ranking/paired_comparison_AB_vs_A.csv',index=False)

def feature_inventory():
    from src.benchmark.featurizers import AnchorAwareWrapper, RDKitFullFeaturizer
    a=AnchorAwareWrapper(descriptor_set='basic',ablation_mode='chemistry_only'); an=a.get_feature_names(); r=RDKitFullFeaturizer();rn=r.get_feature_names()
    exact=set(x.replace('rdkit_','') for x in rn)&set(an)
    rows=[]
    for group,names,obj,level,agg,missing,scale in [('Group A',an,'full peptide SMILES plus parsed edit records','full-molecule plus edit-count','none','non-finite to zero in benchmark','none for XGBoost'),('RDKit',rn,'full peptide SMILES','full-molecule','none','training-median descriptor imputation','none for XGBoost')]:
      for n in names: rows.append({'feature_system':group,'descriptor_name':n,'dimension':len(names),'input_object':obj,'calculation_level':level,'aggregation':agg,'edit_type_indicators':group=='Group A' and n in ['num_edits','num_edit_families'],'missing_value_handling':missing,'feature_scaling':scale,'constant_feature_removal':'none','exact_overlap_after_prefix_removal':(n.replace('rdkit_','') in exact)})
    df=pd.DataFrame(rows);df.to_csv(OUT/'feature_inventory/group_a_vs_rdkit_feature_inventory.csv',index=False)
    (OUT/'feature_inventory/group_a_vs_rdkit_feature_inventory.md').write_text('# Group A versus RDKit feature inventory\n\nGroup A uses 8 selected whole-peptide RDKit physicochemical descriptors plus two edit-count features. The RDKit baseline computes the dynamically discovered full RDKit 2D descriptor set from the same full-peptide SMILES and imputes descriptor failures with training-set medians. Neither implementation is edit-level chemical-descriptor aggregation. This conflicts with registry text claiming per-edit/order-invariant aggregation.\n\nExact name overlaps are recorded in the CSV.\n')

def main():
    mkdirs();manifest();rdf,reused=audit_and_primary();edf=estimator_matrix();paired(rdf,edf);scaffold();feature_inventory()
    pd.DataFrame([{'status':'BLOCKED','reason':'Committed cutoff summaries lack compound-level predictions; legacy generator is non-runnable.'}]).to_csv(OUT/'time_forward/cutoff_statistics.csv',index=False)
    pd.DataFrame([{'status':'BLOCKED','reason':'Exact eight-cutoff ECFP nearest-neighbor reconstruction is required.'}]).to_csv(OUT/'time_forward/similarity_diagnostics.csv',index=False)
    pd.DataFrame([{'status':'BLOCKED','reason':'Grouped confidence intervals require compound-level predictions.'}]).to_csv(OUT/'time_forward/model_performance_with_ci.csv',index=False)
    pd.DataFrame([{'status':'BLOCKED','reason':'Five-seed SHAP values/models are not committed; retraining is required.'}]).to_csv(OUT/'shap/seed_level_group_attribution.csv',index=False)
    pd.DataFrame([{'group':'A','committed_single_analysis_proportion':0.3287},{'group':'B','committed_single_analysis_proportion':0.4215},{'group':'C','committed_single_analysis_proportion':0.2498}]).to_csv(OUT/'shap/group_summary_with_ci.csv',index=False)
    pd.DataFrame([{'subgroup':'B1','committed_single_analysis_proportion':0.1751},{'subgroup':'B2','committed_single_analysis_proportion':0.1417},{'subgroup':'B3','committed_single_analysis_proportion':0.1047}]).to_csv(OUT/'shap/subgroup_summary_with_ci.csv',index=False)
    pd.DataFrame([{'status':'BLOCKED','reason':'Per-seed top-feature rankings require SHAP recomputation.'}]).to_csv(OUT/'shap/top_features.csv',index=False)
    blocked_primary=70-len(rdf);blocked_est=(edf.status=='BLOCKED').sum()
    report=f"""# Minor Revision Experimental Report\n\n## A. Executive status\n\nPARTIALLY COMPLETE\n\nThe audit verified the corrected 7,224-sample dataset and canonical seeds 0–4. {len(rdf)} of 70 primary-ablation seed cells and {(edf.status=='completed_reused').sum()} of 175 estimator-matrix cells are protocol-matched and reusable. Missing full-protocol training was not replaced by lower-budget results.\n\n## B. Reused versus newly run experiments\n\nReused {reused} primary cells: committed 50-trial random/cluster A, A+B, and A+B+C cells, plus random A+C supplementary cells with five seeds and 50 trials. Reused estimator cells are listed row-by-row in `estimator_matrix/seed_level_results.csv`; A+B and A+C were absent from the committed 10-trial matrix. No performance run was newly completed.\n\n## C. Complete primary-ablation results\n\nThe requested seven-way table is incomplete. See `primary_ablation/summary_with_ci.csv`. {blocked_primary} cells are BLOCKED.\n\n## D. Key paired comparisons\n\nAvailable five-seed paired results and 95% t intervals are in `paired_statistics/paired_differences.csv`. Comparisons lacking both protocol-matched sides are omitted and remain inconclusive.\n\n## E. Estimator conclusions\n\nRandom Forest versus XGBoost for A+B+C is available with paired seed analysis possible from the machine-readable table; rounded means alone are not treated as reliable superiority. The incomplete A+B/A+C matrix prevents a complete estimator-dependence conclusion.\n\n## F. Scaffold-ranking conclusion\n\nSee `scaffold_ranking/paired_comparison_AB_vs_A.csv`; inference uses peptide-family bootstrap and a sign test.\n\n## G. Time-forward diagnosis\n\nBLOCKED for revision-grade inference: committed cutoff summaries contain eight cutoffs but no compound-level predictions, and the existing generator is non-runnable (wrong raw path and undefined variables). CIs and jointly grouped model differences cannot be reconstructed without rerunning.\n\n## H. SHAP conclusion\n\nBLOCKED for five-seed uncertainty: committed SHAP tables are single aggregate analyses. Existing full-model attribution reports A=0.3287, C=0.2498, B1=0.1751, B2=0.1417, B3=0.1047; combined B=0.4215, so B—not A—is the largest conceptual group, while A is the largest individual subblock. SHAP is model attribution, not causal evidence.\n\n## I. Group A versus RDKit\n\nThe code inventory shows both operate on full-peptide SMILES. Group A is a selected 8-descriptor whole-molecule panel plus two edit-count features; RDKit is the broad dynamically discovered 2D descriptor set with training-median imputation. Registry claims of per-edit aggregation do not match implementation.\n\n## J. Manuscript-safe statements\n\n- Claim: Combined Group B is larger than Group A in the committed full-model SHAP summary. Status: SUPPORTED WITH CAUTION. Evidence: B1+B2+B3=0.4215 versus A=0.3287; single aggregate analysis, no five-seed CI.\n- Claim: Chemistry is the largest overall conceptual SHAP group. Status: NOT SUPPORTED.\n- Claim: The seven-way ablation is complete. Status: NOT SUPPORTED.\n- Claim: The estimator-by-descriptor matrix is complete. Status: NOT SUPPORTED.\n\n## K. Remaining blockers\n\n- Primary ablation: {blocked_primary} missing seed cells; measured first full-search trial took about 31 seconds, projecting roughly 17 CPU-hours for the missing 2,000 trials.\n- Estimator matrix: {blocked_est} missing A+B/A+C cells.\n- Time-forward: no compound predictions and broken legacy generator.\n- SHAP: no five-seed SHAP artifacts/models; retraining required.\n- Existing prediction files lack explicit sequence-cluster identifiers; cluster assignments are not shipped separately.\n"""
    (OUT/'MINOR_REVISION_EXPERIMENT_REPORT.md').write_text(report)
    print(json.dumps({'status':'PARTIALLY COMPLETE','reused_runs':int(reused+(edf.status=='completed_reused').sum()),'newly_completed_runs':0,'blocked_runs':int(blocked_primary+blocked_est),'output':str(OUT)},indent=2))
if __name__=='__main__':main()
