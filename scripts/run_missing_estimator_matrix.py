#!/usr/bin/env python3
"""Complete the 50 absent 10-trial A+B/A+C estimator-matrix cells."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from run_estimator_comparison import run_one

FEATURES=['Chem+Site','Chem+Context']
ESTIMATORS=['ridge','elasticnet','random_forest','svr','xgboost']

def valid(p):
    if not (p.exists() and p.with_name('predictions.csv').exists() and p.with_name('optuna_trials.csv').exists()): return False
    try:
        d=json.loads(p.read_text())
        return d.get('status')=='completed' and d.get('n_trials')==10 and sum(1 for _ in p.with_name('optuna_trials.csv').open())-1==10
    except Exception:return False

def main():
    tasks=[]
    for f in FEATURES:
      safe=f.replace(' ','_').replace('+','_')
      for e in ESTIMATORS:
       for s in range(5):
        p=ROOT/f'results/benchmark/estimator_comparison/{safe}_{e}/seed_{s}/metrics.json'
        if not valid(p): tasks.append((e,f,s))
    print(f'Missing estimator cells to execute: {len(tasks)}',flush=True)
    ok=bad=0
    with ProcessPoolExecutor(max_workers=8) as ex:
      futs={ex.submit(run_one,*t):t for t in tasks}
      for fut in as_completed(futs):
       t=futs[fut]
       try:
        r=fut.result(); ok+=1; print(t,'completed',r['test_metrics']['r2'],flush=True)
       except Exception as exc:
        bad+=1; print(t,'failed',repr(exc),flush=True)
    print({'completed':ok,'failed':bad,'expected':len(tasks)},flush=True)
if __name__=='__main__':main()
