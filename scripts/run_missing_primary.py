#!/usr/bin/env python3
"""Run only absent 50-trial primary-ablation cells with bounded parallelism."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json, sys, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.benchmark.runner import run_single_model

MODELS = {
    "B": "site_B_xgb", "C": "context_C_xgb",
    "A+C": "chem_context_AC_xgb", "B+C": "site_context_BC_xgb",
}

def valid(path):
    if not path.exists() or not path.with_name("predictions.csv").exists(): return False
    try:
        d=json.loads(path.read_text())
        t=path.with_name("optuna_trials.csv")
        return d.get("status")=="completed" and t.exists() and sum(1 for _ in t.open())-1==50
    except Exception: return False

def one(task):
    split, model, seed, cfg = task
    return run_single_model(model, split, seed, cfg, ROOT/"results/final_experiments/raw_runs/primary_ablation", resume=True)

def main():
    cfg=yaml.safe_load((ROOT/"configs/benchmark/primary_ablation.yaml").read_text())
    tasks=[]
    for split in ("random","sequence_cluster"):
        for fs,model in MODELS.items():
            for seed in range(5):
                p=ROOT/f"results/final_experiments/raw_runs/primary_ablation/{split}/seed_{seed}/{model}/metrics.json"
                if not valid(p): tasks.append((split,model,seed,cfg))
    print(f"Missing primary cells to execute: {len(tasks)}", flush=True)
    done=failed=0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(one,t):t[:3] for t in tasks}
        for fut in as_completed(futs):
            key=futs[fut]
            try:
                r=fut.result(); ok=r.get("status")=="completed"; done+=ok; failed+=not ok
                print(key, r.get("status"), r.get("test_metrics",{}).get("r2"), flush=True)
            except Exception as e:
                failed+=1; print(key,"failed",repr(e),flush=True)
    print(json.dumps({"completed":done,"failed":failed,"expected":len(tasks)}),flush=True)
if __name__=="__main__": main()
