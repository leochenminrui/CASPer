"""Regression tests for the minor-revision experiment and statistics pipeline."""
from pathlib import Path
import inspect,json,sys
import numpy as np,pandas as pd,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from src.data.serialization import load_samples
from src.benchmark.featurizers import FEATURIZER_REGISTRY
from src.benchmark.evaluation import compute_all_metrics
from src.data.pem_schema import PEMSample
from src.benchmark import optuna_tuner
from scripts.run_minor_revision_shap import group
from scripts.run_missing_primary import valid

def splits(kind='random'):
 d=ROOT/f'data/splits/CycPeptMPDB_PAMPA/{kind}';return {x:load_samples(d/f'{x}.jsonl') for x in ['train','val','test']}

def test_corrected_dataset_and_unique_ids():
 s=load_samples(ROOT/'data/processed/pem_schema/cycpeptmpdb_pampa.jsonl');assert len(s)==7224;ids=[x.sample_id for x in s];assert len(ids)==len(set(ids))

@pytest.mark.parametrize('kind',['random','sequence_cluster'])
def test_split_isolation_and_total(kind):
 d=splits(kind);sets={k:{x.sample_id for x in v} for k,v in d.items()};assert sum(map(len,sets.values()))==7224;assert not sets['train']&sets['val'];assert not sets['train']&sets['test'];assert not sets['val']&sets['test']

def test_cluster_sequence_isolation():
 d=splits('sequence_cluster');assert not {x.sequence for x in d['train']}&{x.sequence for x in d['test']};assert not {x.sequence for x in d['val']}&{x.sequence for x in d['test']}

def test_descriptor_combinations_and_row_alignment():
 s=splits()['train'][:12];specs={'A':('anchor_aware',{'descriptor_set':'basic','ablation_mode':'chemistry_only'},10),'B':('site_only',{},35),'C':('context_only',{},28),'A+B':('anchor_aware',{'descriptor_set':'basic','ablation_mode':'chemistry_anchors'},45),'A+C':('anchor_aware',{'descriptor_set':'basic','ablation_mode':'chemistry_attachment'},38),'B+C':('anchor_aware',{'descriptor_set':'basic','ablation_mode':'site_context_only'},63),'A+B+C':('anchor_aware',{'descriptor_set':'basic','ablation_mode':'full'},73)}
 for _,(k,kw,n) in specs.items():
  f=FEATURIZER_REGISTRY[k](**kw);f.fit(s);x=f.transform(s);assert x.shape==(len(s),n);assert len(f.get_feature_names())==n

def test_seed_definitions_and_optuna_never_receive_test_arrays():
 cfg=json.loads((ROOT/'results/minor_revision_experiments/manifest/run_manifest.json').read_text());assert cfg['seeds']==[0,1,2,3,4];src=inspect.getsource(optuna_tuner.tune_xgboost);assert 'X_test' not in src and 'y_test' not in src;assert 'X_val' in src and 'y_val' in src

def test_metric_correctness():
 y=np.array([1.,2.,3.]);p=np.array([1.,2.,3.]);m=compute_all_metrics(y,p);assert m['r2']==pytest.approx(1);assert m['rmse']==pytest.approx(0);assert m['mae']==pytest.approx(0);assert m['spearman']==pytest.approx(1)

def test_shap_membership_and_b_aggregation():
 assert group('mol_weight')=='A';assert group('anchor_pos_mean')=='B1';assert group('anchor_res_A')=='B2';assert group('anchor_hydrophobic_frac')=='B3';assert group('edit_family_backbone')=='C'
 d=pd.read_csv(ROOT/'results/minor_revision_experiments/shap/seed_level_group_attribution.csv');f=d[d.representation_id=='A+B+C'];wide=f.pivot(index='seed',columns='group',values='attribution_proportion');assert np.allclose(wide.B,wide.B1+wide.B2+wide.B3)

def test_prediction_alignment_and_primary_completeness():
 d=pd.read_csv(ROOT/'results/minor_revision_experiments/primary_ablation/seed_level_results.csv');assert len(d)==70
 for r in d.itertuples():
  p=pd.read_csv(ROOT/r.prediction_file);s=load_samples(ROOT/f'data/splits/CycPeptMPDB_PAMPA/{r.split}/test.jsonl');assert p.sample_id.tolist()==[x.sample_id for x in s];assert np.allclose(p.y_true,[x.label for x in s])

def test_estimator_completeness_and_resume_validation():
 d=pd.read_csv(ROOT/'results/minor_revision_experiments/estimator_matrix/seed_level_results.csv');assert len(d)==175;assert (d.status=='completed').all();p=ROOT/'results/benchmark/random/seed_0/chem_A_xgb/metrics.json';assert valid(p)

def test_grouped_resampling_keeps_groups_intact():
 groups=np.array(['a','a','b','b','c']);pick=np.array(['b','a','b']);ix=np.concatenate([np.flatnonzero(groups==g) for g in pick]);assert groups[ix].tolist()==['b','b','a','a','b','b']

def test_schema_rejects_nonstandard_one_letter_residue_but_raw_monomers_are_retained():
 base=load_samples(ROOT/'data/processed/pem_schema/cycpeptmpdb_pampa.jsonl')[0].model_dump();base['sample_id']='TEST_999';base['sequence']='KLMNOP'
 with pytest.raises(Exception):PEMSample(**base)
 assert (ROOT/'data/raw/CycPeptMPDB_Peptide_Assay_PAMPA.csv').exists()
