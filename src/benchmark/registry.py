"""
Unified model registry for CASPer benchmark.

Defines every model with:
- model_id: short unique identifier (used in filenames)
- model_name: human-readable name for tables
- role: internal_baseline / primary_paper / ablation_control /
        generic_benchmark
- featurizer: which featurizer class to use
- featurizer_kwargs: keyword args for the featurizer
- estimator_type: xgboost / random_forest
- purpose: one-line explanation for benchmark summaries
- requires_external: list of dependencies that may be missing
- hpo: whether Optuna tuning is supported
- status: implemented / requires_external / skipped
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ModelSpec:
    model_id: str
    model_name: str
    role: str
    purpose: str
    featurizer: str  # key into FEATURIZER_REGISTRY
    featurizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    estimator_type: str = "xgboost"
    estimator_kwargs: Dict[str, Any] = field(default_factory=dict)
    requires_external: List[str] = field(default_factory=list)
    hpo: bool = True
    hpo_reason: str = ""
    status: str = "implemented"


# ─── Model Registry ──────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, ModelSpec] = {

    # ═══ A. Internal Baselines ══════════════════════════════════════════════

    "seq_aa_xgb": ModelSpec(
        model_id="seq_aa_xgb",
        model_name="AA Composition + XGBoost",
        role="internal_baseline",
        purpose="Minimal sequence-only baseline: amino-acid composition only, "
                "no chemical modification information",
        featurizer="aa_composition",
        featurizer_kwargs={"use_aa_composition": True, "use_property_composition": True,
                          "use_basic_features": True, "use_dipeptide": False},
        estimator_type="xgboost",
    ),

    "position_only_xgb": ModelSpec(
        model_id="position_only_xgb",
        model_name="Position-Only + XGBoost",
        role="internal_baseline",
        purpose="Control: anchor count + site-location statistics only. "
                "Tests whether site-aware gain is a positional shortcut "
                "rather than chemistry-anchor interaction",
        featurizer="position_only",
        featurizer_kwargs={},
        estimator_type="xgboost",
    ),

    # ═══ B. Primary Descriptor Models (this paper) ══════════════════════════

    "chem_A_xgb": ModelSpec(
        model_id="chem_A_xgb",
        model_name="Chem-Only (Group A) + XGBoost",
        role="primary_paper",
        purpose="Chemistry-only model: global molecular descriptors (Group A) "
                "without any anchor/site information. Primary baseline for "
                "anchor-aware comparison",
        featurizer="anchor_aware",
        featurizer_kwargs={"descriptor_set": "basic", "ablation_mode": "chemistry_only"},
        estimator_type="xgboost",
    ),

    "site_B_xgb": ModelSpec(
        model_id="site_B_xgb",
        model_name="Site-Only (Group B) + XGBoost",
        role="ablation_control",
        purpose="Site descriptors only (Group B: position + residue + context) "
                "without chemistry (Group A), enabling a direct test of "
                "test whether B alone carries signal",
        featurizer="site_only",
        featurizer_kwargs={},
        estimator_type="xgboost",
    ),

    "context_C_xgb": ModelSpec(
        model_id="context_C_xgb",
        model_name="Context-Only (Group C) + XGBoost",
        role="ablation_control",
        purpose="Context/attachment descriptors only (Group C) without "
                "chemistry or anchor-site features",
        featurizer="context_only",
        featurizer_kwargs={},
        estimator_type="xgboost",
    ),

    "site_context_BC_xgb": ModelSpec(
        model_id="site_context_BC_xgb",
        model_name="Site+Context (Groups B+C) + XGBoost",
        role="ablation_control",
        purpose="Site + context without chemistry: tests contribution of "
                "positional and contextual features alone",
        featurizer="anchor_aware",
        featurizer_kwargs={"descriptor_set": "basic", "ablation_mode": "site_context_only"},
        estimator_type="xgboost",
    ),

    "chem_site_AB_xgb": ModelSpec(
        model_id="chem_site_AB_xgb",
        model_name="Chem+Site (Groups A+B) + XGBoost",
        role="primary_paper",
        purpose="Chemistry + anchor-site features (Groups A+B). Stricter "
                "chemistry-plus-site model — primary anchor-aware comparison",
        featurizer="anchor_aware",
        featurizer_kwargs={"descriptor_set": "basic", "ablation_mode": "chemistry_anchors"},
        estimator_type="xgboost",
    ),

    "chem_context_AC_xgb": ModelSpec(
        model_id="chem_context_AC_xgb",
        model_name="Chem+Context (Groups A+C) + XGBoost",
        role="ablation_control",
        purpose="Chemistry + attachment/context features (Groups A+C). "
                "Tests context contribution without site-specific anchors",
        featurizer="anchor_aware",
        featurizer_kwargs={"descriptor_set": "basic", "ablation_mode": "chemistry_attachment"},
        estimator_type="xgboost",
    ),

    "full_ABC_xgb": ModelSpec(
        model_id="full_ABC_xgb",
        model_name="Full ABC (Groups A+B+C) + XGBoost",
        role="primary_paper",
        purpose="Full site-conditioned descriptor model: chemistry (A) + "
                "anchor-site (B) + context/attachment (C). Main proposed model",
        featurizer="anchor_aware",
        featurizer_kwargs={"descriptor_set": "basic", "ablation_mode": "full"},
        estimator_type="xgboost",
    ),

    "chem_B1_xgb": ModelSpec(
        model_id="chem_B1_xgb",
        model_name="Chem + B1 (Position Stats) + XGBoost",
        role="ablation_control",
        purpose="Ablation: Chemistry + anchor position statistics (B1) only. "
                "Isolates position-statistics contribution",
        featurizer="anchor_aware",
        featurizer_kwargs={"descriptor_set": "basic", "ablation_mode": "chemistry_position"},
        estimator_type="xgboost",
    ),

    "chem_B2_xgb": ModelSpec(
        model_id="chem_B2_xgb",
        model_name="Chem + B2 (Residue Comp) + XGBoost",
        role="ablation_control",
        purpose="Ablation: Chemistry + anchor residue composition (B2) only. "
                "Isolates residue-identity contribution",
        featurizer="anchor_aware",
        featurizer_kwargs={"descriptor_set": "basic", "ablation_mode": "chemistry_residue"},
        estimator_type="xgboost",
    ),

    "chem_B3_xgb": ModelSpec(
        model_id="chem_B3_xgb",
        model_name="Chem + B3 (Residue Properties) + XGBoost",
        role="ablation_control",
        purpose="Ablation: Chemistry + anchor residue properties (B3) only. "
                "Isolates residue-property (hydrophobicity, charge, etc.) contribution",
        featurizer="anchor_aware",
        featurizer_kwargs={"descriptor_set": "basic", "ablation_mode": "chemistry_context"},
        estimator_type="xgboost",
    ),

    # ═══ C. Generic Chemistry Benchmarks ════════════════════════════════════

    "ecfp_xgb": ModelSpec(
        model_id="ecfp_xgb",
        model_name="ECFP4 (Morgan) + XGBoost",
        role="generic_benchmark",
        purpose="Morgan fingerprint (radius=2, nBits=2048) + XGBoost. "
                "Multi-edit peptides use mean aggregation of per-edit fingerprints. "
                "Standard cheminformatics baseline",
        featurizer="ecfp",
        featurizer_kwargs={"radius": 2, "nBits": 2048, "aggregation": "mean"},
        estimator_type="xgboost",
    ),

    "rdkit_full_xgb": ModelSpec(
        model_id="rdkit_full_xgb",
        model_name="RDKit Full 2D + XGBoost",
        role="generic_benchmark",
        purpose="Full RDKit 2D molecular descriptors (all available) + XGBoost. "
                "Multi-edit peptides use order-invariant aggregation. "
                "NaN/inf values imputed with median",
        featurizer="rdkit_full",
        featurizer_kwargs={"aggregation": "mean"},
        estimator_type="xgboost",
    ),

    "ecfp_rf": ModelSpec(
        model_id="ecfp_rf",
        model_name="ECFP4 (Morgan) + Random Forest",
        role="generic_benchmark",
        purpose="Morgan fingerprint + Random Forest. "
                "Non-boosting traditional ML baseline for comparison with XGBoost",
        featurizer="ecfp",
        featurizer_kwargs={"radius": 2, "nBits": 2048, "aggregation": "mean"},
        estimator_type="random_forest",
    ),

}


# ─── Helper Functions ───────────────────────────────────────────────────────

def get_model_config(model_id: str) -> Optional[ModelSpec]:
    """Get model spec by ID."""
    return MODEL_REGISTRY.get(model_id)


def list_models_by_role() -> Dict[str, List[str]]:
    """Group model_ids by role, ordered by the desired table order."""
    role_order = [
        "internal_baseline",
        "primary_paper",
        "ablation_control",
        "generic_benchmark",
    ]
    grouped = {}
    for role in role_order:
        grouped[role] = [
            mid for mid, spec in MODEL_REGISTRY.items() if spec.role == role
        ]
    return grouped


def list_implemented_models() -> List[str]:
    """Return model_ids that are implemented and runnable now."""
    return [mid for mid, spec in MODEL_REGISTRY.items()
            if spec.status == "implemented"]
