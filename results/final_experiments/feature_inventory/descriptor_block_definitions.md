# Supplementary descriptor-block definitions

The descriptor blocks used in model ablations, SHAP aggregation, figure
colors, legends, and captions follow the canonical feature order returned by
`AnchorAwareDescriptorFeaturizer.get_feature_names()`.

| Block | Definition | Feature count |
|---|---|---:|
| A | Whole-peptide chemistry | 10 |
| B1 | Anchor-position statistics | 6 |
| B2 | Anchor-residue identity | 20 |
| B3 | Anchor-residue properties and local terminal environment | 9 |
| C | Scaffold/attachment/multi-edit context | 28 |

## Why terminal anchor counts are in B3

`anchor_n_terminal_count` and `anchor_c_terminal_count` are intentionally B3
features. They describe whether anchor sites occur in the local N- or
C-terminal environment and are constructed alongside anchor-site property
fractions (hydrophobic, charged, polar, aromatic, small, and proline) and
`anchor_terminal_frac`.

B1 is restricted to six global anchor-location summaries:
`anchor_count_total`, `anchor_count_unique`, `anchor_density`,
`anchor_pos_mean`, `anchor_pos_std`, and `anchor_pos_range`. Therefore,
`C-terminal anchor count` is colored as B3 in Figure 2; this is not a plotting
reclassification.

## Group C terminology

All figures and captions use **Group C: scaffold/attachment/multi-edit
context**. This block includes edit-family distributions, edit diversity,
specific edit indicators, cyclization descriptors, and multi-edit aggregation
features. In particular, `edit_n_methylation`, `edit_d_amino_acid`,
`anchor_pair_count`, and `cyclization_ring_size` are Group C features.
