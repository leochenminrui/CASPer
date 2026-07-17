# Group A versus RDKit feature inventory

Group A uses 8 selected whole-peptide RDKit physicochemical descriptors plus two edit-count features. The RDKit baseline computes the dynamically discovered full RDKit 2D descriptor set from the same full-peptide SMILES and imputes descriptor failures with training-set medians. Neither implementation is edit-level chemical-descriptor aggregation. This conflicts with registry text claiming per-edit/order-invariant aggregation.

Exact name overlaps are recorded in the CSV.
