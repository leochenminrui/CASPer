# Scaffold-Focused Ranking Simulation — Methods

## Design
To address concerns about random split over-optimism and cluster split harshness, we implemented a retrospective scaffold-focused ranking simulation.

### Family Definition
- `family_id = Source + Monomer_Length + Molecule_Shape`
- Minimum family size: 8 compounds
- Only families with valid Year and PAMPA labels

### Ranking Protocol
1. Historical training: all compounds from same Source with Year < target_year
2. Support set: 30% of family compounds (min 3)
3. Test set: remaining 70% of family compounds
4. Models trained on historical + support, evaluated on test

### Leakage Prevention
- Same_Peptides_Permeability excluded from features
- Duplicate leakage detection via Structurally_Unique_ID cross-check
- Strict mode: exclude families with any leakage

### Results
- Families evaluated: 0
- Total model-family runs: 0
- Families skipped: 0
- Reasons: insufficient historical data, too few members, leakage detected

## Limitations
- Year is publication year, not experiment date — may not reflect true temporal order
- Historical data may include structurally related compounds not in the same family
- Small within-family sample sizes limit statistical power
