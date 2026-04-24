# PyTLF

Using Claude Code to develop standardized TLFs using Python.

End-to-end demo pipeline for a simulated clinical trial (**Study ABC-001**, a
2-arm placebo-controlled study with 50 subjects across 3 sites). The pipeline
flows from fake patient CRF data through CDISC SDTM domains and ADaM analysis
datasets, and produces three publication-style summary tables.

## Pipeline

```
raw CRF  --->  SDTM  --->  ADaM  --->  TLFs
 CSVs          DM, AE,     ADSL,       Demographics,
               VS, TV,     ADAE,       AE by SOC/PT,
               SV          ADVS        VS summary
```

## Repository layout

```
PyTLF/
├── run_pipeline.py          # End-to-end orchestrator
├── requirements.txt
├── src/
│   ├── generate_raw.py      # Fake CRF data (enrollment, visits, vitals, AEs)
│   ├── generate_sdtm.py     # DM, AE, VS, TV, SV
│   ├── generate_adam.py     # ADSL, ADAE, ADVS
│   └── generate_tlfs.py     # Demographic, AE SOC/PT, VS summary tables
├── data/
│   ├── raw/                 # Raw CRF CSV exports
│   ├── sdtm/                # SDTM domains
│   └── adam/                # ADaM datasets
└── output/                  # TLF tables (.csv + .txt)
```

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Tables are written to `output/` as both CSV (for downstream tooling) and
fixed-width `.txt` (for human review).

## Tables produced

1. **Table 1 — Demographic and Baseline Characteristics** (`output/t_demographics.*`)
   Built from ADSL. Columns per treatment + Total; rows for N, age
   summary + age group, sex, race, ethnicity.
2. **Table 2 — TEAEs by SOC and Preferred Term** (`output/t_ae_soc_pt.*`)
   Built from ADAE (TRTEMFL = "Y"). Subject counts and percentages per
   treatment, with System Organ Class rollups and indented Preferred Terms.
3. **Table 3 — Summary Statistics of Vital Signs** (`output/t_vs_summary.*`)
   Built from ADVS. N, Mean, SD, Median, Min, Max per PARAMCD and visit.

## Reproducibility

Raw data generation uses a fixed seed (`np.random.default_rng(20260424)`), so
the outputs are deterministic from run to run.
