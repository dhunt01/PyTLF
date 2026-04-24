"""Generate fake raw CRF data for a simulated clinical trial (Study ABC-001).

Produces CSV files in ``data/raw/`` representing what a typical patient CRF
capture system might export: enrollment/demographics, visit dates, vitals,
and adverse events.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

STUDY_ID = "ABC-001"
N_SUBJECTS = 50
SITES = ["101", "102", "103"]
COUNTRIES = {"101": "USA", "102": "USA", "103": "CAN"}
TREATMENTS = ["Placebo", "ABC Drug 50mg"]

VISITS = [
    ("SCREENING", -14),
    ("BASELINE", 1),
    ("WEEK 2", 15),
    ("WEEK 4", 29),
    ("WEEK 8", 57),
    ("END OF TREATMENT", 85),
]

AE_CATALOG = [
    ("Headache", "Nervous system disorders", "Headache"),
    ("Nausea", "Gastrointestinal disorders", "Nausea"),
    ("Fatigue", "General disorders and administration site conditions", "Fatigue"),
    ("Dizziness", "Nervous system disorders", "Dizziness"),
    ("Diarrhea", "Gastrointestinal disorders", "Diarrhoea"),
    ("Upper respiratory infection", "Infections and infestations",
     "Upper respiratory tract infection"),
    ("Back pain", "Musculoskeletal and connective tissue disorders", "Back pain"),
    ("Insomnia", "Psychiatric disorders", "Insomnia"),
    ("Rash", "Skin and subcutaneous tissue disorders", "Rash"),
    ("Hypertension", "Vascular disorders", "Hypertension"),
]


def _rng() -> np.random.Generator:
    return np.random.default_rng(seed=20260424)


def _subject_ids() -> list[str]:
    return [f"{STUDY_ID}-{i:04d}" for i in range(1, N_SUBJECTS + 1)]


def generate_enrollment() -> pd.DataFrame:
    rng = _rng()
    subjects = _subject_ids()
    rows = []
    base_enroll = date(2025, 6, 1)
    for idx, subj in enumerate(subjects):
        site = rng.choice(SITES)
        sex = rng.choice(["M", "F"], p=[0.48, 0.52])
        race = rng.choice(
            ["WHITE", "BLACK OR AFRICAN AMERICAN", "ASIAN",
             "AMERICAN INDIAN OR ALASKA NATIVE", "OTHER"],
            p=[0.70, 0.15, 0.10, 0.03, 0.02],
        )
        ethnic = rng.choice(["HISPANIC OR LATINO", "NOT HISPANIC OR LATINO"], p=[0.18, 0.82])
        age = int(rng.integers(22, 76))
        dob = date(2025 - age, int(rng.integers(1, 13)), int(rng.integers(1, 28)))
        treatment = TREATMENTS[idx % 2]
        enroll_dt = base_enroll + timedelta(days=int(rng.integers(0, 90)))
        rand_dt = enroll_dt + timedelta(days=int(rng.integers(0, 7)))
        rows.append({
            "STUDYID": STUDY_ID,
            "SUBJID": subj.split("-")[-1],
            "USUBJID": subj,
            "SITEID": str(site),
            "COUNTRY": COUNTRIES[str(site)],
            "SEX": sex,
            "RACE": race,
            "ETHNIC": ethnic,
            "AGE": age,
            "DOB": dob.isoformat(),
            "TREATMENT": treatment,
            "ENROLLDT": enroll_dt.isoformat(),
            "RANDDT": rand_dt.isoformat(),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "enrollment.csv", index=False)
    return df


def generate_visits(enroll: pd.DataFrame) -> pd.DataFrame:
    rng = _rng()
    rows = []
    for _, subj in enroll.iterrows():
        rand_dt = date.fromisoformat(subj["RANDDT"])
        completed_all = rng.random() > 0.12
        last_visit_idx = len(VISITS) if completed_all else int(rng.integers(2, len(VISITS)))
        for i, (visit, offset) in enumerate(VISITS[:last_visit_idx]):
            actual = rand_dt + timedelta(days=offset + int(rng.integers(-2, 3)))
            rows.append({
                "USUBJID": subj["USUBJID"],
                "VISIT": visit,
                "VISIT_DATE": actual.isoformat(),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "visits.csv", index=False)
    return df


def generate_vitals(visits: pd.DataFrame, enroll: pd.DataFrame) -> pd.DataFrame:
    rng = _rng()
    treat_map = dict(zip(enroll["USUBJID"], enroll["TREATMENT"]))
    rows = []
    for _, v in visits.iterrows():
        trt = treat_map[v["USUBJID"]]
        drug_effect = -3.0 if trt == "ABC Drug 50mg" and v["VISIT"] not in ("SCREENING", "BASELINE") else 0.0
        sbp = int(round(rng.normal(128 + drug_effect, 9)))
        dbp = int(round(rng.normal(80 + drug_effect / 2, 6)))
        pulse = int(round(rng.normal(72, 7)))
        temp = round(rng.normal(36.8, 0.3), 1)
        weight = round(rng.normal(78.0, 12.0), 1)
        height = round(rng.normal(170.0, 9.0), 1)
        tests = [
            ("SYSBP", "Systolic Blood Pressure", sbp, "mmHg"),
            ("DIABP", "Diastolic Blood Pressure", dbp, "mmHg"),
            ("PULSE", "Pulse Rate", pulse, "beats/min"),
            ("TEMP", "Temperature", temp, "C"),
            ("WEIGHT", "Weight", weight, "kg"),
            ("HEIGHT", "Height", height, "cm"),
        ]
        for code, name, val, unit in tests:
            if code == "HEIGHT" and v["VISIT"] != "SCREENING":
                continue
            rows.append({
                "USUBJID": v["USUBJID"],
                "VISIT": v["VISIT"],
                "VISIT_DATE": v["VISIT_DATE"],
                "TEST_CODE": code,
                "TEST_NAME": name,
                "RESULT": val,
                "UNIT": unit,
            })
    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "vitals.csv", index=False)
    return df


def generate_aes(enroll: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    rng = _rng()
    treat_map = dict(zip(enroll["USUBJID"], enroll["TREATMENT"]))
    end_dt_map = (
        visits.groupby("USUBJID")["VISIT_DATE"]
        .max()
        .apply(date.fromisoformat)
        .to_dict()
    )
    start_dt_map = dict(zip(enroll["USUBJID"], enroll["RANDDT"].map(date.fromisoformat)))
    rows = []
    for subj in enroll["USUBJID"]:
        trt = treat_map[subj]
        rate = 2.5 if trt == "ABC Drug 50mg" else 1.4
        n_ae = int(rng.poisson(rate))
        for _ in range(n_ae):
            ae_term, _, _ = AE_CATALOG[int(rng.integers(0, len(AE_CATALOG)))]
            start = start_dt_map[subj] + timedelta(days=int(rng.integers(1, 80)))
            if start > end_dt_map[subj]:
                start = end_dt_map[subj]
            dur = int(rng.integers(1, 10))
            end = min(start + timedelta(days=dur), end_dt_map[subj])
            sev = rng.choice(["MILD", "MODERATE", "SEVERE"], p=[0.65, 0.30, 0.05])
            rel = rng.choice(["NOT RELATED", "RELATED"], p=[0.55, 0.45])
            serious = rng.choice(["N", "Y"], p=[0.95, 0.05])
            outcome = rng.choice(
                ["RECOVERED/RESOLVED", "NOT RECOVERED/NOT RESOLVED", "RECOVERING/RESOLVING"],
                p=[0.80, 0.10, 0.10],
            )
            rows.append({
                "USUBJID": subj,
                "AE_TERM": ae_term,
                "AE_START_DATE": start.isoformat(),
                "AE_END_DATE": end.isoformat(),
                "AE_SEVERITY": sev,
                "AE_RELATIONSHIP": rel,
                "AE_SERIOUS": serious,
                "AE_OUTCOME": outcome,
            })
    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "adverse_events.csv", index=False)
    return df


def main() -> None:
    enroll = generate_enrollment()
    visits = generate_visits(enroll)
    generate_vitals(visits, enroll)
    generate_aes(enroll, visits)
    print(f"Raw CRF data written to {RAW_DIR}")


if __name__ == "__main__":
    main()
