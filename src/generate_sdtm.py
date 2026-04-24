"""Transform raw CRF data into SDTM-compliant domains.

Produces DM, AE, VS, TV, SV as CSV files in ``data/sdtm/``. Variable names and
controlled terminology follow the CDISC SDTM IG conventions at a reasonable
fidelity for a simulated study.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SDTM_DIR = ROOT / "data" / "sdtm"
SDTM_DIR.mkdir(parents=True, exist_ok=True)

STUDY_ID = "ABC-001"

ARM_MAP = {
    "Placebo": ("PBO", "Placebo"),
    "ABC Drug 50mg": ("ABC50", "ABC Drug 50 mg"),
}

VISIT_META = [
    (1, "SCREENING", -14),
    (2, "BASELINE", 1),
    (3, "WEEK 2", 15),
    (4, "WEEK 4", 29),
    (5, "WEEK 8", 57),
    (6, "END OF TREATMENT", 85),
]

AE_DICT = {
    "Headache": ("HEADACHE", "Nervous system disorders"),
    "Nausea": ("NAUSEA", "Gastrointestinal disorders"),
    "Fatigue": ("FATIGUE", "General disorders and administration site conditions"),
    "Dizziness": ("DIZZINESS", "Nervous system disorders"),
    "Diarrhea": ("DIARRHOEA", "Gastrointestinal disorders"),
    "Upper respiratory infection": (
        "UPPER RESPIRATORY TRACT INFECTION",
        "Infections and infestations",
    ),
    "Back pain": ("BACK PAIN", "Musculoskeletal and connective tissue disorders"),
    "Insomnia": ("INSOMNIA", "Psychiatric disorders"),
    "Rash": ("RASH", "Skin and subcutaneous tissue disorders"),
    "Hypertension": ("HYPERTENSION", "Vascular disorders"),
}


def _load_raw() -> dict[str, pd.DataFrame]:
    return {
        "enroll": pd.read_csv(RAW_DIR / "enrollment.csv"),
        "visits": pd.read_csv(RAW_DIR / "visits.csv"),
        "vitals": pd.read_csv(RAW_DIR / "vitals.csv"),
        "aes": pd.read_csv(RAW_DIR / "adverse_events.csv"),
    }


def build_dm(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    enroll = raw["enroll"].copy()
    visits = raw["visits"].copy()
    ref_start = visits.groupby("USUBJID")["VISIT_DATE"].min().rename("RFSTDTC")
    ref_end = visits.groupby("USUBJID")["VISIT_DATE"].max().rename("RFENDTC")
    dm = enroll.merge(ref_start, on="USUBJID").merge(ref_end, on="USUBJID")
    dm["DOMAIN"] = "DM"
    dm["ARMCD"] = dm["TREATMENT"].map(lambda t: ARM_MAP[t][0])
    dm["ARM"] = dm["TREATMENT"].map(lambda t: ARM_MAP[t][1])
    dm["ACTARMCD"] = dm["ARMCD"]
    dm["ACTARM"] = dm["ARM"]
    dm["BRTHDTC"] = dm["DOB"]
    dm["AGEU"] = "YEARS"
    dm = dm[[
        "STUDYID", "DOMAIN", "USUBJID", "SUBJID", "RFSTDTC", "RFENDTC",
        "SITEID", "BRTHDTC", "AGE", "AGEU", "SEX", "RACE", "ETHNIC",
        "ARMCD", "ARM", "ACTARMCD", "ACTARM", "COUNTRY",
    ]]
    dm.to_csv(SDTM_DIR / "DM.csv", index=False)
    return dm


def build_ae(raw: dict[str, pd.DataFrame], dm: pd.DataFrame) -> pd.DataFrame:
    ae_raw = raw["aes"].copy()
    if ae_raw.empty:
        ae = pd.DataFrame(columns=[
            "STUDYID", "DOMAIN", "USUBJID", "AESEQ", "AETERM", "AEDECOD",
            "AEBODSYS", "AESEV", "AESER", "AEREL", "AEOUT", "AESTDTC", "AEENDTC",
        ])
        ae.to_csv(SDTM_DIR / "AE.csv", index=False)
        return ae
    ae_raw["STUDYID"] = STUDY_ID
    ae_raw["DOMAIN"] = "AE"
    ae_raw["AEDECOD"] = ae_raw["AE_TERM"].map(lambda t: AE_DICT[t][0])
    ae_raw["AEBODSYS"] = ae_raw["AE_TERM"].map(lambda t: AE_DICT[t][1])
    ae_raw["AESEV"] = ae_raw["AE_SEVERITY"]
    ae_raw["AESER"] = ae_raw["AE_SERIOUS"]
    ae_raw["AEREL"] = ae_raw["AE_RELATIONSHIP"].map(
        {"RELATED": "RELATED", "NOT RELATED": "NOT RELATED"}
    )
    ae_raw["AEOUT"] = ae_raw["AE_OUTCOME"]
    ae_raw["AETERM"] = ae_raw["AE_TERM"]
    ae_raw["AESTDTC"] = ae_raw["AE_START_DATE"]
    ae_raw["AEENDTC"] = ae_raw["AE_END_DATE"]
    ae_raw = ae_raw.sort_values(["USUBJID", "AESTDTC"])
    ae_raw["AESEQ"] = ae_raw.groupby("USUBJID").cumcount() + 1
    ae = ae_raw[[
        "STUDYID", "DOMAIN", "USUBJID", "AESEQ", "AETERM", "AEDECOD", "AEBODSYS",
        "AESEV", "AESER", "AEREL", "AEOUT", "AESTDTC", "AEENDTC",
    ]].reset_index(drop=True)
    ae.to_csv(SDTM_DIR / "AE.csv", index=False)
    return ae


def build_vs(raw: dict[str, pd.DataFrame], dm: pd.DataFrame) -> pd.DataFrame:
    vs = raw["vitals"].copy()
    vs["STUDYID"] = STUDY_ID
    vs["DOMAIN"] = "VS"
    vs["VSTESTCD"] = vs["TEST_CODE"]
    vs["VSTEST"] = vs["TEST_NAME"]
    vs["VSORRES"] = vs["RESULT"].astype(str)
    vs["VSORRESU"] = vs["UNIT"]
    vs["VSSTRESC"] = vs["VSORRES"]
    vs["VSSTRESN"] = pd.to_numeric(vs["VSORRES"], errors="coerce")
    vs["VSSTRESU"] = vs["VSORRESU"]
    vs["VSDTC"] = vs["VISIT_DATE"]
    visit_num = {v: n for n, v, _ in VISIT_META}
    vs["VISITNUM"] = vs["VISIT"].map(visit_num)
    rf = dm.set_index("USUBJID")["RFSTDTC"].to_dict()
    vs["VSDY"] = vs.apply(
        lambda r: (date.fromisoformat(r["VSDTC"]) - date.fromisoformat(rf[r["USUBJID"]])).days
        + (1 if date.fromisoformat(r["VSDTC"]) >= date.fromisoformat(rf[r["USUBJID"]]) else 0),
        axis=1,
    )
    vs = vs.sort_values(["USUBJID", "VISITNUM", "VSTESTCD"])
    vs["VSSEQ"] = vs.groupby("USUBJID").cumcount() + 1
    vs = vs[[
        "STUDYID", "DOMAIN", "USUBJID", "VSSEQ", "VSTESTCD", "VSTEST",
        "VSORRES", "VSORRESU", "VSSTRESC", "VSSTRESN", "VSSTRESU",
        "VISITNUM", "VISIT", "VSDTC", "VSDY",
    ]].reset_index(drop=True)
    vs.to_csv(SDTM_DIR / "VS.csv", index=False)
    return vs


def build_tv() -> pd.DataFrame:
    rows = [
        {
            "STUDYID": STUDY_ID,
            "DOMAIN": "TV",
            "VISITNUM": n,
            "VISIT": v,
            "VISITDY": d,
            "ARMCD": "",
            "ARM": "",
            "TVSTRL": "",
            "TVENRL": "",
        }
        for n, v, d in VISIT_META
    ]
    tv = pd.DataFrame(rows)
    tv.to_csv(SDTM_DIR / "TV.csv", index=False)
    return tv


def build_sv(raw: dict[str, pd.DataFrame], dm: pd.DataFrame) -> pd.DataFrame:
    visits = raw["visits"].copy()
    visit_num = {v: n for n, v, _ in VISIT_META}
    visits["STUDYID"] = STUDY_ID
    visits["DOMAIN"] = "SV"
    visits["VISITNUM"] = visits["VISIT"].map(visit_num)
    visits["SVSTDTC"] = visits["VISIT_DATE"]
    visits["SVENDTC"] = visits["VISIT_DATE"]
    rf = dm.set_index("USUBJID")["RFSTDTC"].to_dict()
    visits["SVSTDY"] = visits.apply(
        lambda r: (date.fromisoformat(r["SVSTDTC"]) - date.fromisoformat(rf[r["USUBJID"]])).days
        + (1 if date.fromisoformat(r["SVSTDTC"]) >= date.fromisoformat(rf[r["USUBJID"]]) else 0),
        axis=1,
    )
    sv = visits[[
        "STUDYID", "DOMAIN", "USUBJID", "VISITNUM", "VISIT",
        "SVSTDTC", "SVENDTC", "SVSTDY",
    ]].sort_values(["USUBJID", "VISITNUM"]).reset_index(drop=True)
    sv.to_csv(SDTM_DIR / "SV.csv", index=False)
    return sv


def main() -> None:
    raw = _load_raw()
    dm = build_dm(raw)
    build_ae(raw, dm)
    build_vs(raw, dm)
    build_tv()
    build_sv(raw, dm)
    print(f"SDTM domains written to {SDTM_DIR}")


if __name__ == "__main__":
    main()
