"""Build ADaM-style analysis datasets from SDTM domains.

Produces ADSL, ADAE, ADVS as CSV files in ``data/adam/``. Follows CDISC ADaM
naming (TRT01P/TRT01A, BASE, CHG, AVAL, AVISIT, etc.) and standard analysis
flags (SAFFL, ITTFL, ANL01FL) at a level sufficient for TLFs.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SDTM_DIR = ROOT / "data" / "sdtm"
ADAM_DIR = ROOT / "data" / "adam"
ADAM_DIR.mkdir(parents=True, exist_ok=True)


def _age_group(age: int) -> str:
    if age < 45:
        return "<45"
    if age < 65:
        return "45-64"
    return ">=65"


def build_adsl() -> pd.DataFrame:
    dm = pd.read_csv(SDTM_DIR / "DM.csv")
    sv = pd.read_csv(SDTM_DIR / "SV.csv")
    adsl = dm.copy()
    adsl["TRT01P"] = adsl["ARM"]
    adsl["TRT01PN"] = adsl["ARMCD"].map({"PBO": 0, "ABC50": 1})
    adsl["TRT01A"] = adsl["ACTARM"]
    adsl["TRT01AN"] = adsl["ACTARMCD"].map({"PBO": 0, "ABC50": 1})
    adsl["AGEGR1"] = adsl["AGE"].apply(_age_group)
    adsl["AGEGR1N"] = adsl["AGEGR1"].map({"<45": 1, "45-64": 2, ">=65": 3})
    adsl["SEXN"] = adsl["SEX"].map({"M": 1, "F": 2})
    adsl["TRTSDT"] = adsl["RFSTDTC"]
    adsl["TRTEDT"] = adsl["RFENDTC"]
    adsl["TRTDURD"] = adsl.apply(
        lambda r: (date.fromisoformat(r["TRTEDT"]) - date.fromisoformat(r["TRTSDT"])).days + 1,
        axis=1,
    )
    eot = sv[sv["VISIT"] == "END OF TREATMENT"]["USUBJID"].unique()
    adsl["EOTFL"] = np.where(adsl["USUBJID"].isin(eot), "Y", "N")
    adsl["COMPLFL"] = adsl["EOTFL"]
    adsl["SAFFL"] = "Y"
    adsl["ITTFL"] = "Y"
    adsl = adsl[[
        "STUDYID", "USUBJID", "SUBJID", "SITEID", "COUNTRY",
        "AGE", "AGEU", "AGEGR1", "AGEGR1N", "SEX", "SEXN",
        "RACE", "ETHNIC",
        "ARM", "ARMCD", "ACTARM", "ACTARMCD",
        "TRT01P", "TRT01PN", "TRT01A", "TRT01AN",
        "TRTSDT", "TRTEDT", "TRTDURD",
        "SAFFL", "ITTFL", "COMPLFL", "EOTFL",
    ]]
    adsl.to_csv(ADAM_DIR / "ADSL.csv", index=False)
    return adsl


def build_adae(adsl: pd.DataFrame) -> pd.DataFrame:
    ae = pd.read_csv(SDTM_DIR / "AE.csv")
    keys = [
        "STUDYID", "USUBJID", "SITEID", "AGE", "AGEGR1", "SEX", "RACE",
        "TRT01P", "TRT01PN", "TRT01A", "TRT01AN", "TRTSDT", "TRTEDT", "SAFFL",
    ]
    adae = ae.merge(adsl[keys], on=["STUDYID", "USUBJID"], how="left")
    adae["TRTA"] = adae["TRT01A"]
    adae["TRTAN"] = adae["TRT01AN"]
    adae["AESTDT"] = adae["AESTDTC"]
    adae["AEENDT"] = adae["AEENDTC"]
    adae["ASTDY"] = adae.apply(
        lambda r: (date.fromisoformat(r["AESTDT"]) - date.fromisoformat(r["TRTSDT"])).days
        + (1 if date.fromisoformat(r["AESTDT"]) >= date.fromisoformat(r["TRTSDT"]) else 0),
        axis=1,
    )
    adae["AENDY"] = adae.apply(
        lambda r: (date.fromisoformat(r["AEENDT"]) - date.fromisoformat(r["TRTSDT"])).days
        + (1 if date.fromisoformat(r["AEENDT"]) >= date.fromisoformat(r["TRTSDT"]) else 0),
        axis=1,
    )
    adae["TRTEMFL"] = np.where(adae["ASTDY"] >= 1, "Y", "N")
    adae["AOCCFL"] = ""
    first_idx = (
        adae[adae["TRTEMFL"] == "Y"]
        .sort_values(["USUBJID", "ASTDY", "AESEQ"])
        .drop_duplicates(["USUBJID", "AEDECOD"], keep="first")
        .index
    )
    adae.loc[first_idx, "AOCCFL"] = "Y"
    adae["ANL01FL"] = np.where(adae["TRTEMFL"] == "Y", "Y", "")
    adae = adae[[
        "STUDYID", "USUBJID", "SITEID", "AGE", "AGEGR1", "SEX", "RACE",
        "TRT01A", "TRT01AN", "TRTA", "TRTAN", "SAFFL",
        "AESEQ", "AETERM", "AEDECOD", "AEBODSYS", "AESEV", "AESER", "AEREL", "AEOUT",
        "AESTDT", "AEENDT", "ASTDY", "AENDY", "TRTEMFL", "AOCCFL", "ANL01FL",
    ]]
    adae.to_csv(ADAM_DIR / "ADAE.csv", index=False)
    return adae


PARAM_META = {
    "SYSBP": ("Systolic Blood Pressure (mmHg)", "mmHg"),
    "DIABP": ("Diastolic Blood Pressure (mmHg)", "mmHg"),
    "PULSE": ("Pulse Rate (beats/min)", "beats/min"),
    "TEMP": ("Temperature (C)", "C"),
    "WEIGHT": ("Weight (kg)", "kg"),
    "HEIGHT": ("Height (cm)", "cm"),
}


def build_advs(adsl: pd.DataFrame) -> pd.DataFrame:
    vs = pd.read_csv(SDTM_DIR / "VS.csv")
    keys = [
        "STUDYID", "USUBJID", "SITEID", "TRT01P", "TRT01PN", "TRT01A", "TRT01AN",
        "AGE", "AGEGR1", "SEX", "RACE", "TRTSDT", "SAFFL",
    ]
    advs = vs.merge(adsl[keys], on=["STUDYID", "USUBJID"], how="left")
    advs["PARAMCD"] = advs["VSTESTCD"]
    advs["PARAM"] = advs["PARAMCD"].map(lambda c: PARAM_META.get(c, (c, ""))[0])
    advs["PARAMN"] = advs["PARAMCD"].astype("category").cat.codes + 1
    advs["AVAL"] = advs["VSSTRESN"]
    advs["AVALU"] = advs["VSSTRESU"]
    advs["AVISIT"] = advs["VISIT"].str.title()
    advs["AVISITN"] = advs["VISITNUM"]
    advs["ADT"] = advs["VSDTC"]
    advs["ADY"] = advs["VSDY"]
    advs["TRTA"] = advs["TRT01A"]
    advs["TRTAN"] = advs["TRT01AN"]
    baseline_mask = advs["VISIT"] == "BASELINE"
    base_df = (
        advs[baseline_mask]
        .groupby(["USUBJID", "PARAMCD"])["AVAL"]
        .mean()
        .rename("BASE")
        .reset_index()
    )
    advs = advs.merge(base_df, on=["USUBJID", "PARAMCD"], how="left")
    advs["ABLFL"] = np.where(baseline_mask, "Y", "")
    advs["CHG"] = advs["AVAL"] - advs["BASE"]
    advs["PCHG"] = np.where(advs["BASE"].ne(0), (advs["CHG"] / advs["BASE"]) * 100, np.nan)
    advs["ANL01FL"] = "Y"
    advs = advs[[
        "STUDYID", "USUBJID", "SITEID",
        "TRT01A", "TRT01AN", "TRTA", "TRTAN", "AGE", "AGEGR1", "SEX", "RACE",
        "PARAMCD", "PARAM", "PARAMN", "AVAL", "AVALU",
        "AVISIT", "AVISITN", "ADT", "ADY",
        "BASE", "CHG", "PCHG", "ABLFL", "SAFFL", "ANL01FL",
    ]]
    advs.to_csv(ADAM_DIR / "ADVS.csv", index=False)
    return advs


def main() -> None:
    adsl = build_adsl()
    build_adae(adsl)
    build_advs(adsl)
    print(f"ADaM datasets written to {ADAM_DIR}")


if __name__ == "__main__":
    main()
