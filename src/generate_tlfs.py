"""Generate TLFs (Tables) from ADaM datasets.

Produces three output tables in ``output/``:
 - Table 1: Demographic and Baseline Characteristics (from ADSL)
 - Table 2: Adverse Events by System Organ Class and Preferred Term (from ADAE)
 - Table 3: Summary of Vital Signs by PARAMCD (from ADVS)

Each table is written as both a ``.csv`` (machine-readable) and a ``.txt``
fixed-width report (human-readable).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADAM_DIR = ROOT / "data" / "adam"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _fmt_n_pct(n: int, denom: int) -> str:
    if denom == 0:
        return "  0"
    return f"{n:>3d} ({n / denom * 100:5.1f}%)"


def _fmt_mean_sd(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return ""
    return f"{s.mean():6.2f} ({s.std(ddof=1):5.2f})"


def _fmt_median_range(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return ""
    return f"{s.median():6.1f} ({s.min():.1f}, {s.max():.1f})"


def table_demographics(adsl: pd.DataFrame) -> pd.DataFrame:
    trt_order = (
        adsl[["TRT01P", "TRT01PN"]]
        .drop_duplicates()
        .sort_values("TRT01PN")["TRT01P"]
        .tolist()
    )
    groups = {trt: adsl[adsl["TRT01P"] == trt] for trt in trt_order}
    groups["Total"] = adsl
    col_order = trt_order + ["Total"]

    rows: list[dict] = []

    def add_row(label: str, values: dict) -> None:
        rows.append({"Characteristic": label, **values})

    add_row("Number of Subjects (N)",
            {trt: f"{len(df):d}" for trt, df in groups.items()})

    add_row("Age (years)", {trt: "" for trt in col_order})
    add_row("  Mean (SD)", {trt: _fmt_mean_sd(df["AGE"]) for trt, df in groups.items()})
    add_row("  Median (Min, Max)",
            {trt: _fmt_median_range(df["AGE"]) for trt, df in groups.items()})

    add_row("Age Group, n (%)", {trt: "" for trt in col_order})
    for grp in ["<45", "45-64", ">=65"]:
        add_row(
            f"  {grp}",
            {trt: _fmt_n_pct((df["AGEGR1"] == grp).sum(), len(df))
             for trt, df in groups.items()},
        )

    add_row("Sex, n (%)", {trt: "" for trt in col_order})
    for val, lbl in [("M", "Male"), ("F", "Female")]:
        add_row(
            f"  {lbl}",
            {trt: _fmt_n_pct((df["SEX"] == val).sum(), len(df))
             for trt, df in groups.items()},
        )

    add_row("Race, n (%)", {trt: "" for trt in col_order})
    for race in sorted(adsl["RACE"].unique()):
        add_row(
            f"  {race.title()}",
            {trt: _fmt_n_pct((df["RACE"] == race).sum(), len(df))
             for trt, df in groups.items()},
        )

    add_row("Ethnicity, n (%)", {trt: "" for trt in col_order})
    for eth in sorted(adsl["ETHNIC"].unique()):
        add_row(
            f"  {eth.title()}",
            {trt: _fmt_n_pct((df["ETHNIC"] == eth).sum(), len(df))
             for trt, df in groups.items()},
        )

    df_out = pd.DataFrame(rows, columns=["Characteristic", *col_order])
    return df_out


def table_ae_soc_pt(adae: pd.DataFrame, adsl: pd.DataFrame) -> pd.DataFrame:
    trt_order = (
        adsl[["TRT01A", "TRT01AN"]]
        .drop_duplicates()
        .sort_values("TRT01AN")["TRT01A"]
        .tolist()
    )
    n_by_trt = adsl.groupby("TRT01A", observed=False).size().to_dict()
    trt_totals = {t: n_by_trt.get(t, 0) for t in trt_order}
    teae = adae[adae["TRTEMFL"] == "Y"].copy()

    rows: list[dict] = []

    def fmt(df_subset: pd.DataFrame, trt: str) -> str:
        subjs = df_subset[df_subset["TRTA"] == trt]["USUBJID"].nunique()
        denom = trt_totals[trt]
        if denom == 0:
            return "  0"
        return f"{subjs:>3d} ({subjs / denom * 100:5.1f}%)"

    any_row = {trt: fmt(teae, trt) for trt in trt_order}
    rows.append({"SOC / Preferred Term": "Subjects with any TEAE", **any_row})

    for soc in sorted(teae["AEBODSYS"].dropna().unique()):
        soc_df = teae[teae["AEBODSYS"] == soc]
        rows.append({
            "SOC / Preferred Term": soc,
            **{trt: fmt(soc_df, trt) for trt in trt_order},
        })
        for pt in sorted(soc_df["AEDECOD"].dropna().unique()):
            pt_df = soc_df[soc_df["AEDECOD"] == pt]
            rows.append({
                "SOC / Preferred Term": f"  {pt.title()}",
                **{trt: fmt(pt_df, trt) for trt in trt_order},
            })

    header_cols = [f"{t} (N={trt_totals[t]})" for t in trt_order]
    rename = {t: h for t, h in zip(trt_order, header_cols)}
    df_out = pd.DataFrame(rows).rename(columns=rename)
    df_out = df_out[["SOC / Preferred Term", *header_cols]]
    return df_out


def table_vs_summary(advs: pd.DataFrame) -> pd.DataFrame:
    trt_order = (
        advs[["TRTA", "TRTAN"]]
        .drop_duplicates()
        .sort_values("TRTAN")["TRTA"]
        .tolist()
    )
    visits_order = (
        advs[["AVISIT", "AVISITN"]]
        .drop_duplicates()
        .sort_values("AVISITN")["AVISIT"]
        .tolist()
    )
    stats = ["N", "Mean", "SD", "Median", "Min", "Max"]

    def _stat(series: pd.Series, name: str) -> str:
        s = series.dropna()
        if s.empty:
            return ""
        if name == "N":
            return f"{int(s.count())}"
        if name == "Mean":
            return f"{s.mean():.2f}"
        if name == "SD":
            return f"{s.std(ddof=1):.2f}" if s.count() > 1 else ""
        if name == "Median":
            return f"{s.median():.2f}"
        if name == "Min":
            return f"{s.min():.2f}"
        if name == "Max":
            return f"{s.max():.2f}"
        return ""

    rows: list[dict] = []
    for pcd, grp in advs.groupby("PARAMCD", sort=False):
        param = grp["PARAM"].iloc[0]
        for visit in visits_order:
            vdf = grp[grp["AVISIT"] == visit]
            if vdf["AVAL"].dropna().empty:
                continue
            for stat in stats:
                row = {
                    "PARAMCD": pcd,
                    "PARAM": param,
                    "Visit": visit,
                    "Statistic": stat,
                }
                for trt in trt_order:
                    row[trt] = _stat(vdf[vdf["TRTA"] == trt]["AVAL"], stat)
                rows.append(row)

    return pd.DataFrame(rows, columns=["PARAMCD", "PARAM", "Visit", "Statistic", *trt_order])


def _write_txt(df: pd.DataFrame, path: Path, title: str) -> None:
    col_widths = {c: max(len(str(c)), df[c].astype(str).map(len).max()) for c in df.columns}
    header = "  ".join(str(c).ljust(col_widths[c]) for c in df.columns)
    sep = "-" * len(header)
    lines = [title, "=" * len(title), "", header, sep]
    for _, row in df.iterrows():
        lines.append("  ".join(str(row[c]).ljust(col_widths[c]) for c in df.columns))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    adsl = pd.read_csv(ADAM_DIR / "ADSL.csv")
    adae = pd.read_csv(ADAM_DIR / "ADAE.csv")
    advs = pd.read_csv(ADAM_DIR / "ADVS.csv")

    t1 = table_demographics(adsl)
    t1.to_csv(OUT_DIR / "t_demographics.csv", index=False)
    _write_txt(t1, OUT_DIR / "t_demographics.txt",
               "Table 1: Demographic and Baseline Characteristics (Safety Population)")

    t2 = table_ae_soc_pt(adae, adsl)
    t2.to_csv(OUT_DIR / "t_ae_soc_pt.csv", index=False)
    _write_txt(t2, OUT_DIR / "t_ae_soc_pt.txt",
               "Table 2: Treatment-Emergent Adverse Events by System Organ Class "
               "and Preferred Term (Safety Population)")

    t3 = table_vs_summary(advs)
    t3.to_csv(OUT_DIR / "t_vs_summary.csv", index=False)
    _write_txt(t3, OUT_DIR / "t_vs_summary.txt",
               "Table 3: Summary Statistics of Vital Signs by Parameter and Visit")

    print(f"TLFs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
