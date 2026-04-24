"""Generate TLFs (Tables) from ADaM datasets.

Produces three output tables in ``output/``:
 - Table 1: Demographic and Baseline Characteristics (from ADSL)
 - Table 2: Adverse Events by System Organ Class and Preferred Term (from ADAE)
 - Table 3: Summary of Vital Signs by PARAMCD (from ADVS)

Each table is written in four formats:
 - ``.csv`` (machine-readable, subject count carried as a sub-header row)
 - ``.txt`` (fixed-width human-readable report with two-line column headers)
 - ``.html`` (styled HTML with two-row ``<thead>``)
 - ``.pdf`` (landscape PDF rendered via reportlab)
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

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


def table_demographics(adsl: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
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
    col_ns = {trt: len(df) for trt, df in groups.items()}
    return df_out, col_ns


def table_ae_soc_pt(adae: pd.DataFrame, adsl: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
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

    df_out = pd.DataFrame(rows, columns=["SOC / Preferred Term", *trt_order])
    return df_out, trt_totals


def table_vs_summary(advs: pd.DataFrame, adsl: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
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

    df_out = pd.DataFrame(rows, columns=["PARAMCD", "PARAM", "Visit", "Statistic", *trt_order])
    n_by_trt = adsl.groupby("TRT01A", observed=False).size().to_dict()
    col_ns = {t: int(n_by_trt.get(t, 0)) for t in trt_order}
    return df_out, col_ns


def _n_sub(col: str, col_ns: dict[str, int]) -> str:
    return f"(N={col_ns[col]})" if col in col_ns else ""


def _write_csv(df: pd.DataFrame, path: Path, col_ns: dict[str, int]) -> None:
    sub = pd.DataFrame(
        [[_n_sub(c, col_ns) for c in df.columns]],
        columns=df.columns,
    )
    pd.concat([sub, df], ignore_index=True).to_csv(path, index=False)


def _write_txt(df: pd.DataFrame, path: Path, title: str, col_ns: dict[str, int]) -> None:
    col_widths = {
        c: max(
            len(str(c)),
            len(_n_sub(c, col_ns)),
            df[c].astype(str).map(len).max(),
        )
        for c in df.columns
    }
    header1 = "  ".join(str(c).ljust(col_widths[c]) for c in df.columns)
    header2 = "  ".join(_n_sub(c, col_ns).ljust(col_widths[c]) for c in df.columns)
    sep = "-" * len(header1)
    lines = [title, "=" * len(title), "", header1, header2, sep]
    for _, row in df.iterrows():
        lines.append("  ".join(str(row[c]).ljust(col_widths[c]) for c in df.columns))
    path.write_text("\n".join(lines) + "\n")


_HTML_CSS = """
  body { font-family: Arial, Helvetica, sans-serif; margin: 2em; color: #222; }
  h1 { font-size: 1.15em; margin-bottom: 0.8em; }
  table.tlf { border-collapse: collapse; font-size: 0.9em; }
  table.tlf thead tr { background: #f0f0f0; }
  table.tlf th, table.tlf td {
    border: 1px solid #999; padding: 4px 8px; vertical-align: top;
  }
  table.tlf th { text-align: center; font-weight: 600; }
  table.tlf td { white-space: pre; font-family: "Courier New", monospace; }
  table.tlf td.label, table.tlf th.label { text-align: left; font-family: Arial; white-space: normal; }
  table.tlf tr.subhead th { font-weight: 400; font-style: italic; }
"""


def _write_html(df: pd.DataFrame, path: Path, title: str, col_ns: dict[str, int]) -> None:
    label_cols = [c for c in df.columns if c not in col_ns]
    head_row1 = "".join(
        f'<th class="{"label" if c in label_cols else "num"}">{escape(str(c))}</th>'
        for c in df.columns
    )
    head_row2 = "".join(
        f'<th class="{"label" if c in label_cols else "num"}">{escape(_n_sub(c, col_ns))}</th>'
        for c in df.columns
    )
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(
            f'<td class="{"label" if c in label_cols else "num"}">{escape(str(row[c]))}</td>'
            for c in df.columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    html = (
        "<!DOCTYPE html>\n"
        f"<html><head><meta charset='utf-8'><title>{escape(title)}</title>"
        f"<style>{_HTML_CSS}</style></head><body>"
        f"<h1>{escape(title)}</h1>"
        "<table class='tlf'>"
        f"<thead><tr>{head_row1}</tr><tr class='subhead'>{head_row2}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></body></html>\n"
    )
    path.write_text(html)


def _write_pdf(df: pd.DataFrame, path: Path, title: str, col_ns: dict[str, int]) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    styles = getSampleStyleSheet()
    header_row1 = list(df.columns)
    header_row2 = [_n_sub(c, col_ns) for c in df.columns]
    data = [header_row1, header_row2] + df.astype(str).values.tolist()
    tbl = Table(data, repeatRows=2)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 1), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Oblique"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, 1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0, 2), (-1, -1), [colors.whitesmoke, colors.white]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    doc.build([Paragraph(escape(title), styles["Title"]), Spacer(1, 6), tbl])


def _write_all(df: pd.DataFrame, stem: str, title: str, col_ns: dict[str, int]) -> None:
    _write_csv(df, OUT_DIR / f"{stem}.csv", col_ns)
    _write_txt(df, OUT_DIR / f"{stem}.txt", title, col_ns)
    _write_html(df, OUT_DIR / f"{stem}.html", title, col_ns)
    _write_pdf(df, OUT_DIR / f"{stem}.pdf", title, col_ns)


def main() -> None:
    adsl = pd.read_csv(ADAM_DIR / "ADSL.csv")
    adae = pd.read_csv(ADAM_DIR / "ADAE.csv")
    advs = pd.read_csv(ADAM_DIR / "ADVS.csv")

    t1, t1_ns = table_demographics(adsl)
    _write_all(t1, "t_demographics",
               "Table 1: Demographic and Baseline Characteristics (Safety Population)",
               t1_ns)

    t2, t2_ns = table_ae_soc_pt(adae, adsl)
    _write_all(t2, "t_ae_soc_pt",
               "Table 2: Treatment-Emergent Adverse Events by System Organ Class "
               "and Preferred Term (Safety Population)",
               t2_ns)

    t3, t3_ns = table_vs_summary(advs, adsl)
    _write_all(t3, "t_vs_summary",
               "Table 3: Summary Statistics of Vital Signs by Parameter and Visit",
               t3_ns)

    print(f"TLFs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
