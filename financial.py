"""Pandas based financial metric analysis (pure functions, easy to test)."""
import re
from pathlib import Path

import pandas as pd

_NUMERIC_LIKE = re.compile(r"^[\s$€£]*-?[\d,]+(\.\d+)?\s*[%bBmMkK]?\s*$")


def load_metrics(csv_path) -> pd.DataFrame:
    path = Path(str(csv_path).strip())
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got '{path.name}'")
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV contains no data rows")
    return df


def numeric_summary(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """describe() over numeric columns: count, mean, std, min, quartiles, max."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("CSV has no numeric columns to summarise")
    return numeric.describe().round(decimals)


def consistency_notes(df: pd.DataFrame) -> list:
    """Cheap numeric consistency checks that are reported alongside the stats."""
    notes = []
    for col, n in df.isna().sum().items():
        if n:
            notes.append(f"Column '{col}' has {n} missing value(s).")
    dupes = int(df.duplicated().sum())
    if dupes:
        notes.append(f"{dupes} duplicated row(s) found.")
    for col in df.select_dtypes(exclude="number").columns:
        values = df[col].dropna().astype(str)
        if len(values) and values.map(lambda v: bool(_NUMERIC_LIKE.match(v))).mean() > 0.5:
            notes.append(
                f"Column '{col}' looks numeric but contains text (e.g. '{values.iloc[0]}'); "
                "it was excluded from the statistics. Use plain numbers."
            )
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if (series < 0).any():
            notes.append(f"Column '{col}' contains negative values (valid for losses, check anyway).")
        if len(series) > 1 and series.nunique() == 1:
            notes.append(f"Column '{col}' is constant; standard deviation is 0.")
    return notes or ["No consistency issues detected."]


def summary_markdown(df: pd.DataFrame) -> str:
    """Markdown report used as the Data Analyst tool output."""
    summary = numeric_summary(df)
    notes = "\n".join(f"- {n}" for n in consistency_notes(df))
    return (
        f"Rows analysed: {len(df)} | Numeric columns: {', '.join(summary.columns)}\n\n"
        f"### Summary statistics\n\n{summary.to_markdown()}\n\n"
        f"### Data consistency checks\n\n{notes}"
    )
