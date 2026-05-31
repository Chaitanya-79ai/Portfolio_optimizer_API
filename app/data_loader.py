"""Excel workbook loading and cleaning utilities.

The optimizer depends on three sheets: fund metadata, fund returns, and factor
returns. This module centralizes validation and normalization so the rest of the
application can work with predictable DataFrames.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Workbook configuration and errors
# ---------------------------------------------------------------------------


# The workbook is expected at the project root so the API and tests use one source.
DATA_FILE = Path(__file__).resolve().parents[1] / "Data.xlsx"


class DataLoadError(RuntimeError):
    """Raised when the Excel workbook cannot be loaded."""


# ---------------------------------------------------------------------------
# Public data accessors
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_data(data_file: Path = DATA_FILE) -> dict[str, pd.DataFrame]:
    """Load and normalize the Excel workbook once per process."""
    if not data_file.exists():
        raise DataLoadError(f"Data file not found: {data_file}")

    try:
        # Keep sheet names explicit so workbook issues produce actionable errors.
        fund_info = pd.read_excel(data_file, sheet_name="Fund Info")
        fund_returns = pd.read_excel(data_file, sheet_name="Fund Returns")
        factor_returns = pd.read_excel(data_file, sheet_name="Factor Returns")
    except Exception as exc:
        raise DataLoadError(f"Could not read workbook: {exc}") from exc

    fund_info = _clean_fund_info(fund_info)
    fund_returns = _clean_return_sheet(fund_returns, "ticker")
    factor_returns = _clean_return_sheet(factor_returns, "index_ticker")

    return {
        "fund_info": fund_info,
        "fund_returns": fund_returns,
        "factor_returns": factor_returns,
    }


def get_fund_info() -> pd.DataFrame:
    """Return a copy of cleaned fund metadata."""
    return load_data()["fund_info"].copy()


def get_fund_returns() -> pd.DataFrame:
    """Return a copy of cleaned fund return history."""
    return load_data()["fund_returns"].copy()


def get_factor_returns() -> pd.DataFrame:
    """Return a copy of cleaned factor return history."""
    return load_data()["factor_returns"].copy()


def get_available_tickers() -> list[str]:
    """Return sorted tickers that can be optimized by the API."""
    fund_info = load_data()["fund_info"]
    return sorted(fund_info["ticker"].unique().tolist())


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------


def _clean_fund_info(df: pd.DataFrame) -> pd.DataFrame:
    """Validate fund metadata and normalize fields used by the optimizer."""
    required = {"ticker", "fund_name", "dividend_yield"}
    missing = required - set(df.columns)
    if missing:
        raise DataLoadError(f"Fund Info sheet missing columns: {sorted(missing)}")

    cleaned = df.copy()
    # Normalize string fields once so downstream code can rely on exact matches.
    cleaned["ticker"] = cleaned["ticker"].astype(str).str.upper().str.strip()
    cleaned["fund_name"] = cleaned["fund_name"].astype(str).str.strip()
    cleaned["dividend_yield"] = (
        pd.to_numeric(cleaned["dividend_yield"], errors="coerce").fillna(0.0)
    )
    return cleaned


def _clean_return_sheet(df: pd.DataFrame, symbol_column: str) -> pd.DataFrame:
    """Validate and normalize a returns worksheet into long-form daily returns."""
    required = {"date", "total_return", symbol_column}
    missing = required - set(df.columns)
    if missing:
        raise DataLoadError(
            f"Return sheet with {symbol_column} missing columns: {sorted(missing)}"
        )

    cleaned = df.copy()
    cleaned["date"] = _parse_excel_dates(cleaned["date"])
    cleaned["total_return"] = pd.to_numeric(cleaned["total_return"], errors="coerce")
    cleaned[symbol_column] = cleaned[symbol_column].astype(str).str.strip()

    if symbol_column == "ticker":
        # Fund tickers are user-facing symbols, so match API normalization.
        cleaned[symbol_column] = cleaned[symbol_column].str.upper()

    # Drop incomplete rows rather than letting NaNs break optimization math later.
    cleaned = cleaned.dropna(subset=["date", "total_return", symbol_column])
    return cleaned.sort_values(["date", symbol_column]).reset_index(drop=True)


def _parse_excel_dates(series: pd.Series) -> pd.Series:
    # Excel serial dates appear when spreadsheets store dates as numbers.
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="D", origin="1899-12-30")
    return pd.to_datetime(series, errors="coerce")
