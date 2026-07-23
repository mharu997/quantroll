"""Shared data loading and plot styling for the quantroll example notebooks.

Prices are daily adjusted closes cached under ``notebooks/data/``. If a cache
file is missing, it is re-downloaded with ``yfinance`` (optional dependency of
the notebooks only, not of quantroll).
"""

from pathlib import Path

import matplotlib as mpl
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"

ETFS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB",
        "XLRE", "XLC", "TLT", "GLD"]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]

STOCK_SECTOR = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "GOOGL": "Tech",
    "AMZN": "Tech", "META": "Tech", "TSLA": "Tech", "AMD": "Tech",
    "INTC": "Tech", "CRM": "Tech", "ADBE": "Tech", "NFLX": "Tech",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "WFC": "Financials", "C": "Financials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "JNJ": "Health", "PFE": "Health", "MRK": "Health", "UNH": "Health",
    "LLY": "Health", "ABBV": "Health",
    "PG": "Staples", "KO": "Staples", "PEP": "Staples", "WMT": "Staples",
    "COST": "Staples", "MCD": "Staples",
    "CAT": "Industrials", "BA": "Industrials", "GE": "Industrials",
    "HON": "Industrials", "UPS": "Industrials", "DE": "Industrials",
    "T": "TelecomMedia", "VZ": "TelecomMedia", "DIS": "TelecomMedia",
}
STOCKS = list(STOCK_SECTOR)
SECTORS = ["Tech", "Financials", "Energy", "Health", "Staples",
           "Industrials", "TelecomMedia"]

# Okabe-Ito: colorblind-safe categorical palette, assigned in FIXED order.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
           "#56B4E9", "#D55E00", "#F0E442", "#555555"]
SECTOR_COLOR = dict(zip(SECTORS, PALETTE))


def load_prices(kind: str = "etf") -> pd.DataFrame:
    """Daily adjusted closes; ``kind`` in {"etf", "stock"}."""
    f = DATA / f"{kind}_prices.csv"
    if not f.exists():
        import yfinance as yf

        tickers = ETFS if kind == "etf" else STOCKS
        px = yf.download(tickers, start="2012-01-01", auto_adjust=True,
                         progress=False)["Close"]
        DATA.mkdir(exist_ok=True)
        px.to_csv(f)
    px = pd.read_csv(f, index_col=0, parse_dates=True)
    px.index.name = "date"
    return px


def daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns; leading NaN preserved (pre-inception)."""
    r = px.pct_change(fill_method=None)
    return r.iloc[1:]


def use_style() -> None:
    """Recessive grids, no chart junk, consistent sizing."""
    mpl.rcParams.update({
        "figure.figsize": (10, 4.2),
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })
