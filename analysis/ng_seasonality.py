import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Helper to get continuous front contract for NG (Henry Hub) from Yahoo
# We'll use 'NG=F' as front-month continuous.
def load_ng_front(start='2007-01-01') -> pd.DataFrame:
    df = yf.download('NG=F', start=start, auto_adjust=False, progress=False)
    df = df.rename(columns={'Adj Close':'AdjClose'})
    df = df.dropna(how='all')
    df.index = pd.to_datetime(df.index)
    return df[['Open','High','Low','Close','AdjClose','Volume']] if 'AdjClose' in df.columns else df[['Open','High','Low','Close','Volume']]

# Approximate contract schedule: NG contracts expire 3 business days prior to first calendar day of delivery month.
# For November contract (NGX), delivery month = November, so expiry is typically around Oct 29-31 depending on weekdays/holidays.
def estimate_ngx_expiry(year: int) -> pd.Timestamp:
    # Find the first day of November
    first_nov = pd.Timestamp(year=year, month=11, day=1)
    # Expiration is 3 business days prior to first of month
    # Count back business days
    bd = 0
    d = first_nov
    while bd < 3:
        d = d - pd.Timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            bd += 1
    return d.normalize()

def compute_seasonal_path(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    # For each year, anchor the path from a fixed lookback window before estimated expiry until expiry
    # We'll use 30 trading days to expiry window.
    window = 30
    out = []
    for y in years:
        expiry = estimate_ngx_expiry(y)
        # slice df up to expiry date (inclusive)
        sub = df.loc[:expiry]
        # align to last 'window' trading days
        sub = sub.tail(window + 1)  # include expiry day
        if len(sub) < window + 1:
            continue
        # Ensure 1-D array for pandas Series construction
        prices = np.ravel(sub['Close'].to_numpy())
        # normalize to price at T-30 for comparability
        base = prices[0]
        rel = prices / base
        out.append(pd.Series(rel, name=y))
    season = pd.concat(out, axis=1)
    # Index from -window .. 0 (expiry)
    season.index = pd.Index(range(-window, 1))  # e.g., -30..0
    season.index.name = 'days_to_expiry'
    return season

def summarize_distribution(season: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame({
        'mean': season.mean(axis=1),
        'p16': season.quantile(0.16, axis=1),
        'p84': season.quantile(0.84, axis=1),
        'p05': season.quantile(0.05, axis=1),
        'p95': season.quantile(0.95, axis=1),
    })
    return stats

def forecast_to_expiry(current_price: float, season_stats: pd.DataFrame, current_dte: int) -> dict:
    # Clamp to available index range
    if current_dte < season_stats.index.min():
        current_dte = season_stats.index.min()
    if current_dte > 0:
        current_dte = 0
    # Ratio from current day to expiry: mean at 0 divided by mean at current_dte
    mean_today = season_stats.loc[current_dte, 'mean']
    mean_exp = season_stats.loc[0, 'mean']
    ratio_mean = mean_exp / mean_today
    # Use quantiles for range
    ratio_lo = season_stats.loc[0, 'p16'] / season_stats.loc[current_dte, 'p16']
    ratio_hi = season_stats.loc[0, 'p84'] / season_stats.loc[current_dte, 'p84']
    # 90% interval (5th to 95th percentile)
    ratio_p05 = season_stats.loc[0, 'p05'] / season_stats.loc[current_dte, 'p05']
    ratio_p95 = season_stats.loc[0, 'p95'] / season_stats.loc[current_dte, 'p95']
    return {
        'point_estimate': float(current_price * ratio_mean),
        'range_1sigma': (float(current_price * ratio_lo), float(current_price * ratio_hi)),
        'range_90': (float(current_price * ratio_p05), float(current_price * ratio_p95)),
        'ratios': {'mean': float(ratio_mean), 'lo': float(ratio_lo), 'hi': float(ratio_hi)},
        'current_dte': int(current_dte),
    }

if __name__ == '__main__':
    df = load_ng_front('2010-01-01')
    # Build years excluding current if incomplete
    today = pd.Timestamp.today().normalize()
    this_year = today.year
    years = [y for y in range(2010, this_year) ]  # use completed seasons through last year
    season = compute_seasonal_path(df, years)
    stats = summarize_distribution(season)
    # Estimate this year's expiry and compute days-to-expiry relative to last trading day available
    expiry = estimate_ngx_expiry(this_year)
    # Find the most recent close and infer trading days to expiry by index distance in df
    # Align df index to business days only
    last_date = df.index[-1]
    # approximate trading dte by business day count
    bdays = pd.bdate_range(last_date, expiry)
    dte = -(len(bdays) - 1)  # 0 at expiry, -N days before
    current_price = float(df['Close'].iloc[-1])
    fc = forecast_to_expiry(current_price, stats, dte)
    print('Last date:', last_date.date())
    print('Current price:', round(current_price, 4))
    print('Estimated expiry:', expiry.date())
    print('Approx trading days to expiry (negative means before):', dte)
    print('Seasonal ratios (mean/lo/hi):', fc['ratios'])
    print('Point estimate:', round(fc['point_estimate'], 4))
    print('Range 1σ:', tuple(round(x,4) for x in fc['range_1sigma']))
    print('Range 90%:', tuple(round(x,4) for x in fc['range_90']))
