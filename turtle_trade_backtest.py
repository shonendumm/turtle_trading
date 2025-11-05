# turtle_trade.py
"""
Turtle-style trading analyzer + position sizing + simple backtest.
Author: ChatGPT (example)
Requirements: pip install yfinance pandas numpy matplotlib
"""

import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

# ----- Utilities & indicators -----
def download_data(ticker: str, start: str = "2018-01-01", end: str = None) -> pd.DataFrame:
    """Download daily OHLCV from yfinance and return a DataFrame with datetime index."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}.")
    # Handle both single-level and multi-level column structures
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    # Select only the columns that exist
    available_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if col in df.columns]
    df = df[available_cols].copy()
    df.index = pd.to_datetime(df.index)
    return df

def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """Compute True Range (TR) series."""
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def compute_atr(df: pd.DataFrame, atr_window: int = 20, use_wilder: bool = False) -> pd.Series:
    """Compute ATR (N). If use_wilder True, use Wilder smoothing (exponential-like)."""
    tr = compute_true_range(df)
    if use_wilder:
        # Wilder smoothing: first value is simple mean, then: ATR_t = (ATR_{t-1}*(n-1) + TR_t)/n
        atr = tr.rolling(window=atr_window).mean().copy()
        atr.iloc[atr_window:] = np.nan  # placeholder; we'll compute iteratively
        # compute the Wilder-smoothed ATR iteratively
        atr_values = []
        trs = tr.values
        n = atr_window
        # first ATR value is simple mean of first n TRs
        if len(trs) >= n:
            first_atr = trs[1:n+1].mean()  # skip index 0 because TR for first row may be NaN due to shift
        else:
            return pd.Series(index=df.index, dtype=float)
        atr_values = [np.nan] * (n)  # until index n
        atr_prev = first_atr
        atr_values.append(atr_prev)
        for t in range(n+1, len(trs)):
            atr_prev = (atr_prev * (n - 1) + trs[t]) / n
            atr_values.append(atr_prev)
        # align length
        atr_series = pd.Series(data=atr_values, index=df.index[:len(atr_values)])
        atr_series = atr_series.reindex(df.index)  # pad with NaN if needed
        return atr_series
    else:
        return tr.rolling(window=atr_window, min_periods=1).mean()

def compute_breakouts(df: pd.DataFrame, lookback_entry: int = 55, lookback_exit: int = 20) -> pd.DataFrame:
    """Compute breakout levels and signals based on highest high and lowest low."""
    df = df.copy()
    df['HH_entry'] = df['High'].rolling(window=lookback_entry, min_periods=lookback_entry).max().shift(1)  # prior breakout level
    df['LL_exit']  = df['Low'].rolling(window=lookback_exit, min_periods=lookback_exit).min().shift(1)
    df['HH20'] = df['High'].rolling(window=20, min_periods=20).max().shift(1)
    # Entry signal: today's close > HH_entry
    df['entry_signal'] = (df['Close'] > df['HH_entry']).astype(int)
    # Exit signal: today's close < LL_exit
    df['exit_signal']  = (df['Close'] < df['LL_exit']).astype(int)
    return df

# ----- Position sizing -----
def turtle_position_size(equity: float, atr: float, risk_per_trade: float = 0.01, stop_multiplier: float = 2.0,
                         allow_fractional: bool = False) -> int:
    """
    Compute number of shares to buy using a Turtle-style sizing:
      shares = floor((equity * risk_per_trade) / (stop_multiplier * ATR))
    where per-share risk = stop_multiplier * ATR.
    """
    if atr <= 0 or np.isnan(atr):
        return 0
    per_share_risk = stop_multiplier * atr
    if per_share_risk <= 0:
        return 0
    raw_shares = (equity * risk_per_trade) / per_share_risk
    if allow_fractional:
        return raw_shares
    return int(math.floor(raw_shares))

# ----- Simple backtester (single ticker) -----
def run_simple_turtle_backtest(df: pd.DataFrame,
                               starting_equity: float = 100_000,
                               risk_per_trade: float = 0.01,
                               stop_multiplier: float = 2.0,
                               lookback_entry: int = 55,
                               lookback_exit: int = 20,
                               atr_window: int = 20,
                               use_wilder_atr: bool = False,
                               max_units: int = 4) -> Tuple[pd.DataFrame, Dict]:
    """
    A simple backtest that:
      - uses 55-day breakout to enter (one 'unit' per breakout),
      - uses 20-day low as exit,
      - uses ATR-based sizing where each new unit is sized with current equity,
      - uses stop at entry - stop_multiplier * ATR (per unit).
    This is illustrative and omits many real-world details (commissions, slippage, partial fills, margin, etc.).
    """
    df = df.copy()
    df['ATR'] = compute_atr(df, atr_window, use_wilder_atr)
    df = compute_breakouts(df, lookback_entry, lookback_exit)
    nrows = len(df)
    equity = starting_equity
    cash = starting_equity
    positions = 0
    entry_price = None
    stop_price = None
    trades = []  # list of trades for analysis
    equity_curve = []

    for i in range(nrows):
        row = df.iloc[i]
        date = df.index[i]
        close = row['Close']
        atr = row['ATR']
        entry_signal = row['entry_signal']
        exit_signal = row['exit_signal']

        # Check exit first: if exit_signal and we have position -> liquidate everything
        if exit_signal and positions > 0:
            proceeds = positions * close
            pnl = proceeds - positions * entry_price
            cash += proceeds
            equity = cash  # no other assets in this simple backtest
            trades.append({
                'date': date, 'type': 'exit', 'price': close, 'shares': positions, 'pnl': pnl
            })
            positions = 0
            entry_price = None
            stop_price = None

        # Check stop loss: if we have a position and today's low <= stop -> stop hit
        if positions > 0 and row['Low'] <= stop_price:
            # assume stop executed at stop_price (could be row['Low'] in reality)
            executed_price = stop_price
            proceeds = positions * executed_price
            pnl = proceeds - positions * entry_price
            cash += proceeds
            equity = cash
            trades.append({
                'date': date, 'type': 'stop', 'price': executed_price, 'shares': positions, 'pnl': pnl
            })
            positions = 0
            entry_price = None
            stop_price = None

        # Entry: if entry_signal and we have fewer than max_units -> buy one unit sized by current equity
        if entry_signal and positions == 0:
            # open first unit
            shares = turtle_position_size(equity, atr, risk_per_trade, stop_multiplier)
            if shares > 0:
                cost = shares * close
                # allow allocation only if cash available (otherwise skip)
                if cost <= cash:
                    cash -= cost
                    positions = shares
                    entry_price = close
                    stop_price = entry_price - stop_multiplier * atr
                    trades.append({
                        'date': date, 'type': 'entry', 'price': close, 'shares': shares
                    })
        elif entry_signal and positions > 0 and positions < max_units:
            # pyramid: buy an additional unit (size computed at current equity)
            shares = turtle_position_size(equity, atr, risk_per_trade, stop_multiplier)
            if shares > 0:
                cost = shares * close
                if cost <= cash:
                    cash -= cost
                    # new average entry price (simple weighted avg)
                    total_shares = positions + shares
                    new_entry_price = (entry_price * positions + close * shares) / total_shares
                    positions = total_shares
                    entry_price = new_entry_price
                    stop_price = entry_price - stop_multiplier * atr
                    trades.append({
                        'date': date, 'type': 'pyramid', 'price': close, 'shares': shares
                    })

        # update equity each day (mark to market)
        mtm = positions * close
        equity = cash + mtm
        equity_curve.append({'date': date, 'equity': equity, 'cash': cash, 'positions': positions, 'mtm': mtm})

    equity_df = pd.DataFrame(equity_curve).set_index('date')
    results = {
        'final_equity': equity,
        'starting_equity': starting_equity,
        'trades': trades,
        'equity_curve': equity_df,
    }
    return df, results

# ----- Quick reporting / plotting helpers -----
def simple_report(results: Dict):
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Starting equity: ${results['starting_equity']:,.2f}")
    print(f"Final equity:    ${results['final_equity']:,.2f}")
    print(f"Total return:    {((results['final_equity'] / results['starting_equity']) - 1) * 100:.2f}%")
    
    trades = results['trades']
    entries = [t for t in trades if t['type'] == 'entry']
    pyramids = [t for t in trades if t['type'] == 'pyramid']
    exits = [t for t in trades if t.get('type') in ('exit', 'stop')]
    wins = [t for t in exits if t.get('pnl', 0) > 0]
    losses = [t for t in exits if t.get('pnl', 0) <= 0]
    
    print(f"\nTRADE SUMMARY:")
    print(f"  Initial entries:  {len(entries)}")
    print(f"  Pyramid adds:     {len(pyramids)}")
    print(f"  Total exits:      {len(exits)}")
    print(f"  Winning exits:    {len(wins)} ({len(wins)/len(exits)*100:.1f}% win rate)" if len(exits) > 0 else "  Winning exits:    0")
    print(f"  Losing exits:     {len(losses)}")
    
    if wins:
        avg_win = sum(t['pnl'] for t in wins) / len(wins)
        print(f"  Avg winning PnL:  ${avg_win:,.2f}")
    if losses:
        avg_loss = sum(t['pnl'] for t in losses) / len(losses)
        print(f"  Avg losing PnL:   ${avg_loss:,.2f}")
    
    if len(results['equity_curve']) > 0:
        eq = results['equity_curve']['equity']
        drawdown = (eq.cummax() - eq) / eq.cummax()
        max_dd = drawdown.max()
        print(f"\nRISK METRICS:")
        print(f"  Max drawdown:     {max_dd:.2%}")
    
    # Show first few trades as examples
    print(f"\n{'='*60}")
    print(f"TRADE DETAILS (first 10):")
    print(f"{'='*60}")
    for i, t in enumerate(trades[:10]):
        pnl_str = f" | PnL: ${t['pnl']:,.2f}" if 'pnl' in t else ""
        print(f"{t['date'].strftime('%Y-%m-%d')} | {t['type'].upper():8s} | Price: ${t['price']:.2f} | Shares: {t['shares']}{pnl_str}")
    
    if len(trades) > 10:
        print(f"... ({len(trades) - 10} more trades)")
    print(f"{'='*60}\n")

def plot_equity_curve(results: Dict):
    eq = results['equity_curve']
    plt.figure(figsize=(10,5))
    plt.plot(eq.index, eq['equity'])
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.show()

def plot_trades_on_price(df: pd.DataFrame, results: Dict, ticker: str = ""):
    """Plot price chart with entry/exit points marked."""
    trades = results['trades']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top panel: Price and trades
    ax1.plot(df.index, df['Close'], label='Close Price', color='black', alpha=0.7, linewidth=1)
    
    # Plot breakout levels
    if 'HH_entry' in df.columns:
        ax1.plot(df.index, df['HH_entry'], label='55-day High (Entry)', 
                color='green', alpha=0.3, linewidth=1, linestyle='--')
    if 'LL_exit' in df.columns:
        ax1.plot(df.index, df['LL_exit'], label='20-day Low (Exit)', 
                color='red', alpha=0.3, linewidth=1, linestyle='--')
    
    # Mark entry points
    entries = [t for t in trades if t['type'] == 'entry']
    pyramids = [t for t in trades if t['type'] == 'pyramid']
    exits = [t for t in trades if t['type'] == 'exit']
    stops = [t for t in trades if t['type'] == 'stop']
    
    if entries:
        entry_dates = [t['date'] for t in entries]
        entry_prices = [t['price'] for t in entries]
        ax1.scatter(entry_dates, entry_prices, marker='^', color='green', 
                   s=100, label='Entry (Breakout)', zorder=5, edgecolors='darkgreen', linewidth=1.5)
    
    if pyramids:
        pyr_dates = [t['date'] for t in pyramids]
        pyr_prices = [t['price'] for t in pyramids]
        ax1.scatter(pyr_dates, pyr_prices, marker='^', color='lightgreen', 
                   s=80, label='Pyramid Add', zorder=5, edgecolors='green', linewidth=1)
    
    if exits:
        exit_dates = [t['date'] for t in exits]
        exit_prices = [t['price'] for t in exits]
        exit_colors = ['blue' if t.get('pnl', 0) > 0 else 'orange' for t in exits]
        ax1.scatter(exit_dates, exit_prices, marker='v', c=exit_colors, 
                   s=100, label='Exit (20-day Low)', zorder=5, edgecolors='darkblue', linewidth=1.5)
    
    if stops:
        stop_dates = [t['date'] for t in stops]
        stop_prices = [t['price'] for t in stops]
        ax1.scatter(stop_dates, stop_prices, marker='X', color='red', 
                   s=100, label='Stop Loss (2×ATR)', zorder=5, edgecolors='darkred', linewidth=1.5)
    
    ax1.set_title(f'{ticker} Price Chart with Turtle Trading Signals', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Equity curve
    eq = results['equity_curve']
    ax2.plot(eq.index, eq['equity'], color='darkblue', linewidth=1.5)
    ax2.fill_between(eq.index, results['starting_equity'], eq['equity'], 
                     where=(eq['equity'] >= results['starting_equity']), 
                     color='green', alpha=0.3, label='Profit')
    ax2.fill_between(eq.index, results['starting_equity'], eq['equity'], 
                     where=(eq['equity'] < results['starting_equity']), 
                     color='red', alpha=0.3, label='Loss')
    ax2.axhline(y=results['starting_equity'], color='gray', linestyle='--', 
               linewidth=1, alpha=0.7, label='Starting Equity')
    ax2.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Equity ($)')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ----- Example usage -----
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Turtle Trading Backtest')
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--start', type=str, default='2018-01-01',
                       help='Start date in YYYY-MM-DD format (default: 2018-01-01)')
    parser.add_argument('--capital', type=float, default=10_000,
                       help='Starting capital (default: 100000)')
    parser.add_argument('--risk', type=float, default=0.005,
                       help='Risk per trade as fraction (default: 0.01 = 1%%)')
    parser.add_argument('--stop-multiplier', type=float, default=2.0,
                       help='Stop loss multiplier of ATR (default: 2.0)')
    parser.add_argument('--entry-lookback', type=int, default=55,
                       help='Entry breakout lookback period (default: 55)')
    parser.add_argument('--exit-lookback', type=int, default=20,
                       help='Exit lookback period (default: 20)')
    parser.add_argument('--atr-window', type=int, default=20,
                       help='ATR calculation window (default: 20)')
    parser.add_argument('--max-units', type=int, default=4,
                       help='Maximum pyramid units (default: 4)')
    
    args = parser.parse_args()
    
    df = download_data(args.ticker, start=args.start)
    df, results = run_simple_turtle_backtest(df,
                                            starting_equity=args.capital,
                                            risk_per_trade=args.risk,
                                            stop_multiplier=args.stop_multiplier,
                                            lookback_entry=args.entry_lookback,
                                            lookback_exit=args.exit_lookback,
                                            atr_window=args.atr_window,
                                            use_wilder_atr=False,
                                            max_units=args.max_units)
    simple_report(results)
    plot_trades_on_price(df, results, args.ticker)

    print("Close chart to exit program")