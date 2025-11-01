# turtle_trade_down.py
"""
Daily scanner for INVERSE Turtle Trading - Finding downtrend/breakdown signals.
Scans top 300 US stocks and identifies the best 20 candidates showing breakdown signals.

This is the opposite of turtle_trade.py:
- Entry: Price breaks BELOW 55-day low (downtrend breakout)
- Exit: Price rises ABOVE 20-day high (cover short)
- Stop: Entry + 2×ATR (stop loss above entry for shorts)

Author: ChatGPT
Requirements: pip install yfinance pandas numpy tqdm
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings('ignore')

# Import functions from turtle_trade_backtest.py
from turtle_trade_backtest import download_data, compute_atr

# Import ticker list function from turtle_trade.py
from turtle_trade import get_sp500_tickers, get_hardcoded_top_stocks, get_top_us_stocks

# ----- Inverse Breakout computation -----
def compute_breakdowns(df: pd.DataFrame, 
                       lookback_entry: int = 55,
                       lookback_exit: int = 20) -> pd.DataFrame:
    """
    Compute breakdown levels (inverse of breakouts).
    LL_entry: 55-day low (breakdown entry level)
    HH_exit: 20-day high (cover/exit level)
    """
    df = df.copy()
    df['LL_entry'] = df['Low'].rolling(window=lookback_entry).min()
    df['HH_exit'] = df['High'].rolling(window=lookback_exit).max()
    return df

# ----- Signal detection -----
def check_breakdown_signal(ticker: str, 
                           lookback_entry: int = 55,
                           lookback_exit: int = 20,
                           atr_window: int = 20,
                           min_price: float = 5.0,
                           min_volume: float = 1_000_000) -> Dict:
    """
    Check if a ticker shows turtle trading BREAKDOWN signal TODAY.
    Returns dict with signal info or None if no signal.
    
    Breakdown = Price breaks BELOW 55-day low (bearish signal for shorting)
    """
    try:
        # Download recent data (need extra days for lookback)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_entry + 30)
        
        df = download_data(ticker, 
                          start=start_date.strftime('%Y-%m-%d'),
                          end=end_date.strftime('%Y-%m-%d'))
        
        if df.empty or len(df) < lookback_entry + 1:
            return None
        
        # Filter out low-price/low-volume stocks
        latest = df.iloc[-1]
        if latest['Close'] < min_price or latest['Volume'] < min_volume:
            return None
        
        # Compute indicators
        df['ATR'] = compute_atr(df, atr_window, use_wilder=False)
        df = compute_breakdowns(df, lookback_entry, lookback_exit)
        
        # Check latest row for breakdown signal
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Breakdown signal: today's close < 55-day low
        has_breakdown_signal = latest['Close'] < latest['LL_entry']
        
        # Also check it's a NEW breakdown (wasn't triggered yesterday)
        is_new_breakdown = prev['Close'] >= prev['LL_entry']
        
        if has_breakdown_signal and is_new_breakdown:
            # Calculate position sizing info (for short positions)
            atr = latest['ATR']
            stop_price = latest['Close'] + 2.0 * atr  # Stop ABOVE entry for shorts
            exit_price = latest['HH_exit']  # 20-day high
            
            # Position sizing metrics
            risk_per_share = stop_price - latest['Close']  # Risk is upward movement
            current_exit_distance = exit_price - latest['Close']  # Distance to cover
            
            # Breakdown strength relative to risk
            breakdown_strength = latest['LL_entry'] - latest['Close']  # How far below low
            strength_risk_ratio = breakdown_strength / risk_per_share if risk_per_share > 0 else 0
            
            return {
                'ticker': ticker,
                'date': df.index[-1].strftime('%Y-%m-%d'),
                'close': latest['Close'],
                'entry_level': latest['LL_entry'],  # 55-day low
                'exit_level': latest['HH_exit'],  # 20-day high
                'stop_price': stop_price,
                'atr': atr,
                'volume': latest['Volume'],
                'breakdown_pct': ((latest['LL_entry'] - latest['Close']) / latest['LL_entry'] * 100),
                'risk_per_share': risk_per_share,
                'exit_distance': current_exit_distance,
                'breakdown_strength': breakdown_strength,
                'strength_risk_ratio': strength_risk_ratio,
            }
        
        return None
        
    except Exception as e:
        # Silently skip problematic tickers
        return None

def check_cover_signal(ticker: str,
                       entry_price: float,
                       lookback_exit: int = 20,
                       atr_window: int = 20,
                       stop_multiplier: float = 2.0) -> Dict:
    """
    For existing SHORT positions, check if cover/exit signal is triggered.
    Returns current price, exit level, stop level, and whether to cover.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_exit + 30)
        
        df = download_data(ticker,
                          start=start_date.strftime('%Y-%m-%d'),
                          end=end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            return None
        
        # Compute indicators
        df['ATR'] = compute_atr(df, atr_window, use_wilder=False)
        df = compute_breakdowns(df, 55, lookback_exit)
        
        latest = df.iloc[-1]
        atr = latest['ATR']
        stop_price = entry_price + stop_multiplier * atr  # Stop ABOVE for shorts
        
        # Check exit conditions (inverse of longs)
        exit_signal = latest['Close'] > latest['HH_exit']  # Price above 20-day high
        stop_hit = latest['High'] >= stop_price  # Price hit stop loss
        
        # P&L for short position (profit when price falls)
        pnl = entry_price - latest['Close']  # Inverse: entry - current
        pnl_pct = (pnl / entry_price) * 100
        
        return {
            'ticker': ticker,
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'entry_price': entry_price,
            'current_price': latest['Close'],
            'exit_level': latest['HH_exit'],  # 20-day high
            'stop_price': stop_price,
            'atr': atr,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_signal': exit_signal,
            'stop_hit': stop_hit,
            'action': 'COVER NOW' if (exit_signal or stop_hit) else 'HOLD SHORT',
        }
        
    except Exception as e:
        return None

# ----- Scanner functions -----
def scan_for_breakdowns(tickers: List[str],
                        lookback_entry: int = 55,
                        lookback_exit: int = 20,
                        atr_window: int = 20,
                        top_n: int = 20) -> pd.DataFrame:
    """
    Scan list of tickers for breakdown signals and return top N candidates.
    """
    print(f"\n{'='*80}")
    print(f"INVERSE TURTLE TRADING - BREAKDOWN SCANNER")
    print(f"{'='*80}")
    print(f"Scanning {len(tickers)} tickers for breakdown signals (shorts)...")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    signals = []
    
    for ticker in tqdm(tickers, desc="Scanning"):
        signal = check_breakdown_signal(ticker, lookback_entry, lookback_exit, atr_window)
        if signal:
            signals.append(signal)
    
    if not signals:
        print("\n❌ No breakdown signals found today.")
        return pd.DataFrame()
    
    # Convert to DataFrame and sort by strength/risk ratio (best signals first)
    df = pd.DataFrame(signals)
    df = df.sort_values('strength_risk_ratio', ascending=False)
    
    # Return top N
    return df.head(top_n)

def print_breakdown_signals(df: pd.DataFrame, capital: float | None = None, risk_pct: float = 0.01):
    """Pretty print breakdown signals. If capital is provided, also show position sizing.
    capital: total trading capital available (e.g., 20000)
    risk_pct: fraction of capital to risk per trade (e.g., 0.01 for 1%)
    """
    if df.empty:
        return
    
    print(f"\n{'='*80}")
    print(f"📉 BREAKDOWN SIGNALS DETECTED: {len(df)} stocks (SHORT CANDIDATES)")
    print(f"{'='*80}\n")
    
    for idx, row in df.iterrows():
        print(f"{'='*80}")
        print(f"📉 {row['ticker']} - SHORT SIGNAL")
        print(f"{'='*80}")
        print(f"  Date:              {row['date']}")
        print(f"  Current Price:     ${row['close']:.2f}")
        print(f"  Entry Level:       ${row['entry_level']:.2f} (55-day low)")
        print(f"  Breakdown Strength: {row['breakdown_pct']:.2f}% below entry level")
        print(f"  ")
        print(f"  📍 POSITION SIZING (SHORT):")
        print(f"  ATR (20-day):      ${row['atr']:.2f}")
        print(f"  Stop Loss:         ${row['stop_price']:.2f} (Entry + 2×ATR)")
        print(f"  Risk per share:    ${row['risk_per_share']:.2f}")
        
        # Optional: position sizing suggestions based on capital and risk
        if capital is not None and row['risk_per_share'] > 0 and row['close'] > 0:
            risk_budget = capital * max(min(risk_pct, 1.0), 0.0)
            shares_by_risk = math.floor(risk_budget / row['risk_per_share']) if risk_budget > 0 else 0
            shares_by_cap = math.floor(capital / row['close'])
            suggested_shares = min(shares_by_risk, shares_by_cap)
            suggested_shares = max(suggested_shares, 0)
            est_cost = suggested_shares * row['close']
            print(f"  -- With capital ${capital:,.0f} and risk {risk_pct*100:.1f}%:")
            print(f"     Shares by risk:     {shares_by_risk:,}")
            print(f"     Shares by capital:  {shares_by_cap:,}")
            print(f"     Suggested shares:   {suggested_shares:,} (~${est_cost:,.2f} margin req)")
        
        print(f"  ")
        print(f"  🎯 EXIT STRATEGY (COVER SHORT):")
        print(f"  Current 20-day high: ${row['exit_level']:.2f}")
        print(f"  Exit distance:       ${row['exit_distance']:.2f} (20-day high - current price)")
        print(f"  ⚠️  NOTE: The 20-day high will FALL as price falls (trailing cover)")
        print(f"      Cover when price rises ABOVE the 20-day high")
        print(f"  ")
        print(f"  📊 METRICS:")
        print(f"  Breakdown strength: ${row['breakdown_strength']:.2f}/share below entry")
        print(f"  Strength/Risk:     {row['strength_risk_ratio']:.2f}x (breakdown momentum vs risk)")
        print(f"  Volume:            {row['volume']:,.0f}")
        print()
    
    print(f"{'='*80}\n")

def monitor_short_positions(positions: List[Dict]) -> pd.DataFrame:
    """
    Monitor existing SHORT positions for cover signals.
    positions: list of dicts with {'ticker': str, 'entry_price': float, 'shares': int}
    """
    print(f"\n{'='*80}")
    print(f"SHORT POSITION MONITORING")
    print(f"{'='*80}")
    print(f"Checking {len(positions)} short positions...")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = []
    
    for pos in tqdm(positions, desc="Monitoring"):
        result = check_cover_signal(pos['ticker'], pos['entry_price'])
        if result:
            result['shares'] = pos.get('shares', 0)
            result['position_value'] = result['shares'] * result['current_price']
            result['total_pnl'] = result['shares'] * result['pnl']  # Already inverted in check_cover_signal
            results.append(result)
    
    if not results:
        print("\n❌ Could not retrieve data for positions.")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    return df

def print_short_position_status(df: pd.DataFrame):
    """Pretty print short position monitoring results."""
    if df.empty:
        return
    
    print(f"\n{'='*80}")
    print(f"📊 SHORT POSITION STATUS: {len(df)} positions")
    print(f"{'='*80}\n")
    
    total_pnl = df['total_pnl'].sum()
    
    for idx, row in df.iterrows():
        action_color = "🚨" if row['action'] == 'COVER NOW' else "✅"
        pnl_sign = "+" if row['pnl'] >= 0 else ""
        
        print(f"{'='*80}")
        print(f"{action_color} {row['ticker']} (SHORT) - {row['action']}")
        print(f"{'='*80}")
        print(f"  Entry Price:       ${row['entry_price']:.2f} (shorted)")
        print(f"  Current Price:     ${row['current_price']:.2f}")
        print(f"  Shares:            {row['shares']:,.0f}")
        print(f"  Position Value:    ${row['position_value']:,.2f}")
        print(f"  P&L:               {pnl_sign}${row['total_pnl']:,.2f} ({pnl_sign}{row['pnl_pct']:.2f}%)")
        print(f"  ")
        print(f"  Cover Level:       ${row['exit_level']:.2f} (20-day high)")
        print(f"  Stop Price:        ${row['stop_price']:.2f} (2× ATR above entry)")
        print(f"  ")
        if row['exit_signal']:
            print(f"  ⚠️  COVER SIGNAL: Price broke above 20-day high!")
        if row['stop_hit']:
            print(f"  🛑 STOP HIT: Price hit stop loss level!")
        print()
    
    print(f"{'='*80}")
    print(f"Total P&L: {'+' if total_pnl >= 0 else ''}${total_pnl:,.2f}")
    print(f"{'='*80}\n")

def save_signals_to_csv(df: pd.DataFrame, filename: str = None):
    """Save signals to CSV file for record keeping."""
    if df.empty:
        return
    
    if filename is None:
        filename = f"output/turtle_breakdown_signals_{datetime.now().strftime('%Y%m%d')}.csv"
    
    df.to_csv(filename, index=False)
    print(f"💾 Signals saved to: {filename}")

# ----- Main execution -----
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inverse Turtle Trading - Breakdown/Downtrend Scanner')
    parser.add_argument('--mode', type=str, default='scan', choices=['scan', 'monitor'],
                       help='Mode: scan for breakdowns or monitor short positions')
    parser.add_argument('--count', type=int, default=300,
                       help='Number of stocks to scan (default: 300)')
    parser.add_argument('--top', type=int, default=20,
                       help='Number of top signals to return (default: 20)')
    parser.add_argument('--save', action='store_true',
                       help='Save results to CSV file')
    parser.add_argument('--capital', type=float, default=20_000,
                        help='Total trading capital to size positions (e.g., 20000)')
    parser.add_argument('--risk', type=float, default=0.01,
                        help='Risk per trade as a fraction of capital (default: 0.01 = 1%)')
    
    args = parser.parse_args()
    
    if args.mode == 'scan':
        # Scan for breakdown signals
        tickers = get_top_us_stocks(args.count)
        
        if not tickers:
            print("❌ Could not retrieve ticker list.")
            exit(1)
        
        signals_df = scan_for_breakdowns(tickers, top_n=args.top)
        print_breakdown_signals(signals_df, capital=args.capital, risk_pct=args.risk)
        
        if args.save and not signals_df.empty:
            save_signals_to_csv(signals_df)
    
    elif args.mode == 'monitor':
        # Example: Monitor existing SHORT positions
        # YOU SHOULD MODIFY THIS with your actual short positions
        print("\n⚠️  EXAMPLE MODE: Update the 'positions' list in the code with your actual short holdings.\n")
        
        # Example short positions (MODIFY THIS with your actual shorts)
        positions = [
            {'ticker': 'EXAMPLE1', 'entry_price': 50.00, 'shares': 100},
            {'ticker': 'EXAMPLE2', 'entry_price': 75.00, 'shares': 50},
            # Add more short positions here
        ]
        
        if not positions:
            print("❌ No positions to monitor. Add short positions to the 'positions' list.")
        else:
            status_df = monitor_short_positions(positions)
            print_short_position_status(status_df)
            
            if args.save and not status_df.empty:
                filename = f"output/short_position_status_{datetime.now().strftime('%Y%m%d')}.csv"
                save_signals_to_csv(status_df, filename)
    
    print("\n✅ Done!\n")
    print("⚠️  DISCLAIMER: Short selling carries significant risk. Price can theoretically")
    print("   rise infinitely, leading to unlimited losses. Only trade with proper risk")
    print("   management and within your risk tolerance. This tool is for educational")
    print("   purposes only and does not constitute financial advice.\n")
