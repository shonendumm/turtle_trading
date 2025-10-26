# turtle_trade_positioning.py
"""
Daily scanner for Turtle Trading entry/exit signals across top US equities.
Scans top 300 US stocks and identifies the best 20 candidates showing entry signals.

Author: ChatGPT
Requirements: pip install yfinance pandas numpy tqdm
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import functions from turtle_trade.py
from turtle_trade import download_data, compute_atr, compute_breakouts

# ----- Top US stocks list -----
def get_hardcoded_top_stocks() -> List[str]:
    """
    Hardcoded list of major US stocks (top ~300 by market cap).
    This is a fallback when Wikipedia scraping fails.
    """
    return [
        # Mega cap tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
        # Large cap tech
        'CRM', 'CSCO', 'ACN', 'AMD', 'IBM', 'INTC', 'QCOM', 'TXN', 'INTU', 'NOW',
        'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'FTNT', 'PANW',
        # Financials
        'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SPGI', 'BLK',
        'C', 'SCHW', 'AXP', 'PGR', 'CB', 'MMC', 'ICE', 'CME', 'MCO', 'AON',
        'PNC', 'USB', 'TFC', 'COF', 'AIG', 'MET', 'PRU', 'AFL', 'ALL', 'TRV',
        # Healthcare
        'UNH', 'LLY', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY',
        'AMGN', 'GILD', 'CVS', 'MDT', 'CI', 'ISRG', 'VRTX', 'REGN', 'ELV', 'HUM',
        'BSX', 'ZTS', 'SYK', 'BDX', 'EW', 'A', 'IQV', 'IDXX', 'RMD', 'DXCM',
        # Consumer
        'WMT', 'HD', 'PG', 'COST', 'KO', 'PEP', 'MCD', 'NKE', 'TM', 'DIS',
        'CMCSA', 'NFLX', 'SBUX', 'LOW', 'TGT', 'TJX', 'BKNG', 'MAR', 'ABNB', 'CMG',
        'PM', 'MO', 'MDLZ', 'GIS', 'HSY', 'KHC', 'CAG', 'CPB', 'SJM', 'K',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HES',
        'WMB', 'KMI', 'HAL', 'BKR', 'FANG', 'DVN', 'MRO', 'APA', 'OVV', 'CTRA',
        # Industrials
        'BA', 'HON', 'UNP', 'RTX', 'CAT', 'GE', 'LMT', 'DE', 'UPS', 'ADP',
        'ITW', 'MMM', 'GD', 'NOC', 'TDG', 'ETN', 'EMR', 'PH', 'PCAR', 'CMI',
        'FDX', 'NSC', 'CSX', 'WM', 'RSG', 'CARR', 'OTIS', 'IR', 'FAST', 'PAYX',
        # Materials
        'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'DOW', 'PPG', 'NUE',
        'APH', 'VMC', 'MLM', 'BALL', 'AVY', 'CF', 'MOS', 'FMC', 'ALB', 'CE',
        # Communication Services
        'GOOG', 'T', 'VZ', 'TMUS', 'CHTR', 'EA', 'TTWO', 'NTES', 'MTCH', 'PARA',
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PCG', 'ED',
        # Real Estate
        'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'O', 'WELL', 'DLR', 'SBAC', 'SPG',
        # Additional tech & growth
        'SHOP', 'SQ', 'PYPL', 'UBER', 'LYFT', 'ROKU', 'SNOW', 'CRWD', 'ZS', 'NET',
        'DDOG', 'MDB', 'TEAM', 'WDAY', 'ZM', 'DOCU', 'TWLO', 'OKTA', 'SPLK', 'ESTC',
        # Biotech
        'BIIB', 'MRNA', 'ALNY', 'SGEN', 'NBIX', 'EXAS', 'INCY', 'TECH', 'UTHR', 'BMRN',
        # Semiconductors
        'ASML', 'TSM', 'AVGO', 'TER', 'MPWR', 'NXPI', 'STM', 'ON', 'SWKS', 'QRVO',
        # Financial services
        'FIS', 'FISV', 'PYPL', 'SQ', 'COIN', 'MSTR', 'HOOD', 'SOFI', 'NU', 'AFRM',
        # E-commerce & retail
        'EBAY', 'ETSY', 'W', 'CHWY', 'CVNA', 'KSS', 'M', 'JWN', 'DKS', 'BBY',
        # Auto
        'F', 'GM', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'STLA', 'HMC', 'RACE',
        # Aerospace & Defense
        'LHX', 'HWM', 'AXON', 'TXT', 'HII', 'LMT', 'NOC', 'GD', 'RTX', 'BA',
        # More diversified
        'BHP', 'RIO', 'VALE', 'SCCO', 'AA', 'X', 'CLF', 'MT', 'TX', 'STLD',
        'ADM', 'BG', 'TSN', 'HRL', 'INGR', 'DAR', 'SMG', 'FMC', 'MOS', 'NTR',
        'LUV', 'DAL', 'UAL', 'AAL', 'JBLU', 'SAVE', 'ALGT', 'HA', 'SKYW', 'MESA',
    ]

def get_sp500_tickers() -> List[str]:
    """Get S&P 500 ticker list from Wikipedia with proper headers."""
    try:
        # Add headers to avoid 403 error
        import urllib.request
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        tables = pd.read_html(urllib.request.urlopen(req))
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        # Clean tickers (some may have special characters)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers
    except Exception as e:
        print(f"⚠️  Could not fetch S&P 500 from Wikipedia: {e}")
        print(f"📋 Using hardcoded list of top US stocks instead...")
        return get_hardcoded_top_stocks()

def get_top_us_stocks(count: int = 300) -> List[str]:
    """
    Get top US stocks. Uses S&P 500 as base, with hardcoded fallback.
    You can modify this to use other sources or custom lists.
    """
    sp500 = get_sp500_tickers()
    
    if not sp500:
        print("❌ Could not retrieve any ticker list.")
        return []
    
    return sp500[:min(count, len(sp500))]

# ----- Signal detection -----
def check_entry_signal(ticker: str, 
                       lookback_entry: int = 55,
                       lookback_exit: int = 20,
                       atr_window: int = 20,
                       min_price: float = 5.0,
                       min_volume: float = 1_000_000) -> Dict:
    """
    Check if a ticker shows turtle trading entry signal TODAY.
    Returns dict with signal info or None if no signal.
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
        df = compute_breakouts(df, lookback_entry, lookback_exit)
        
        # Check latest row for entry signal
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Entry signal: today's close > 55-day high
        has_entry_signal = latest['Close'] > latest['HH_entry']
        
        # Also check it's a NEW breakout (wasn't triggered yesterday)
        is_new_breakout = prev['Close'] <= prev['HH_entry']
        
        if has_entry_signal and is_new_breakout:
            # Calculate position sizing info
            atr = latest['ATR']
            stop_price = latest['Close'] - 2.0 * atr
            exit_price = latest['LL_exit']
            
            # Position sizing metrics
            # Note: The exit level (20-day low) will RISE as price rises (trailing stop)
            # So the current exit level is just informational - it's not a profit target
            risk_per_share = latest['Close'] - stop_price
            current_exit_distance = latest['Close'] - exit_price
            
            # Breakout strength relative to risk (not traditional risk/reward since no fixed target)
            breakout_strength = latest['Close'] - latest['HH_entry']
            strength_risk_ratio = breakout_strength / risk_per_share if risk_per_share > 0 else 0
            
            return {
                'ticker': ticker,
                'date': df.index[-1].strftime('%Y-%m-%d'),
                'close': latest['Close'],
                'entry_level': latest['HH_entry'],
                'exit_level': latest['LL_exit'],
                'stop_price': stop_price,
                'atr': atr,
                'volume': latest['Volume'],
                'breakout_pct': ((latest['Close'] - latest['HH_entry']) / latest['HH_entry'] * 100),
                'risk_per_share': risk_per_share,
                'exit_distance': current_exit_distance,
                'breakout_strength': breakout_strength,
                'strength_risk_ratio': strength_risk_ratio,
            }
        
        return None
        
    except Exception as e:
        # Silently skip problematic tickers
        return None

def check_exit_signal(ticker: str,
                      entry_price: float,
                      lookback_exit: int = 20,
                      atr_window: int = 20,
                      stop_multiplier: float = 2.0) -> Dict:
    """
    For existing positions, check if exit signal is triggered.
    Returns current price, exit level, stop level, and whether to exit.
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
        df = compute_breakouts(df, 55, lookback_exit)
        
        latest = df.iloc[-1]
        atr = latest['ATR']
        stop_price = entry_price - stop_multiplier * atr
        
        # Check exit conditions
        exit_signal = latest['Close'] < latest['LL_exit']
        stop_hit = latest['Low'] <= stop_price
        
        pnl = latest['Close'] - entry_price
        pnl_pct = (pnl / entry_price) * 100
        
        return {
            'ticker': ticker,
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'entry_price': entry_price,
            'current_price': latest['Close'],
            'exit_level': latest['LL_exit'],
            'stop_price': stop_price,
            'atr': atr,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_signal': exit_signal,
            'stop_hit': stop_hit,
            'action': 'EXIT NOW' if (exit_signal or stop_hit) else 'HOLD',
        }
        
    except Exception as e:
        return None

# ----- Scanner functions -----
def scan_for_entries(tickers: List[str],
                     lookback_entry: int = 55,
                     lookback_exit: int = 20,
                     atr_window: int = 20,
                     top_n: int = 20) -> pd.DataFrame:
    """
    Scan list of tickers for entry signals and return top N candidates.
    """
    print(f"\n{'='*80}")
    print(f"TURTLE TRADING ENTRY SCANNER")
    print(f"{'='*80}")
    print(f"Scanning {len(tickers)} tickers for entry signals...")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    signals = []
    
    for ticker in tqdm(tickers, desc="Scanning"):
        signal = check_entry_signal(ticker, lookback_entry, lookback_exit, atr_window)
        if signal:
            signals.append(signal)
    
    if not signals:
        print("\n❌ No entry signals found today.")
        return pd.DataFrame()
    
    # Convert to DataFrame and sort by strength/risk ratio (best signals first)
    df = pd.DataFrame(signals)
    df = df.sort_values('strength_risk_ratio', ascending=False)
    
    # Return top N
    return df.head(top_n)

def print_entry_signals(df: pd.DataFrame):
    """Pretty print entry signals."""
    if df.empty:
        return
    
    print(f"\n{'='*80}")
    print(f"🎯 ENTRY SIGNALS DETECTED: {len(df)} stocks")
    print(f"{'='*80}\n")
    
    for idx, row in df.iterrows():
        print(f"{'='*80}")
        print(f"📈 {row['ticker']}")
        print(f"{'='*80}")
        print(f"  Date:              {row['date']}")
        print(f"  Current Price:     ${row['close']:.2f}")
        print(f"  Entry Level:       ${row['entry_level']:.2f} (55-day high)")
        print(f"  Breakout Strength: {row['breakout_pct']:.2f}% above entry level")
        print(f"  ")
        print(f"  📍 POSITION SIZING:")
        print(f"  ATR (20-day):      ${row['atr']:.2f}")
        print(f"  Stop Loss:         ${row['stop_price']:.2f} (Entry - 2×ATR)")
        print(f"  Risk per share:    ${row['risk_per_share']:.2f}")
        print(f"  ")
        print(f"  🎯 EXIT STRATEGY:")
        print(f"  Current 20-day low: ${row['exit_level']:.2f}")
        print(f"  Exit cushion:       ${row['exit_distance']:.2f} (current price - 20-day low)")
        print(f"  ⚠️  NOTE: The 20-day low will RISE as price rises (trailing exit)")
        print(f"      Exit when price falls BELOW the 20-day low")
        print(f"  ")
        print(f"  📊 METRICS:")
        print(f"  Breakout strength: ${row['breakout_strength']:.2f}/share above entry")
        print(f"  Strength/Risk:     {row['strength_risk_ratio']:.2f}x (breakout momentum vs risk)")
        print(f"  Volume:            {row['volume']:,.0f}")
        print()
    
    print(f"{'='*80}\n")

def monitor_positions(positions: List[Dict]) -> pd.DataFrame:
    """
    Monitor existing positions for exit signals.
    positions: list of dicts with {'ticker': str, 'entry_price': float, 'shares': int}
    """
    print(f"\n{'='*80}")
    print(f"POSITION MONITORING")
    print(f"{'='*80}")
    print(f"Checking {len(positions)} positions...")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = []
    
    for pos in tqdm(positions, desc="Monitoring"):
        result = check_exit_signal(pos['ticker'], pos['entry_price'])
        if result:
            result['shares'] = pos.get('shares', 0)
            result['position_value'] = result['shares'] * result['current_price']
            result['total_pnl'] = result['shares'] * result['pnl']
            results.append(result)
    
    if not results:
        print("\n❌ Could not retrieve data for positions.")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    return df

def print_position_status(df: pd.DataFrame):
    """Pretty print position monitoring results."""
    if df.empty:
        return
    
    print(f"\n{'='*80}")
    print(f"📊 POSITION STATUS: {len(df)} positions")
    print(f"{'='*80}\n")
    
    total_pnl = df['total_pnl'].sum()
    
    for idx, row in df.iterrows():
        action_color = "🚨" if row['action'] == 'EXIT NOW' else "✅"
        pnl_sign = "+" if row['pnl'] >= 0 else ""
        
        print(f"{'='*80}")
        print(f"{action_color} {row['ticker']} - {row['action']}")
        print(f"{'='*80}")
        print(f"  Entry Price:       ${row['entry_price']:.2f}")
        print(f"  Current Price:     ${row['current_price']:.2f}")
        print(f"  Shares:            {row['shares']:,.0f}")
        print(f"  Position Value:    ${row['position_value']:,.2f}")
        print(f"  P&L:               {pnl_sign}${row['total_pnl']:,.2f} ({pnl_sign}{row['pnl_pct']:.2f}%)")
        print(f"  ")
        print(f"  Exit Level:        ${row['exit_level']:.2f} (20-day low)")
        print(f"  Stop Price:        ${row['stop_price']:.2f} (2× ATR)")
        print(f"  ")
        if row['exit_signal']:
            print(f"  ⚠️  EXIT SIGNAL: Price broke below 20-day low!")
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
        filename = f"output/turtle_signals_{datetime.now().strftime('%Y%m%d')}.csv"
    
    df.to_csv(filename, index=False)
    print(f"💾 Signals saved to: {filename}")

# ----- Main execution -----
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Turtle Trading Position Scanner')
    parser.add_argument('--mode', type=str, default='scan', choices=['scan', 'monitor'],
                       help='Mode: scan for entries or monitor existing positions')
    parser.add_argument('--count', type=int, default=300,
                       help='Number of stocks to scan (default: 300)')
    parser.add_argument('--top', type=int, default=20,
                       help='Number of top signals to return (default: 20)')
    parser.add_argument('--save', action='store_true',
                       help='Save results to CSV file')
    
    args = parser.parse_args()
    
    if args.mode == 'scan':
        # Scan for entry signals
        tickers = get_top_us_stocks(args.count)
        
        if not tickers:
            print("❌ Could not retrieve ticker list.")
            exit(1)
        
        signals_df = scan_for_entries(tickers, top_n=args.top)
        print_entry_signals(signals_df)
        
        if args.save and not signals_df.empty:
            save_signals_to_csv(signals_df)
    
    elif args.mode == 'monitor':
        # Example: Monitor existing positions
        # YOU SHOULD MODIFY THIS with your actual positions
        print("\n⚠️  EXAMPLE MODE: Update the 'positions' list in the code with your actual holdings.\n")
        
        # Example positions (MODIFY THIS with your actual positions)
        positions = [
            {'ticker': 'AAPL', 'entry_price': 150.00, 'shares': 100},
            {'ticker': 'MSFT', 'entry_price': 330.00, 'shares': 50},
            # Add more positions here
        ]
        
        if not positions:
            print("❌ No positions to monitor. Add positions to the 'positions' list.")
        else:
            status_df = monitor_positions(positions)
            print_position_status(status_df)
            
            if args.save and not status_df.empty:
                filename = f"output/position_status_{datetime.now().strftime('%Y%m%d')}.csv"
                save_signals_to_csv(status_df, filename)
    
    print("\n✅ Done!\n")
