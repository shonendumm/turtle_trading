# turtle_trade_sgx.py
"""
Daily scanner for Turtle Trading entry/exit signals across top Singapore equities.
Scans top 200 SGX stocks and identifies the best 20 candidates showing entry signals.

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
from turtle_trade_backtest import download_data, compute_atr, compute_breakouts

# ----- Top Singapore stocks list -----
# Alternative data sources for SGX tickers (for future reference):
# 1. Wikipedia: https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_Singapore_Exchange
# 2. SGX Website: https://www.sgx.com/securities/stock-screener (requires web scraping)
# 3. SGX Data API: https://api2.sgx.com/... (unofficial, may change)
# 4. Financial data providers: Alpha Vantage, Financial Modeling Prep (may require API key)
#
# Current implementation uses Wikipedia with hardcoded fallback.

def get_hardcoded_sgx_stocks() -> List[str]:
    """
    Hardcoded list of major Singapore stocks (top ~200 by market cap).
    Yahoo Finance uses .SI suffix for SGX stocks.
    """
    return [
        # STI Components (Straits Times Index - Top 30)
        'D05.SI',   # DBS Group Holdings
        'O39.SI',   # OCBC Bank
        'U11.SI',   # United Overseas Bank (UOB)
        'Z74.SI',   # Singapore Telecommunications (Singtel)
        'C38U.SI',  # CapitaLand Integrated Commercial Trust
        'C31.SI',   # CapitaLand Investment
        'BN4.SI',   # Keppel Corporation
        'ME8U.SI',  # Mapletree Logistics Trust
        'N2IU.SI',  # Mapletree Industrial Trust
        'M44U.SI',  # Mapletree Pan Asia Commercial Trust
        'S68.SI',   # Singapore Exchange (SGX)
        'C52.SI',   # ComfortDelGro
        'G13.SI',   # Genting Singapore
        'Y92.SI',   # Thai Beverage
        'U96.SI',   # Sembcorp Industries
        'BS6.SI',   # YZJ Shipbldg SGD
        'V03.SI',   # Venture Corporation
        'S58.SI',   # SATS Ltd
        'A17U.SI',  # Ascendas REIT
        'J69U.SI',  # Frasers Logistics & Commercial Trust
        'C09.SI',   # City Developments
        'H78.SI',   # Hongkong Land Holdings
        'U14.SI',   # UOL Group
        'F34.SI',   # Wilmar International
        'S63.SI',   # Singapore Technologies Engineering
        'C6L.SI',   # Singapore Airlines
        'BN2.SI',   # Jardine Cycle & Carriage
        'C07.SI',   # Jardine Matheson Holdings
        'CC3.SI',   # StarHub
        'AWX.SI',   # AEM Holdings
        
        # Large Cap REITs
        'J91U.SI',  # ESR-LOGOS REIT
        'UD1U.SI',  # Keppel DC REIT
        'T82U.SI',  # Suntec REIT
        'K71U.SI',  # Keppel REIT
        'AJBU.SI',  # Keppel Infrastructure Trust
        'RW0U.SI',  # Frasers Centrepoint Trust
        'SK6U.SI',  # Sabana REIT
        'AU8U.SI',  # Frasers Hospitality Trust
        'M1GU.SI',  # Manulife US REIT
        'OU8U.SI',  # First REIT
        
        # Banking & Finance
        'CY6U.SI',  # CapitaLand Ascendas REIT
        'CGS.SI',   # Cogent Holdings
        'OV8.SI',   # Sheng Siong Group
        'P8Z.SI',   # PropNex
        'ERA.SI',   # ERA Singapore
        '5TT.SI',   # IREIT Global
        'P5S.SI',   # Hi-P International
        
        # Technology & Manufacturing
        'S51.SI',   # Sembcorp Marine
        'U10.SI',   # UMS Holdings
        'F9D.SI',   # Boustead Singapore
        'F86.SI',   # Frencken Group
        'H07.SI',   # ASTI Holdings
        'BOU.SI',   # Civmec
        'MZH.SI',   # Mermaid Maritime
        'BEX.SI',   # GSS Energy
        '43E.SI',   # Magnus Energy Group
        'B2F.SI',   # Serial System
        'B9A.SI',   # BBR Holdings
        
        # Healthcare & Pharmaceuticals
        'Q0F.SI',   # IHH Healthcare
        '1D1.SI',   # Raffles Medical Group
        'S07.SI',   # Singapore Medical Group
        'T24.SI',   # Healthway Medical Corporation
        
        # Consumer & Retail
        'F03.SI',   # Food Empire Holdings
        'N01.SI',   # Pacific Century Regional Developments
        'F13.SI',   # Food Junction Holdings
        'K2LU.SI',  # Dasin Retail Trust
        'P9D.SI',   # Delfi Limited
        '5GI.SI',   # Koufu Group
        'C41.SI',   # Cortina Holdings
        'B26.SI',   # OSIM International
        'B61.SI',   # Breadtalk Group
        '5AB.SI',   # OUE Lippo Healthcare
        
        # Hospitality & Leisure
        'H18.SI',   # Hotel Grand Central
        'H30.SI',   # Hong Fok Corporation
        'H15.SI',   # Hong Leong Finance
        'H13.SI',   # Hong Leong Asia
        'S51.SI',   # Straits Trading Company
        
        # Industrial & Engineering
        'S59.SI',   # Sitra Holdings
        'E5H.SI',   # Golden Agri-Resources
        'F25.SI',   # Bumitama Agri
        'Q01.SI',   # First Resources
        'S19.SI',   # Tuan Sing Holdings
        'E28.SI',   # Frencken Group
        '5CP.SI',   # AusGroup
        'B03.SI',   # CWT Limited
        
        # Property & Construction
        'TQ5.SI',   # Soilbuild Business Space REIT
        'A26.SI',   # Amara Holdings
        'P15.SI',   # CSE Global
        'U77.SI',   # Universal Robina Corporation
        'T13.SI',   # TEE International
        'AP4.SI',   # Riverstone Holdings
        '5TP.SI',   # Teckwah Industrial Corporation
        
        # Marine & Offshore
        'CGN.SI',   # CSE Global
        'P8A.SI',   # PACC Offshore Services Holdings
        'S20.SI',   # Swissco Holdings
        'BMD.SI',   # Vibrant Group
        '5UF.SI',   # Baker Technology
        '5GD.SI',   # Seroja Investments
        
        # Electronics & Components
        'CKP.SI',   # Creative Technology
        'AVM.SI',   # Aztech Global
        'AWN.SI',   # Advanced Systems Automation
        'W05.SI',   # Willas-Array Electronics
        'P34.SI',   # Valuetronics Holdings
        'S08.SI',   # Singamas Container Holdings
        
        # Transportation & Logistics
        'C29.SI',   # Cosco Shipping International
        'S56.SI',   # Samudera Shipping Line
        'S19.SI',   # Seatown Holdings International
        
        # Additional Large & Mid Caps
        'OUE.SI',   # OUE Limited
        'AJBU.SI',  # Keppel Infrastructure Trust
        'ME8U.SI',  # Mapletree Logistics Trust
        'MJI.SI',   # MTQ Corporation
        '1F3.SI',   # ISR Capital
        'BNE.SI',   # Jumbo Group
        '5JS.SI',   # Kitchen Culture Holdings
        '5WJ.SI',   # PEC Ltd
        '500.SI',   # LEGG Mason Western Asset Managed Municipals Fund Inc.
        'R14.SI',   # Riverstone Holdings
        
        # Small/Mid Cap Growth
        'T6I.SI',   # AVI-Tech Electronics
        '1F1.SI',   # mm2 Asia
        '5GG.SI',   # Civmec Limited
        'D8DU.SI',  # Dasin Retail Trust
        'CHJ.SI',   # CSE Global Limited
        'TCJ.SI',   # Top Glove Corporation
        '568.SI',   # Yanlord Land Group
        '5HH.SI',   # Hanwell Holdings
        'Q5T.SI',   # Qian Hu Corporation
        'G92.SI',   # Chip Eng Seng Corporation
        'S41.SI',   # Singapore Shipping Corporation
        'AFG.SI',   # AusGroup Limited
        '5GJ.SI',   # Nordic Group Limited
        '43F.SI',   # Rotary Engineering
        'BDA.SI',   # Serial System Ltd
        
        # Others
        'F117.SI',  # Singapura Finance
        'G3B.SI',   # Chip Eng Seng Corporation
        'BCV.SI',   # Mercurius Capital
        'M04.SI',   # Mewah International
        'S7P.SI',   # ISDN Holdings
        'T19.SI',   # Tee Land Limited
        'BEW.SI',   # Sin Ghee Huat Corporation
        '544.SI',   # Hwa Hong Corporation
        '5OT.SI',   # Memtech International
        'H02.SI',   # Hatten Land
        'U09.SI',   # Yoma Strategic Holdings
        'T09.SI',   # Thomson Medical Group
        'S85.SI',   # Sheng Siong Group
        'QES.SI',   # Q&M Dental Group
        'F17.SI',   # Food Empire Holdings
        'S69.SI',   # Singapore O&G
        'EB5.SI',   # ECS Holdings
        'T43.SI',   # TEE Land Limited
        'NO4.SI',   # Oxley Holdings
        'TDG.SI',   # Tiong Seng Holdings
    ]

def get_sgx_tickers() -> List[str]:
    """
    Get SGX ticker list from stockanalysis.com with fallback to hardcoded list.
    """
    try:
        # Try stockanalysis.com first
        import urllib.request
        url = 'https://stockanalysis.com/list/singapore-exchange/'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        tables = pd.read_html(urllib.request.urlopen(req))
        
        # The page should have a table with a "Symbol" column
        tickers = []
        for table in tables:
            if 'Symbol' in table.columns:
                raw_tickers = table['Symbol'].tolist()
                # Add .SI suffix and clean
                for ticker in raw_tickers:
                    if pd.notna(ticker) and str(ticker).strip():
                        # Convert to string and clean
                        ticker_str = str(ticker).strip()
                        # Remove any existing .SI suffix to avoid duplicates
                        ticker_str = ticker_str.replace('.SI', '')
                        # Add .SI suffix for Yahoo Finance
                        tickers.append(f"{ticker_str}.SI")
                break  # Found the right table, stop searching
        
        if tickers:
            print(f"✅ Fetched {len(tickers)} SGX tickers from stockanalysis.com")
            return tickers
        else:
            raise ValueError("No tickers found in Symbol column")
            
    except Exception as e:
        print(f"⚠️  Could not fetch SGX list from stockanalysis.com: {e}")
        print(f"📋 Using hardcoded list of top Singapore stocks instead...")
        return get_hardcoded_sgx_stocks()

def get_sgx_from_yahoo_screener() -> List[str]:
    """
    Alternative method: Get SGX stocks by querying Yahoo Finance.
    This is a fallback method if Wikipedia fails.
    Note: This method is less reliable and may be slow.
    """
    try:
        # Yahoo Finance doesn't have a direct API for listing all SGX stocks
        # But we can try to get them from a known index
        import yfinance as yf
        
        # Try to get STI (Straits Times Index) components
        sti = yf.Ticker("^STI")
        # This might not work as Yahoo doesn't always provide constituents
        
        print("⚠️  Yahoo Finance doesn't provide a comprehensive SGX stock list API")
        return get_hardcoded_sgx_stocks()
        
    except Exception as e:
        print(f"⚠️  Yahoo Finance method failed: {e}")
        return get_hardcoded_sgx_stocks()

def get_top_sgx_stocks(count: int = 200) -> List[str]:
    """
    Get top SGX stocks.
    """
    sgx_stocks = get_sgx_tickers()
    
    if not sgx_stocks:
        print("❌ Could not retrieve any ticker list.")
        return []
    
    return sgx_stocks[:min(count, len(sgx_stocks))]

# ----- Signal detection -----
def check_entry_signal(ticker: str, 
                       lookback_entry: int = 55,
                       lookback_exit: int = 20,
                       atr_window: int = 20,
                       min_price: float = 0.50,  # SGD 0.50 minimum (lower than US)
                       min_volume: float = 100_000) -> Dict:  # Lower volume threshold for SGX
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
            risk_per_share = latest['Close'] - stop_price
            current_exit_distance = latest['Close'] - exit_price
            
            # Breakout strength relative to risk
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
    print(f"TURTLE TRADING ENTRY SCANNER - SINGAPORE STOCKS")
    print(f"{'='*80}")
    print(f"Scanning {len(tickers)} SGX tickers for entry signals...")
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

def print_entry_signals(df: pd.DataFrame, capital: float | None = None, risk_pct: float = 0.01):
    """Pretty print entry signals. If capital is provided, also show position sizing.
    capital: total trading capital available (e.g., 20000 SGD)
    risk_pct: fraction of capital to risk per trade (e.g., 0.01 for 1%)
    """
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
        print(f"  Current Price:     SGD ${row['close']:.3f}")
        print(f"  Entry Level:       SGD ${row['entry_level']:.3f} (55-day high)")
        print(f"  Breakout Strength: {row['breakout_pct']:.2f}% above entry level")
        print(f"  ")
        print(f"  📍 POSITION SIZING:")
        print(f"  ATR (20-day):      SGD ${row['atr']:.3f}")
        print(f"  Stop Loss:         SGD ${row['stop_price']:.3f} (Entry - 2×ATR)")
        print(f"  Risk per share:    SGD ${row['risk_per_share']:.3f}")
        
        # Optional: position sizing suggestions based on capital and risk
        if capital is not None and row['risk_per_share'] > 0 and row['close'] > 0:
            risk_budget = capital * max(min(risk_pct, 1.0), 0.0)
            shares_by_risk = math.floor(risk_budget / row['risk_per_share']) if risk_budget > 0 else 0
            shares_by_cap = math.floor(capital / row['close'])
            suggested_shares = min(shares_by_risk, shares_by_cap)
            suggested_shares = max(suggested_shares, 0)
            est_cost = suggested_shares * row['close']
            print(f"  -- With capital SGD ${capital:,.0f} and risk {risk_pct*100:.1f}%:")
            print(f"     Shares by risk:     {shares_by_risk:,}")
            print(f"     Shares by capital:  {shares_by_cap:,}")
            print(f"     Suggested shares:   {suggested_shares:,} (~SGD ${est_cost:,.2f})")
        
        print(f"  ")
        print(f"  🎯 EXIT STRATEGY:")
        print(f"  Current 20-day low: SGD ${row['exit_level']:.3f}")
        print(f"  Exit cushion:       SGD ${row['exit_distance']:.3f} (current price - 20-day low)")
        print(f"  ⚠️  NOTE: The 20-day low will RISE as price rises (trailing exit)")
        print(f"      Exit when price falls BELOW the 20-day low")
        print(f"  ")
        print(f"  📊 METRICS:")
        print(f"  Breakout strength: SGD ${row['breakout_strength']:.3f}/share above entry")
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
    print(f"POSITION MONITORING - SINGAPORE STOCKS")
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
        print(f"  Entry Price:       SGD ${row['entry_price']:.3f}")
        print(f"  Current Price:     SGD ${row['current_price']:.3f}")
        print(f"  Shares:            {row['shares']:,.0f}")
        print(f"  Position Value:    SGD ${row['position_value']:,.2f}")
        print(f"  P&L:               {pnl_sign}SGD ${row['total_pnl']:,.2f} ({pnl_sign}{row['pnl_pct']:.2f}%)")
        print(f"  ")
        print(f"  Exit Level:        SGD ${row['exit_level']:.3f} (20-day low)")
        print(f"  Stop Price:        SGD ${row['stop_price']:.3f} (2× ATR)")
        print(f"  ")
        if row['exit_signal']:
            print(f"  ⚠️  EXIT SIGNAL: Price broke below 20-day low!")
        if row['stop_hit']:
            print(f"  🛑 STOP HIT: Price hit stop loss level!")
        print()
    
    print(f"{'='*80}")
    print(f"Total P&L: {'+' if total_pnl >= 0 else ''}SGD ${total_pnl:,.2f}")
    print(f"{'='*80}\n")

def save_signals_to_csv(df: pd.DataFrame, filename: str = None):
    """Save signals to CSV file for record keeping."""
    if df.empty:
        return
    
    if filename is None:
        filename = f"output/turtle_signals_sgx_{datetime.now().strftime('%Y%m%d')}.csv"
    
    df.to_csv(filename, index=False)
    print(f"💾 Signals saved to: {filename}")

# ----- Main execution -----
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Turtle Trading Position Scanner - Singapore Stocks')
    parser.add_argument('--mode', type=str, default='scan', choices=['scan', 'monitor'],
                       help='Mode: scan for entries or monitor existing positions')
    parser.add_argument('--count', type=int, default=300,
                       help='Number of SGX stocks to scan (default: 300)')
    parser.add_argument('--top', type=int, default=20,
                       help='Number of top signals to return (default: 20)')
    parser.add_argument('--save', action='store_true',
                       help='Save results to CSV file')
    parser.add_argument('--capital', type=float, default=20_000,
                        help='Total trading capital in SGD to size positions (e.g., 20000)')
    parser.add_argument('--risk', type=float, default=0.01,
                        help='Risk per trade as a fraction of capital (default: 0.01 = 1%)')
    
    args = parser.parse_args()
    
    if args.mode == 'scan':
        # Scan for entry signals
        tickers = get_top_sgx_stocks(args.count)
        
        if not tickers:
            print("❌ Could not retrieve ticker list.")
            exit(1)
        
        signals_df = scan_for_entries(tickers, top_n=args.top)
        print_entry_signals(signals_df, capital=args.capital, risk_pct=args.risk)
        
        if args.save and not signals_df.empty:
            save_signals_to_csv(signals_df)
    
    elif args.mode == 'monitor':
        # Example: Monitor existing positions
        # YOU SHOULD MODIFY THIS with your actual SGX positions
        print("\n⚠️  EXAMPLE MODE: Update the 'positions' list in the code with your actual holdings.\n")
        
        # Example positions (MODIFY THIS with your actual SGX positions)
        positions = [
            {'ticker': 'QC7.SI', 'entry_price': 0.505, 'shares': 1000},   # DBS
            {'ticker': 'G92.SI', 'entry_price': 1.49, 'shares': 1000},   # China Aviation Oil
            # Add more positions here
        ]
        
        if not positions:
            print("❌ No positions to monitor. Add positions to the 'positions' list.")
        else:
            status_df = monitor_positions(positions)
            print_position_status(status_df)
            
            if args.save and not status_df.empty:
                filename = f"output/position_status_sgx_{datetime.now().strftime('%Y%m%d')}.csv"
                save_signals_to_csv(status_df, filename)
    
    print("\n✅ Done!\n")
