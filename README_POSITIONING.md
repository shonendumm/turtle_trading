# Turtle Trade Positioning Scanner

Daily scanner to identify turtle trading entry and exit signals across top US equities.

## Installation

First, install the required package (tqdm for progress bars):

```bash
pip install tqdm
```

## Usage

### 1. Scan for Entry Signals (Daily)

Run this every day to find new entry opportunities:

```bash
python turtle_trade_positioning.py --mode scan --top 20 --save --capital 20000 --risk 0.01
```

**Parameters:**
- `--mode scan`: Scan for entry signals
- `--count 300`: Scan top 300 S&P 500 stocks
- `--top 20`: Return top 20 best signals
- `--save`: Optional, saves results to CSV file
- `--capital`: overall capital you are using for trading
- `--risk`: percentage risk, amount loss/entry price, default is 0.01 (1%), conservative use 0.005

**Output:** You'll get a list of 20 stocks showing:
- Current price and entry breakout level
- Stop loss price (Entry - 2×ATR)
- Exit target (20-day low)
- Risk per share and potential reward
- Volume and breakout strength

### 2. Monitor Existing Positions

Check your current positions for exit signals:

```bash
python turtle_trade.py --mode monitor --save
```

**Before running**, edit the `positions` list in the code (around line 340):

```python
positions = [
    {'ticker': 'AAPL', 'entry_price': 150.00, 'shares': 100},
    {'ticker': 'MSFT', 'entry_price': 330.00, 'shares': 50},
    # Add your positions here
]
```

**Output:** For each position, you'll see:
- Current P&L ($ and %)
- Exit level and stop loss price
- 🚨 **EXIT NOW** alert if exit triggered
- 🛑 **STOP HIT** alert if stop loss triggered

## How to Use Daily

### Morning Routine (Before Market Open):
1. Run the scanner to find entry signals
2. Review the top 20 candidates
3. Select stocks with:
   - Good risk/reward ratio (>1.5)
   - Strong breakout (>0.5% above entry level)
   - High volume confirmation

### During Trading Day:
- Enter positions at market price if breakout confirmed
- Set stop loss at: Entry Price - 2×ATR
- Note the exit level (20-day low)

### End of Day:
1. Run position monitor to check your holdings
2. Exit positions that trigger:
   - Exit signal (close below 20-day low)
   - Stop loss hit (close below stop price)

## Example Workflow

```bash
# Monday morning: Find new opportunities
python turtle_trade.py --mode scan --top 20 --save

# Review output, select 3 stocks to enter
# Enter trades during the day

# Monday evening: Check positions
python turtle_trade.py --mode monitor

# Repeat daily
```

## Position Sizing

For each entry signal, the scanner shows:
- **Risk per share**: Entry - Stop = Distance to stop loss
- **To size your position**: 
  - Decide risk per trade (e.g., 1% of account = $1,000)
  - Shares to buy = Risk Amount / Risk per share
  - Example: $1,000 / $5 risk = 200 shares

## Files Generated

When using `--save` flag:
- `output/turtle_signals_YYYYMMDD.csv`: Entry signals
- `output/position_status_YYYYMMDD.csv`: Position monitoring results

## Notes

- The scanner fetches live data from Yahoo Finance
- Scanning 300 stocks takes ~5-10 minutes
- Only stocks with price > $5 and volume > 1M are considered
- Signals are based on closing prices, not intraday data
- Always verify signals before trading!

## Customization

Edit these parameters in the code:
- `lookback_entry`: Default 55 days (classic turtle)
- `lookback_exit`: Default 20 days (classic turtle)
- `atr_window`: Default 20 days
- `stop_multiplier`: Default 2.0 (2× ATR stop)
- `min_price`: Default $5 (minimum stock price)
- `min_volume`: Default 1M shares (minimum liquidity)
