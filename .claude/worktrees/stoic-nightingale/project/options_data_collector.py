#!/usr/bin/env python3
"""
OPTIONS DATA COLLECTOR - SEQUENTIAL VERSION
Collects options chain data sequentially (12-15 min for 6000 symbols)

Use this if:
- Parallel version has issues
- You want to monitor each symbol individually
- Your system has limited memory

Usage:
    python options_data_collector.py --symbols 6000

For faster collection, use:
    python options_data_collector_parallel.py --workers 8
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)


class OptionsCollectorSequential:
    """Sequential options data collector"""
    
    def __init__(self, output_dir: str = "project/data/features"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {'collected': 0, 'failed': 0}
    
    def get_us_stocks(self, count: int = 6000) -> List[str]:
        """Get US stock symbols"""
        base_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B',
            'JNJ', 'JPM', 'V', 'WMT', 'PG', 'UNH', 'MA', 'HD', 'MRK', 'ABBV',
        ]
        
        # Extend with common tickers
        extended = base_stocks + [
            'XOM', 'PFE', 'CVX', 'KO', 'INTC', 'CSCO', 'VZ', 'PEP', 'MCD',
            'LLY', 'PM', 'AMD', 'ABT', 'T', 'BA', 'COST', 'NFLX', 'NKE', 'DIS'
        ]
        
        return list(dict.fromkeys(extended))[:count]
    
    def collect_options_data(self, symbol: str) -> Optional[Dict]:
        """Collect options for single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                return None
            
            chain = ticker.option_chain(expirations[0])
            calls, puts = chain.calls, chain.puts
            
            if calls.empty or puts.empty:
                return None
            
            data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'put_call_ratio': puts['volume'].sum() / max(calls['volume'].sum(), 1),
                'iv_rank': calls['impliedVolatility'].mean(),
                'open_interest_call': calls['openInterest'].sum(),
                'open_interest_put': puts['openInterest'].sum(),
                'volume_call': calls['volume'].sum(),
                'volume_put': puts['volume'].sum(),
            }
            
            return data
            
        except Exception:
            return None
    
    def collect_batch(self, symbols: List[str]):
        """Sequential collection"""
        print(f"\n📊 OPTIONS DATA (SEQUENTIAL)")
        print(f"Symbols: {len(symbols)}\n")
        
        for symbol in tqdm(symbols):
            data = self.collect_options_data(symbol)
            if data:
                try:
                    df = pd.DataFrame([data])
                    df.to_parquet(
                        self.output_dir / f"{symbol}.parquet",
                        compression='snappy'
                    )
                    self.stats['collected'] += 1
                except:
                    self.stats['failed'] += 1
            else:
                self.stats['failed'] += 1
        
        # Results
        print(f"\n✅ Collected: {self.stats['collected']}")
        print(f"❌ Failed: {self.stats['failed']}")
        print(f"📁 Output: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=int, default=6000)
    parser.add_argument('--output', type=str, default='project/data/features')
    args = parser.parse_args()
    
    collector = OptionsCollectorSequential(args.output)
    symbols = collector.get_us_stocks(args.symbols)
    collector.collect_batch(symbols)


if __name__ == '__main__':
    main()
