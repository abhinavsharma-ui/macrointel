path = 'project/run.py'
with open(path, 'r') as f:
    lines = f.readlines()

with open(path, 'w') as f:
    for line in lines:
        f.write(line)
        if "def _get_target_symbols(self" in line:
            # Calculate exact spacing of the def line and add 4 spaces
            leading_spaces = len(line) - len(line.lstrip())
            indent = " " * (leading_spaces + 4)
            # Inject the 150 stocks
            f.write(indent + 'return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "V", "JPM", "WMT", "JNJ", "MA", "PG", "HD", "CVX", "MRK", "ABBV", "COST", "PEP", "KO", "BAC", "XOM", "TMO", "CSCO", "MCD", "ABT", "CRM", "INTC", "DHR", "NFLX", "AMD", "CMCSA", "ADBE", "WFC", "TXN", "PM", "VZ", "COP", "NEE", "BMY", "UNP", "LIN", "RTX", "HON", "AMGN", "LOW", "QCOM", "SPGI", "INTU", "IBM", "CAT", "GE", "NOW", "PLD", "BA", "GS", "ISRG", "BLK", "MDLZ", "SYK", "BKNG", "T", "ELV", "GILD", "DE", "AMAT", "LMT", "C", "ADI", "ADP", "VRTX", "TJX", "MMC", "CB", "CHTR", "SBUX", "PANW", "REGN", "CI", "BSX", "PGR", "ZTS", "CVS", "SO", "FI", "TMUS", "MU", "BDX", "DUK", "CME", "EQIX", "SNPS", "EOG", "AON", "ITW", "KLAC", "SLB", "CDNS", "WM", "SHW", "CSX", "ORLY", "ICE", "MCO", "CL", "FCX", "MO", "NOC", "MCK", "FDX", "HCA", "TGT", "PSA", "PXD", "MPC", "PH", "NXPI", "EW", "VLO", "AEP", "MAR", "PNC", "EMR", "USB", "KMB", "ROP", "LRCX", "AIG", "AZO", "EXC", "NSC", "AJG", "ROST", "TT", "CMG", "BIIB", "AFL", "TRV", "MSI", "O", "TEL", "GPN", "PAYX"]\n')
print("Perfect injection complete.")
