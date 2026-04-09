"""
Stock Universe
===============
500+ symbols across NSE (India) and NYSE/NASDAQ (US).
Organised by sector for sector-rotation analysis.

Usage:
    from pipeline.universe import NSE_SYMBOLS, US_SYMBOLS, ALL_SYMBOLS
    from pipeline.universe import get_sector, get_universe
"""

import os
from pathlib import Path
from typing import List

# ─────────────────────────────────────────────────────────────
# NSE Universe — 200 symbols
# ─────────────────────────────────────────────────────────────

NSE_LARGE_CAP = [
    # IT
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    "LTIM.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS", "OFSS.NS",
    # Banking & Finance
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS",
    "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "SBICARD.NS",
    # Energy & Oil
    "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "GAIL.NS",
    "POWERGRID.NS", "NTPC.NS", "ADANIGREEN.NS", "TATAPOWER.NS", "CESC.NS",
    # Auto
    "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "ASHOKLEY.NS", "TVSMOTOR.NS", "BOSCHLTD.NS", "MOTHERSON.NS",
    # Pharma
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS",
    "LUPIN.NS", "AUROPHARMA.NS", "TORNTPHARM.NS", "ALKEM.NS", "IPCA.NS",
    # FMCG
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
    "MARICO.NS", "COLPAL.NS", "GODREJCP.NS", "EMAMILTD.NS", "VBL.NS",
    # Metals & Mining
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS",
    "NMDC.NS", "SAIL.NS", "HINDCOPPER.NS", "NATIONALUM.NS", "MOIL.NS",
    # Infrastructure & Real Estate
    "LARSEN.NS", "ULTRACEMCO.NS", "GRASIM.NS", "AMBUJACEM.NS", "ACC.NS",
    "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "PHOENIXLTD.NS",
    # Telecom & Media
    "BHARTIARTL.NS", "IDEA.NS", "TATACOMM.NS", "INDIAMART.NS", "ZOMATO.NS",
    # Consumer & Retail
    "TITAN.NS", "TRENT.NS", "DMART.NS", "NYKAA.NS", "PAYTM.NS",
    # Chemicals
    "PIDILITIND.NS", "SRF.NS", "DEEPAKNTR.NS", "AARTI.NS", "NAVINFLUOR.NS",
    # Insurance
    "HDFCLIFE.NS", "SBILIFE.NS", "ICICIlombard.NS", "ICICIPRULIFE.NS", "LICI.NS",
    # Aviation & Logistics
    "INDIGO.NS", "DELHIVERY.NS", "CONCOR.NS", "BLUEDART.NS", "GESHIP.NS",
]

NSE_MID_CAP = [
    "IRCTC.NS", "POLYCAB.NS", "DIXON.NS", "KALYANKJIL.NS", "SAPPHIRE.NS",
    "KPITTECH.NS", "TATAELXSI.NS", "LTTS.NS", "CYIENT.NS", "RATEGAIN.NS",
    "ANGELONE.NS", "BSE.NS", "CAMS.NS", "CDSL.NS", "MCX.NS",
    "AARTIIND.NS", "CLEAN.NS", "LAXMIMACH.NS", "GRINDWELL.NS", "SCHAEFFLER.NS",
    "ABFRL.NS", "PAGEIND.NS", "MANYAVAR.NS", "VEDANT.NS", "SHOPERSTOP.NS",
    "METROPOLIS.NS", "DRLALPATH.NS", "THYROCARE.NS", "VIJAYABANK.NS", "EQUITASBNK.NS",
]

NSE_SYMBOLS = NSE_LARGE_CAP + NSE_MID_CAP


# ─────────────────────────────────────────────────────────────
# US Universe — 300 symbols
# ─────────────────────────────────────────────────────────────

US_MEGA_CAP = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "ORCL", "AMD",
    "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "NXPI", "ON",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "COF", "USB",
    "V", "MA", "PYPL", "SQ", "COIN", "SCHW", "HOOD", "IBKR", "ICE", "CME",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "BMY", "AMGN", "GILD", "BIIB",
    "ISRG", "SYK", "BSX", "EW", "MDT", "ZBH", "HOLX", "DXCM", "PODD", "TDOC",
    # Consumer
    "AMZN", "WMT", "TGT", "COST", "HD", "LOW", "MCD", "SBUX", "NKE", "LULU",
    "TJX", "ROST", "DG", "DLTR", "KR", "SFM", "ULTA", "FIVE", "BJ", "OLLI",
    # Energy
    "XOM", "CVX", "COP", "EOG", "PXD", "MPC", "VLO", "PSX", "OXY", "SLB",
    "HAL", "BKR", "DVN", "FANG", "APA", "MRO", "HES", "CTRA", "PR", "SM",
    # Industrial
    "CAT", "DE", "HON", "GE", "RTX", "LMT", "NOC", "BA", "GD", "LHX",
    "UPS", "FDX", "CSX", "UNP", "NSC", "DAL", "UAL", "AAL", "LUV", "ALK",
    # ETFs (for macro signals)
    "SPY", "QQQ", "IWM", "DIA", "VTI", "GLD", "SLV", "TLT", "HYG", "VIX",
    # Communication
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "PARA", "WBD", "FOXA",
    # Real Estate
    "AMT", "PLD", "EQIX", "CCI", "SPG", "O", "PSA", "EQR", "AVB", "DRE",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PCG", "ES", "ETR",
    # New Tech / AI plays
    "PLTR", "AI", "SNOW", "DDOG", "NET", "ZS", "CRWD", "PANW", "S", "FTNT",
    "NOW", "CRM", "WDAY", "ADSK", "TEAM", "HUBS", "VEEV", "DOCU", "ZM", "OKTA",
    "UBER", "LYFT", "ABNB", "DASH", "RBLX", "SNAP", "PINS", "SPOT", "MTCH", "IAC",
]

US_SYMBOLS = list(dict.fromkeys(US_MEGA_CAP))   # deduplicate preserving order

# ─────────────────────────────────────────────────────────────
# Combined universe
# ─────────────────────────────────────────────────────────────

ALL_SYMBOLS = US_SYMBOLS + NSE_SYMBOLS

# Smaller curated list for fast startup / testing
CORE_US_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    "JPM", "BAC", "V", "MA", "UNH", "JNJ", "XOM", "CVX",
    "SPY", "QQQ", "GLD", "TLT",
]

CORE_NSE_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
    "BHARTIARTL.NS", "KOTAKBANK.NS", "WIPRO.NS", "AXISBANK.NS", "ITC.NS",
]

CORE_SYMBOLS = CORE_US_SYMBOLS + CORE_NSE_SYMBOLS

LEADER_SYMBOLS = list(
    dict.fromkeys(
        CORE_SYMBOLS
        + [
            "AMD",
            "AVGO",
            "NFLX",
            "QCOM",
            "ORCL",
            "CRM",
            "PLTR",
            "COST",
            "WMT",
            "SPY",
            "QQQ",
            "RELIANCE.NS",
            "HDFCBANK.NS",
            "TCS.NS",
            "INFY.NS",
        ]
    )
)
LEADER_SYMBOL_SET = set(LEADER_SYMBOLS)

DAY_TRADE_US_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "MRVL", "QCOM", "PLTR",
    "TSLA", "META", "AMZN", "GOOGL", "NFLX", "UBER", "LYFT", "ABNB",
    "COIN", "HOOD", "AI", "SNOW", "NET", "CRWD", "PANW", "ZS",
    "DAL", "UAL", "AAL", "LUV", "ALK", "XOM", "CVX", "COP",
    "SPY", "QQQ", "IWM", "VIX", "SM", "FANG", "OXY", "MRO",
]

DAY_TRADE_NSE_SYMBOLS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "BANKBARODA.NS",
    "BAJFINANCE.NS", "TCS.NS", "INFY.NS", "TATAMOTORS.NS", "MARUTI.NS",
    "M&M.NS", "INDIGO.NS", "IRCTC.NS", "DELHIVERY.NS", "CONCOR.NS",
    "ZOMATO.NS", "BHARTIARTL.NS", "BSE.NS", "MCX.NS", "ANGELONE.NS",
    "ADANIGREEN.NS", "TATAPOWER.NS", "ONGC.NS", "IOC.NS", "BPCL.NS",
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "AUROPHARMA.NS", "PIDILITIND.NS",
]

DAY_TRADE_SYMBOLS = list(dict.fromkeys(DAY_TRADE_US_SYMBOLS + DAY_TRADE_NSE_SYMBOLS))
DAY_TRADE_SYMBOL_SET = set(DAY_TRADE_SYMBOLS)
DAY_TRADE_EXPANDED_SYMBOLS = list(
    dict.fromkeys(
        DAY_TRADE_SYMBOLS
        + LEADER_SYMBOLS
        + US_SYMBOLS
        + NSE_LARGE_CAP
    )
)
DAY_TRADE_EXPANDED_SYMBOL_SET = set(DAY_TRADE_EXPANDED_SYMBOLS)


def _dedupe_symbols(values: List[str]) -> List[str]:
    return list(dict.fromkeys([str(value).strip().upper() for value in values if str(value).strip()]))


def _parse_symbol_blob(raw: str) -> List[str]:
    text = str(raw or "")
    chunks = []
    for line in text.splitlines():
        chunks.extend(line.replace(";", ",").split(","))
    cleaned = []
    for value in chunks:
        item = str(value).strip()
        if not item or item.startswith("#"):
            continue
        cleaned.append(item)
    return _dedupe_symbols(cleaned)


def _read_symbol_file(path_value: str) -> List[str]:
    path_text = str(path_value or "").strip()
    if not path_text:
        return []
    path = Path(path_text)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.exists():
        return []
    try:
        return _parse_symbol_blob(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _env_symbols(*env_names: str) -> List[str]:
    values: List[str] = []
    for env_name in env_names:
        raw = os.getenv(env_name, "")
        if raw:
            values.extend(_parse_symbol_blob(raw))
    return _dedupe_symbols(values)


def _extra_universe_symbols(mode: str) -> List[str]:
    mode_key = str(mode or "").lower()
    extras: List[str] = []
    extras.extend(_env_symbols("UNIVERSE_EXTRA_SYMBOLS"))
    extras.extend(_read_symbol_file(os.getenv("UNIVERSE_EXTRA_FILE", "")))

    if mode_key in {"full", "us", "daytrade", "daytrade_us", "throughput", "throughput_us"}:
        extras.extend(_env_symbols("US_UNIVERSE_EXTRA_SYMBOLS"))
        extras.extend(_read_symbol_file(os.getenv("US_UNIVERSE_EXTRA_FILE", "")))
    if mode_key in {"full", "nse", "daytrade", "daytrade_nse", "throughput", "throughput_nse"}:
        extras.extend(_env_symbols("NSE_UNIVERSE_EXTRA_SYMBOLS"))
        extras.extend(_read_symbol_file(os.getenv("NSE_UNIVERSE_EXTRA_FILE", "")))
    if mode_key in {"daytrade", "daytrade_us", "daytrade_nse", "throughput", "throughput_us", "throughput_nse"}:
        extras.extend(_env_symbols("DAY_TRADE_EXTRA_SYMBOLS"))
        extras.extend(_read_symbol_file(os.getenv("DAY_TRADE_EXTRA_FILE", "")))

    return _dedupe_symbols(extras)

# ─────────────────────────────────────────────────────────────
# Sector mapping
# ─────────────────────────────────────────────────────────────

SECTOR_MAP = {
    # US sectors
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","GOOGL":"Technology",
    "META":"Technology","AMZN":"Consumer","TSLA":"Auto","AVGO":"Technology",
    "AMD":"Technology","INTC":"Technology","QCOM":"Technology",
    "JPM":"Finance","BAC":"Finance","WFC":"Finance","GS":"Finance","MS":"Finance",
    "V":"Finance","MA":"Finance","PYPL":"Finance","BLK":"Finance",
    "JNJ":"Healthcare","UNH":"Healthcare","PFE":"Healthcare","ABBV":"Healthcare",
    "MRK":"Healthcare","LLY":"Healthcare","AMGN":"Healthcare","GILD":"Healthcare",
    "XOM":"Energy","CVX":"Energy","COP":"Energy","EOG":"Energy","SLB":"Energy",
    "SPY":"ETF","QQQ":"ETF","GLD":"ETF","TLT":"ETF","IWM":"ETF",
    "CAT":"Industrial","DE":"Industrial","HON":"Industrial","GE":"Industrial",
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities",
    "PLTR":"Technology","SNOW":"Technology","CRWD":"Technology","NET":"Technology",
    "SOLUSDT":"Crypto","XRPUSDT":"Crypto","DOGEUSDT":"Crypto","ADAUSDT":"Crypto","LINKUSDT":"Crypto",
    "BTCUSDT":"Crypto","ETHUSDT":"Crypto",
    # NSE sectors
    "TCS.NS":"IT","INFY.NS":"IT","WIPRO.NS":"IT","HCLTECH.NS":"IT","TECHM.NS":"IT",
    "HDFCBANK.NS":"Banking","ICICIBANK.NS":"Banking","KOTAKBANK.NS":"Banking",
    "AXISBANK.NS":"Banking","SBIN.NS":"Banking","BAJFINANCE.NS":"Finance",
    "RELIANCE.NS":"Energy","ONGC.NS":"Energy","IOC.NS":"Energy","BPCL.NS":"Energy",
    "HINDUNILVR.NS":"FMCG","ITC.NS":"FMCG","NESTLEIND.NS":"FMCG",
    "SUNPHARMA.NS":"Pharma","DRREDDY.NS":"Pharma","CIPLA.NS":"Pharma",
    "TATASTEEL.NS":"Metals","JSWSTEEL.NS":"Metals","HINDALCO.NS":"Metals",
    "MARUTI.NS":"Auto","TATAMOTORS.NS":"Auto","M&M.NS":"Auto",
    "BHARTIARTL.NS":"Telecom","ZOMATO.NS":"Consumer","TITAN.NS":"Consumer",
    "LARSEN.NS":"Infrastructure","ULTRACEMCO.NS":"Infrastructure",
}


def get_sector(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "Other")


def is_leader_symbol(symbol: str) -> bool:
    return symbol in LEADER_SYMBOL_SET


def _expanded_day_trade_enabled() -> bool:
    raw = os.getenv("DAY_TRADE_EXPANDED_UNIVERSE", "1")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def is_day_trade_symbol(symbol: str) -> bool:
    if _expanded_day_trade_enabled():
        return symbol in DAY_TRADE_EXPANDED_SYMBOL_SET
    return symbol in DAY_TRADE_SYMBOL_SET


def get_universe(mode: str = "core") -> list:
    """
    mode: 'core'  — 35 key symbols (fast, good for testing)
          'full'  — 500+ symbols (slow first fetch, best signals)
          'us'    — US only
          'nse'   — NSE only
    """
    if mode == "core":
        universe = CORE_SYMBOLS
    elif mode == "full":
        universe = ALL_SYMBOLS
    elif mode == "us":
        universe = US_SYMBOLS
    elif mode == "nse":
        universe = NSE_SYMBOLS
    elif mode == "daytrade":
        universe = DAY_TRADE_EXPANDED_SYMBOLS if _expanded_day_trade_enabled() else DAY_TRADE_SYMBOLS
    elif mode == "daytrade_us":
        universe = DAY_TRADE_US_SYMBOLS
    elif mode == "daytrade_nse":
        universe = DAY_TRADE_NSE_SYMBOLS
    elif mode == "throughput":
        universe = _dedupe_symbols(ALL_SYMBOLS + DAY_TRADE_EXPANDED_SYMBOLS + LEADER_SYMBOLS)
    elif mode == "throughput_us":
        universe = _dedupe_symbols(
            US_SYMBOLS + DAY_TRADE_US_SYMBOLS + [symbol for symbol in LEADER_SYMBOLS if not symbol.endswith(".NS")]
        )
    elif mode == "throughput_nse":
        universe = _dedupe_symbols(
            NSE_SYMBOLS + DAY_TRADE_NSE_SYMBOLS + [symbol for symbol in LEADER_SYMBOLS if symbol.endswith(".NS")]
        )
    else:
        universe = CORE_SYMBOLS

    universe = _dedupe_symbols(list(universe) + _extra_universe_symbols(mode))

    try:
        limit = int(float(os.getenv("UNIVERSE_LIMIT", "0") or 0))
    except Exception:
        limit = 0
    if limit > 0:
        universe = universe[: max(1, limit)]
    return universe


COMPANY_NAME_OVERRIDES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "GOOGL": "Alphabet Google",
    "META": "Meta Platforms",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "AMD": "Advanced Micro Devices",
    "AVGO": "Broadcom",
    "ORCL": "Oracle",
    "NFLX": "Netflix",
    "QCOM": "Qualcomm",
    "INTC": "Intel",
    "MRVL": "Marvell Technology",
    "BAC": "Bank of America",
    "JPM": "JPMorgan Chase",
    "WFC": "Wells Fargo",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "C": "Citigroup",
    "BLK": "BlackRock",
    "MA": "Mastercard",
    "V": "Visa",
    "WMT": "Walmart",
    "COST": "Costco",
    "HD": "Home Depot",
    "LOW": "Lowe's",
    "TGT": "Target",
    "NKE": "Nike",
    "MCD": "McDonald's",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "COP": "ConocoPhillips",
    "EOG": "EOG Resources",
    "OXY": "Occidental Petroleum",
    "DAL": "Delta Air Lines",
    "UAL": "United Airlines",
    "AAL": "American Airlines",
    "LUV": "Southwest Airlines",
    "ALK": "Alaska Air",
    "ABNB": "Airbnb",
    "UBER": "Uber",
    "LYFT": "Lyft",
    "ISRG": "Intuitive Surgical",
    "SYK": "Stryker",
    "BSX": "Boston Scientific",
    "EW": "Edwards Lifesciences",
    "PLTR": "Palantir",
    "CRM": "Salesforce",
    "NOW": "ServiceNow",
    "NET": "Cloudflare",
    "ZS": "Zscaler",
    "CRWD": "CrowdStrike",
    "PANW": "Palo Alto Networks",
    "AMT": "American Tower",
    "PCG": "PG&E",
    "TMUS": "T-Mobile",
    "IAC": "IAC",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "WIPRO.NS": "Wipro",
    "HCLTECH.NS": "HCLTech",
    "TECHM.NS": "Tech Mahindra",
    "LTIM.NS": "LTIMindtree",
    "COFORGE.NS": "Coforge",
    "PERSISTENT.NS": "Persistent Systems",
    "OFSS.NS": "Oracle Financial Services",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "AXISBANK.NS": "Axis Bank",
    "SBIN.NS": "State Bank of India",
    "INDUSINDBK.NS": "IndusInd Bank",
    "BANKBARODA.NS": "Bank of Baroda",
    "FEDERALBNK.NS": "Federal Bank",
    "IDFCFIRSTB.NS": "IDFC First Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "CHOLAFIN.NS": "Cholamandalam Investment",
    "MUTHOOTFIN.NS": "Muthoot Finance",
    "SBICARD.NS": "SBI Cards",
    "ONGC.NS": "Oil and Natural Gas Corporation",
    "IOC.NS": "Indian Oil",
    "BPCL.NS": "Bharat Petroleum",
    "GAIL.NS": "GAIL India",
    "POWERGRID.NS": "Power Grid Corporation",
    "NTPC.NS": "NTPC",
    "ADANIGREEN.NS": "Adani Green Energy",
    "TATAPOWER.NS": "Tata Power",
    "CESC.NS": "CESC",
    "MARUTI.NS": "Maruti Suzuki",
    "TATAMOTORS.NS": "Tata Motors",
    "M&M.NS": "Mahindra and Mahindra",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "EICHERMOT.NS": "Eicher Motors",
    "ASHOKLEY.NS": "Ashok Leyland",
    "TVSMOTOR.NS": "TVS Motor",
    "BOSCHLTD.NS": "Bosch India",
    "MOTHERSON.NS": "Samvardhana Motherson",
    "SUNPHARMA.NS": "Sun Pharma",
    "DRREDDY.NS": "Dr Reddy's Laboratories",
    "CIPLA.NS": "Cipla",
    "DIVISLAB.NS": "Divi's Laboratories",
    "BIOCON.NS": "Biocon",
    "LUPIN.NS": "Lupin",
    "AUROPHARMA.NS": "Aurobindo Pharma",
    "TORNTPHARM.NS": "Torrent Pharma",
    "ALKEM.NS": "Alkem Laboratories",
    "IPCA.NS": "IPCA Laboratories",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC",
    "NESTLEIND.NS": "Nestle India",
    "BRITANNIA.NS": "Britannia Industries",
    "DABUR.NS": "Dabur",
    "MARICO.NS": "Marico",
    "COLPAL.NS": "Colgate Palmolive India",
    "GODREJCP.NS": "Godrej Consumer Products",
    "VBL.NS": "Varun Beverages",
    "TATASTEEL.NS": "Tata Steel",
    "JSWSTEEL.NS": "JSW Steel",
    "HINDALCO.NS": "Hindalco Industries",
    "VEDL.NS": "Vedanta",
    "COALINDIA.NS": "Coal India",
    "NATIONALUM.NS": "National Aluminium",
    "LARSEN.NS": "Larsen and Toubro",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "GRASIM.NS": "Grasim Industries",
    "AMBUJACEM.NS": "Ambuja Cements",
    "ACC.NS": "ACC",
    "DLF.NS": "DLF",
    "GODREJPROP.NS": "Godrej Properties",
    "OBEROIRLTY.NS": "Oberoi Realty",
    "PHOENIXLTD.NS": "Phoenix Mills",
    "BHARTIARTL.NS": "Bharti Airtel",
    "IDEA.NS": "Vodafone Idea",
    "TATACOMM.NS": "Tata Communications",
    "INDIAMART.NS": "IndiaMART",
    "ZOMATO.NS": "Zomato",
    "TITAN.NS": "Titan Company",
    "TRENT.NS": "Trent",
    "DMART.NS": "Avenue Supermarts DMart",
    "NYKAA.NS": "Nykaa",
    "PAYTM.NS": "One 97 Paytm",
    "PIDILITIND.NS": "Pidilite Industries",
    "SRF.NS": "SRF",
    "DEEPAKNTR.NS": "Deepak Nitrite",
    "AARTI.NS": "Aarti Industries",
    "NAVINFLUOR.NS": "Navin Fluorine",
    "HDFCLIFE.NS": "HDFC Life",
    "SBILIFE.NS": "SBI Life",
    "LICI.NS": "Life Insurance Corporation",
    "INDIGO.NS": "InterGlobe Aviation IndiGo",
    "DELHIVERY.NS": "Delhivery",
    "CONCOR.NS": "Container Corporation of India",
    "BLUEDART.NS": "Blue Dart",
    "GESHIP.NS": "Great Eastern Shipping",
    "IRCTC.NS": "Indian Railway Catering and Tourism",
    "POLYCAB.NS": "Polycab India",
    "DIXON.NS": "Dixon Technologies",
    "KALYANKJIL.NS": "Kalyan Jewellers",
    "KPITTECH.NS": "KPIT Technologies",
    "TATAELXSI.NS": "Tata Elxsi",
    "LTTS.NS": "L&T Technology Services",
    "CYIENT.NS": "Cyient",
    "RATEGAIN.NS": "RateGain Travel Technologies",
    "ANGELONE.NS": "Angel One",
    "BSE.NS": "BSE",
    "CAMS.NS": "Computer Age Management Services",
    "CDSL.NS": "Central Depository Services",
    "MCX.NS": "Multi Commodity Exchange",
    "AARTIIND.NS": "Aarti Industries",
    "CLEAN.NS": "Clean Science",
    "LAXMIMACH.NS": "Lakshmi Machine Works",
    "GRINDWELL.NS": "Grindwell Norton",
    "SCHAEFFLER.NS": "Schaeffler India",
    "ABFRL.NS": "Aditya Birla Fashion",
    "PAGEIND.NS": "Page Industries",
    "VEDANT.NS": "Vedant Fashions",
    "SHOPERSTOP.NS": "Shoppers Stop",
    "METROPOLIS.NS": "Metropolis Healthcare",
    "DRLALPATH.NS": "Dr Lal PathLabs",
    "THYROCARE.NS": "Thyrocare",
    "VIJAYABANK.NS": "Vijaya Bank",
    "EQUITASBNK.NS": "Equitas Small Finance Bank",
    "SOLUSDT": "Solana",
    "XRPUSDT": "XRP",
    "DOGEUSDT": "Dogecoin",
    "ADAUSDT": "Cardano",
    "LINKUSDT": "Chainlink",
    "BTCUSDT": "Bitcoin",
    "ETHUSDT": "Ethereum",
}


def get_market(symbol: str) -> str:
    if symbol.upper().endswith(("USDT", "USDC", "BUSD")):
        return "crypto"
    return "nse" if symbol.endswith(".NS") else "us"


def get_company_name(symbol: str) -> str:
    override = COMPANY_NAME_OVERRIDES.get(symbol)
    if override:
        return override
    base = symbol.replace(".NS", "").replace("^", "").replace("-", " ").strip()
    return base.title() if base else symbol


def get_symbol_aliases(symbol: str) -> List[str]:
    company_name = get_company_name(symbol)
    sector = get_sector(symbol)
    raw = symbol.replace(".NS", "").replace("^", "").strip()
    aliases = [
        company_name,
        raw,
        raw.replace("-", " "),
        company_name.replace("&", "and"),
        f"{company_name} stock",
    ]
    if symbol.endswith(".NS"):
        aliases.extend(
            [
                f"{company_name} ltd",
                f"{company_name} limited",
                f"{company_name} india",
            ]
        )
    if sector and sector != "Other":
        aliases.extend(
            [
                f"{company_name} {sector}",
                f"{raw} {sector}",
        ]
    )
    if symbol.upper().endswith(("USDT", "USDC", "BUSD")):
        base_asset = raw
        for suffix in ("USDT", "USDC", "BUSD"):
            if base_asset.endswith(suffix):
                base_asset = base_asset[: -len(suffix)]
                break
        if base_asset:
            aliases.extend(
                [
                    base_asset,
                    f"{base_asset} token",
                    f"{company_name} token",
                    f"{base_asset} coin",
                ]
            )
    seen = set()
    deduped: List[str] = []
    for alias in aliases:
        normalized = " ".join(str(alias).split()).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped
