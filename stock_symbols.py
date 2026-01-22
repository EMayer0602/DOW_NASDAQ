# DOW 30 and NASDAQ 100 Stock Symbols for Trading
# Compatible with Interactive Brokers

# DOW 30 Components (as of 2024)
DOW_30 = [
    "AAPL",   # Apple
    "AMGN",   # Amgen
    "AXP",    # American Express
    "BA",     # Boeing
    "CAT",    # Caterpillar
    "CRM",    # Salesforce
    "CSCO",   # Cisco
    "CVX",    # Chevron
    "DIS",    # Disney
    "DOW",    # Dow Inc
    "GS",     # Goldman Sachs
    "HD",     # Home Depot
    "HON",    # Honeywell
    "IBM",    # IBM
    "INTC",   # Intel
    "JNJ",    # Johnson & Johnson
    "JPM",    # JPMorgan Chase
    "KO",     # Coca-Cola
    "MCD",    # McDonald's
    "MMM",    # 3M
    "MRK",    # Merck
    "MSFT",   # Microsoft
    "NKE",    # Nike
    "PG",     # Procter & Gamble
    "TRV",    # Travelers
    "UNH",    # UnitedHealth
    "V",      # Visa
    "VZ",     # Verizon
    "WBA",    # Walgreens
    "WMT",    # Walmart
]

# NASDAQ 100 Top Components (most liquid)
NASDAQ_100_TOP = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "AMZN",   # Amazon
    "NVDA",   # NVIDIA
    "GOOGL",  # Alphabet Class A
    "GOOG",   # Alphabet Class C
    "META",   # Meta Platforms
    "TSLA",   # Tesla
    "AVGO",   # Broadcom
    "COST",   # Costco
    "PEP",    # PepsiCo
    "ADBE",   # Adobe
    "NFLX",   # Netflix
    "AMD",    # AMD
    "CSCO",   # Cisco
    "INTC",   # Intel
    "CMCSA",  # Comcast
    "TMUS",   # T-Mobile
    "QCOM",   # Qualcomm
    "TXN",    # Texas Instruments
    "INTU",   # Intuit
    "AMGN",   # Amgen
    "AMAT",   # Applied Materials
    "ISRG",   # Intuitive Surgical
    "HON",    # Honeywell
    "BKNG",   # Booking Holdings
    "SBUX",   # Starbucks
    "VRTX",   # Vertex Pharma
    "GILD",   # Gilead
    "ADI",    # Analog Devices
    "MDLZ",   # Mondelez
    "ADP",    # ADP
    "REGN",   # Regeneron
    "LRCX",   # Lam Research
    "MU",     # Micron
    "PANW",   # Palo Alto Networks
    "SNPS",   # Synopsys
    "KLAC",   # KLA Corp
    "CDNS",   # Cadence Design
    "PYPL",   # PayPal
    "MELI",   # MercadoLibre
    "ORLY",   # O'Reilly Auto
    "MAR",    # Marriott
    "ABNB",   # Airbnb
    "FTNT",   # Fortinet
    "CTAS",   # Cintas
    "MNST",   # Monster Beverage
    "WDAY",   # Workday
    "KDP",    # Keurig Dr Pepper
    "DXCM",   # DexCom
]

# Combined unique symbols (remove duplicates)
ALL_STOCKS = list(set(DOW_30 + NASDAQ_100_TOP))

# Default symbols to trade (start small)
DEFAULT_TRADING_SYMBOLS = [
    "AAPL",   # Apple - very liquid
    "MSFT",   # Microsoft - very liquid
    "NVDA",   # NVIDIA - high volatility
    "AMZN",   # Amazon - liquid
    "GOOGL",  # Google - liquid
    "META",   # Meta - good volatility
    "TSLA",   # Tesla - high volatility
    "JPM",    # JPMorgan - financials
    "V",      # Visa - stable
    "HD",     # Home Depot - retail
]

# IB Contract specifications
def get_ib_contract(symbol: str):
    """Create IB contract for US stock."""
    from ib_insync import Stock
    return Stock(symbol, 'SMART', 'USD')

# Map symbol to exchange (for IB)
SYMBOL_EXCHANGE_MAP = {sym: "SMART" for sym in ALL_STOCKS}

# Trading hours (US Eastern Time)
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
PRE_MARKET_OPEN = "04:00"
AFTER_HOURS_CLOSE = "20:00"

# Minimum price for trading (avoid penny stocks)
MIN_STOCK_PRICE = 5.0

# Position sizing
DEFAULT_POSITION_VALUE = 1000  # USD per position
MAX_POSITIONS = 10
