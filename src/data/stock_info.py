# Mock stock information, including risk status (e.g., ST, *ST)
# In a real system, this would be fetched from a financial data provider.

MOCK_STOCK_INFO = {
    "SH600519": { # Kweichow Moutai
        "status": "Normal",
        "name": "Kweichow Moutai Co.,Ltd",
        # Example financial data point for rule engine testing
        "debt_to_equity_ratio": 0.2
    },
    "ST_KANGMEI": { # Fictional ST Kangmei for testing
        "status": "ST",
        "name": "ST Kangmei Pharmaceutical Co.,Ltd.",
        "debt_to_equity_ratio": 0.85 # Example: High ratio
    },
    "POTENTIAL_DELIST": { # Fictional stock for delisting rule test
        "status": "Normal", # Initially normal
        "name": "Potential Delist Corp.",
        "debt_to_equity_ratio": 0.5
    },
    "AAPL": {
        "status": "Normal",
        "name": "Apple Inc.",
        "debt_to_equity_ratio": 1.5 # Apple has high D/E due to share buybacks
    }
}

# This would be more dynamic in a real system, perhaps with dates for status changes
# For simplicity, status is static here, or rules are applied based on other metrics.

def get_stock_status(ticker: str) -> str | None:
    """Returns the risk status of a stock (e.g., 'Normal', 'ST', '*ST')."""
    if ticker in MOCK_STOCK_INFO:
        return MOCK_STOCK_INFO[ticker].get("status")
    return None

def get_stock_financial_metric(ticker: str, metric_name: str) -> float | None:
    """Returns a specific financial metric for a stock from mock data."""
    if ticker in MOCK_STOCK_INFO:
        return MOCK_STOCK_INFO[ticker].get(metric_name)
    return None

if __name__ == '__main__':
    print(f"Status of SH600519: {get_stock_status('SH600519')}")
    print(f"Status of ST_KANGMEI: {get_stock_status('ST_KANGMEI')}")
    print(f"Status of UNKNOWN_TICKER: {get_stock_status('UNKNOWN_TICKER')}")

    print(f"Debt/Equity for SH600519: {get_stock_financial_metric('SH600519', 'debt_to_equity_ratio')}")
    print(f"Debt/Equity for ST_KANGMEI: {get_stock_financial_metric('ST_KANGMEI', 'debt_to_equity_ratio')}")
