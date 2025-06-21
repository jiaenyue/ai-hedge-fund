from datetime import datetime

# Mock adjustment data (dividends, splits)
# In a real system, this would be fetched from a financial data provider.
# Factor for split: 1 old share becomes 'factor' new shares. Price per share is divided by 'factor'.
# For stock dividends (bonus issues), it can also be represented as a split factor.
# E.g., a 10% stock dividend means 1 share becomes 1.1 shares (factor = 1.1).

MOCK_ADJUSTMENT_DATA = {
    "SH600519": { # Example: Kweichow Moutai
        "2023-06-29": {"type": "dividend", "cash_per_share": 25.911}, # Fictional large dividend for testing
        # No recent splits for Moutai, but let's imagine one for testing
        "2022-05-10": {"type": "split", "factor": 2.0} # 2-for-1 split
    },
    "AAPL": { # Example with US stock data format for broader testing
        "2023-11-10": {"type": "dividend", "cash_per_share": 0.24},
        "2023-08-11": {"type": "dividend", "cash_per_share": 0.24},
        "2020-08-31": {"type": "split", "factor": 4.0} # 4-for-1 split
    }
}

def get_adjustments_for_date(ticker: str, date_str: str) -> list[dict]:
    """
    Returns a list of adjustment events (if any) for a given ticker on a specific date.
    Date_str should be in "YYYY-MM-DD" format.
    """
    if ticker in MOCK_ADJUSTMENT_DATA:
        return [MOCK_ADJUSTMENT_DATA[ticker][date_str]] if date_str in MOCK_ADJUSTMENT_DATA[ticker] else []
    return []

def get_all_adjustments_for_ticker(ticker: str) -> dict:
    """
    Returns all adjustment data for a ticker.
    { "YYYY-MM-DD": {"type": "dividend", ...} }
    """
    return MOCK_ADJUSTMENT_DATA.get(ticker, {})

if __name__ == '__main__':
    # Example usage
    print("SH600519 adjustments on 2023-06-29:", get_adjustments_for_date("SH600519", "2023-06-29"))
    print("SH600519 adjustments on 2022-05-10:", get_adjustments_for_date("SH600519", "2022-05-10"))
    print("AAPL adjustments on 2020-08-31:", get_adjustments_for_date("AAPL", "2020-08-31"))
    print("AAPL adjustments on 2023-01-01 (none expected):", get_adjustments_for_date("AAPL", "2023-01-01"))

    print("\nAll adjustments for SH600519:", get_all_adjustments_for_ticker("SH600519"))
