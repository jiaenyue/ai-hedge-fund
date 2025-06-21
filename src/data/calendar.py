from datetime import date, timedelta, datetime
import pandas as pd

# Mock holiday list (YYYY-MM-DD format) - for demonstration purposes
# In a real scenario, this would come from an API or a comprehensive holiday calendar library
MOCK_HOLIDAYS_SSE = [
    date(2024, 1, 1),  # New Year's Day
    date(2024, 2, 9),  # Chinese New Year's Eve (example, often a special arrangement)
    date(2024, 2, 10), # Chinese New Year
    date(2024, 2, 11), # Chinese New Year
    date(2024, 2, 12), # Chinese New Year
    date(2024, 2, 13), # Chinese New Year
    date(2024, 2, 14), # Chinese New Year
    date(2024, 2, 15), # Chinese New Year
    date(2024, 2, 16), # Chinese New Year
    date(2024, 5, 1),  # Labour Day
    date(2024, 10, 1), # National Day
    date(2024, 10, 2), # National Day
    date(2024, 10, 3), # National Day
]

# SZSE might have a similar or identical calendar, but could differ.
# For this mock, we'll use the same holiday list.
MOCK_HOLIDAYS_SZSE = MOCK_HOLIDAYS_SSE


def get_trading_calendar(start_date_str: str, end_date_str: str, exchange: str = "SSE") -> list[pd.Timestamp]:
    """
    Returns a list of actual trading days (as pandas Timestamps) for the given date range and exchange.
    This is a MOCK implementation.
    In a real system, this would query an official exchange API or a financial data provider.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    holidays = []
    if exchange.upper() == "SSE":
        holidays = MOCK_HOLIDAYS_SSE
    elif exchange.upper() == "SZSE":
        holidays = MOCK_HOLIDAYS_SZSE
    else:
        # Default to SSE holidays if exchange is unknown for this mock
        holidays = MOCK_HOLIDAYS_SSE
        print(f"Warning: Unknown exchange '{exchange}'. Defaulting to SSE holiday calendar.")

    trading_days = []
    current_day = start_date
    while current_day <= end_date:
        # Monday is 0, Sunday is 6
        if current_day.weekday() < 5:  # Monday to Friday
            if current_day not in holidays:
                trading_days.append(pd.Timestamp(current_day))
        current_day += timedelta(days=1)

    return trading_days

if __name__ == '__main__':
    # Example usage:
    sse_calendar_2024_01 = get_trading_calendar("2024-01-01", "2024-01-31", "SSE")
    print("SSE Trading Days in Jan 2024 (Mock):")
    for day in sse_calendar_2024_01:
        print(day.strftime("%Y-%m-%d %A"))

    print("\n" + "="*30 + "\n")

    sse_calendar_2024_02 = get_trading_calendar("2024-02-01", "2024-02-29", "SSE")
    print("SSE Trading Days in Feb 2024 (Mock) - Expecting Chinese New Year closure:")
    for day in sse_calendar_2024_02:
        print(day.strftime("%Y-%m-%d %A"))

    szse_calendar_2024_05 = get_trading_calendar("2024-04-28", "2024-05-05", "SZSE")
    print("\nSZSE Trading Days around Labour Day 2024 (Mock):")
    for day in szse_calendar_2024_05:
        print(day.strftime("%Y-%m-%d %A"))
