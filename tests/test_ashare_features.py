import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta

# Assuming src is in PYTHONPATH or accessible
from src.backtester import Backtester
from src.data.calendar import MOCK_HOLIDAYS_SSE
from src.data.adjustments import MOCK_ADJUSTMENT_DATA
from src.data.stock_info import MOCK_STOCK_INFO

# A simple dummy agent for testing backtester logic
def dummy_agent(tickers, start_date, end_date, portfolio, model_name, model_provider, selected_analysts):
    # Agent makes no decisions, allows us to control trades directly in tests
    return {"decisions": {}, "analyst_signals": {}}

class TestAShareFeatures(unittest.TestCase):

    def setUp(self):
        # Basic setup for most tests
        self.tickers = ["SH600519", "ST_KANGMEI", "AAPL"] # Using a mix
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-31" # Short period for basic tests
        self.initial_capital = 100000.0

        # Mock get_price_data to return predictable prices
        self.mock_price_patcher = patch('src.tools.api.get_price_data')
        self.mock_get_price_data = self.mock_price_patcher.start()

        # Mock prefetch_data to avoid actual API calls during tests
        self.mock_prefetch_patcher = patch.object(Backtester, 'prefetch_data', MagicMock())
        self.mock_prefetch_data = self.mock_prefetch_patcher.start()

        # Default mock price behavior: return a DataFrame with a close price
        def default_price_data(ticker, date_from, date_to):
            price = 10.0 # Default price for most tests
            if ticker == "SH600519":
                price = 100.0
            elif ticker == "ST_KANGMEI":
                price = 5.0

            # Simulate price movement if needed for specific tests
            # For now, constant price unless overridden in test
            dt_to = pd.to_datetime(date_to)
            return pd.DataFrame([{
                "time": dt_to.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "open": price, "high": price, "low": price, "close": price, "volume": 1000
            }])
        self.mock_get_price_data.side_effect = default_price_data


    def tearDown(self):
        self.mock_price_patcher.stop()
        self.mock_prefetch_patcher.stop()

    def _run_backtest_with_trades(self, backtester, trade_sequence, initial_prices=None):
        """
        Helper to run backtest by overriding agent decisions.
        trade_sequence is a list of tuples: (date_str, ticker, action, quantity, expected_exec_price)
        initial_prices is a dict: {ticker: price} for the day before the first trade_sequence date
        """

        agent_decisions_by_date = {}
        all_trade_dates = sorted(list(set(trade[0] for trade in trade_sequence)))

        if initial_prices:
            backtester.run_backtest() # Run once to initialize prev_day_close_prices if needed for first actual trade day
            # This is a bit of a hack; a cleaner way would be to set prev_day_close_prices directly.
            # For now, let's assume the first day of trading has no limits if not set.
            # Or, set it directly:
            if hasattr(backtester, 'previous_day_close_prices') and all_trade_dates:
                 # Find the date just before the first trade date to set initial prices
                first_trade_dt = datetime.strptime(all_trade_dates[0], "%Y-%m-%d")
                day_before_first_trade = first_trade_dt - timedelta(days=1) # approx

                # This part is tricky because previous_day_close_prices is updated *after* trades.
                # For simplicity, let's assume tests will handle the first day's limit condition if critical.
                # A better approach would be to allow direct setting of previous_day_close_prices before a run.
                # Let's manually set it for the test if needed.
                # backtester.previous_day_close_prices = initial_prices.copy() # This is not how it's structured
                pass


        for date_str, ticker, action, quantity, price_at_trade in trade_sequence:
            if date_str not in agent_decisions_by_date:
                agent_decisions_by_date[date_str] = {}
            if ticker not in agent_decisions_by_date[date_str]:
                agent_decisions_by_date[date_str][ticker] = {"action": "hold", "quantity": 0} # Default hold

            agent_decisions_by_date[date_str][ticker] = {"action": action, "quantity": quantity}

        def side_effect_agent(tickers, start_date, end_date, portfolio, model_name, model_provider, selected_analysts):
            # `end_date` here is the `current_date_str` in backtester loop
            decisions_for_today = agent_decisions_by_date.get(end_date, {})
            return {"decisions": decisions_for_today, "analyst_signals": {}}

        # Mock the agent passed to Backtester
        backtester.agent = MagicMock(side_effect=side_effect_agent)

        # Mock get_price_data to return specific prices for trade execution days if needed
        # This part needs to be more dynamic based on trade_sequence's expected prices
        original_price_side_effect = self.mock_get_price_data.side_effect

        def price_side_effect_for_trades(ticker, date_from, date_to):
            # date_to is current_date_str
            for trade_date, trade_ticker, _, _, trade_price in trade_sequence:
                if trade_date == date_to and trade_ticker == ticker:
                    dt_to = pd.to_datetime(date_to)
                    return pd.DataFrame([{
                        "time": dt_to.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "open": trade_price, "high": trade_price, "low": trade_price, "close": trade_price, "volume": 1000
                    }])
            # Fallback to original default or specific test setup
            if callable(original_price_side_effect):
                 return original_price_side_effect(ticker,date_from,date_to)
            return default_price_data(ticker,date_from,date_to) # Fallback

        self.mock_get_price_data.side_effect = price_side_effect_for_trades

        backtester.run_backtest()
        self.mock_get_price_data.side_effect = original_price_side_effect # Restore


    # Test Cases Start Here

    def test_t_plus_1_rule(self):
        """ Test T+1 Rule:当日买入禁止卖出验证 """
        print("\nRunning test_t_plus_1_rule...")
        bt = Backtester(
            agent=dummy_agent,
            tickers=["AAPL"],
            start_date="2023-01-02", # Mon (assuming 2023-01-01 Sun)
            end_date="2023-01-04", # Wed
            initial_capital=self.initial_capital,
            position_lock_days=1 # Explicitly T+1
        )

        # Day 1 (2023-01-02): Buy AAPL
        # Day 2 (2023-01-03): Try to sell AAPL (should fail), Buy more AAPL
        # Day 3 (2023-01-04): Sell AAPL bought on Day 1 (should succeed)

        trades = [
            ("2023-01-02", "AAPL", "buy", 10, 150.00), # Buy 10 shares at $150
            ("2023-01-03", "AAPL", "sell", 5, 152.00),  # Attempt to sell 5 shares (bought on 01-02, T+1 rule applies)
            ("2023-01-03", "AAPL", "buy", 5, 151.00),   # Buy 5 more shares
            ("2023-01-04", "AAPL", "sell", 10, 155.00) # Attempt to sell 10 shares (from 01-02)
        ]

        self._run_backtest_with_trades(bt, trades)

        # Check portfolio state after each relevant day by inspecting bt.portfolio
        # This requires a bit more instrumentation or careful checking of logs/final state.
        # For simplicity, let's check final state.
        # On 2023-01-02: Bought 10 AAPL. Lots: [{"date": "2023-01-02", "q": 10, "p": 150}]
        # On 2023-01-03: Sell of 5 should be rejected. Bought 5 more.
        #   Lots: [{"date": "2023-01-02", "q": 10, "p": 150}, {"date": "2023-01-03", "q": 5, "p": 151}]
        #   Shares: 15. Cash: 100000 - (10*150) - (5*151) = 100000 - 1500 - 755 = 97745
        # On 2023-01-04: Sell 10 (from 01-02 lot).
        #   Original 10 from 01-02 are sellable.
        #   Remaining lots: [{"date": "2023-01-03", "q": 5, "p": 151}]
        #   Shares: 5.
        #   Cash: 97745 + (10*155) = 97745 + 1550 = 99295
        #   Realized P&L from sale: (155 - 150) * 10 = 50

        final_pos_aapl = bt.portfolio["positions"]["AAPL"]
        self.assertEqual(final_pos_aapl["long_shares"], 5, "T+1: Incorrect final share quantity for AAPL")
        self.assertAlmostEqual(bt.portfolio["cash"], 99295, places=2, msg="T+1: Incorrect final cash")
        self.assertAlmostEqual(bt.portfolio["realized_gains"]["AAPL"]["long"], 50, places=2, msg="T+1: Incorrect realized P&L")
        self.assertEqual(len(final_pos_aapl["long_lots"]), 1, "T+1: Incorrect number of lots remaining")
        if final_pos_aapl["long_lots"]:
            self.assertEqual(final_pos_aapl["long_lots"][0]["date_acquired"], "2023-01-03", "T+1: Remaining lot has incorrect date")
            self.assertEqual(final_pos_aapl["long_lots"][0]["quantity"], 5, "T+1: Remaining lot has incorrect quantity")

    def test_price_limits(self):
        """ Test Price Limits (涨跌停处理) - SH600519 example """
        print("\nRunning test_price_limits...")
        ticker_symbol = "SH600519"
        prev_close_price = 100.0
        limit_up_price = round(prev_close_price * 1.1, 2) # 110.0
        limit_down_price = round(prev_close_price * 0.9, 2) # 90.0

        bt = Backtester(
            agent=dummy_agent,
            tickers=[ticker_symbol],
            start_date="2023-01-03", # Trades on this day
            end_date="2023-01-03",
            initial_capital=self.initial_capital,
            limit_up_ratio=1.1,
            limit_down_ratio=0.9
        )

        # Manually set previous day close for the first day of trading in the test.
        # The run_backtest loop updates previous_day_close_prices *after* the first iteration's trades.
        # To test limits on the *first active day* of the test range, we prime it.
        # This requires access to the internal `previous_day_close_prices` or a helper.
        # For now, we will run a "setup" day.

        setup_trades = [("2023-01-02", ticker_symbol, "buy", 0, prev_close_price)] # Dummy trade to set prev_close via current_price

        trades_to_test = [
            # Attempt to buy at limit up price (should be rejected)
            ("2023-01-03", ticker_symbol, "buy", 10, limit_up_price),
            # Attempt to buy above limit up price (should be rejected)
            ("2023-01-03", ticker_symbol, "buy", 10, limit_up_price + 1.0),
            # Attempt to buy just below limit up (should succeed)
            ("2023-01-03", ticker_symbol, "buy", 1, limit_up_price - 0.01),
             # Setup for sell: need some shares first, bought on a previous day.
             # Let's re-init backtester for sell test or make it multi-day.
        ]

        # --- Test Buy Limits ---
        # To correctly set previous_day_close_prices for 2023-01-03,
        # the backtester needs to have processed 2023-01-02 where current_prices was prev_close_price.
        # This is a bit complex with the current _run_backtest_with_trades helper.
        # A simpler way: mock get_price_data to make 2023-01-02 close be prev_close_price,
        # then run for 2023-01-03.

        def custom_price_side_effect(ticker, date_from, date_to):
            dt_to_str = pd.to_datetime(date_to).strftime("%Y-%m-%d")
            if ticker == ticker_symbol:
                if dt_to_str == "2023-01-02": # Previous day that sets up prev_close
                     return pd.DataFrame([{"time": "t", "close": prev_close_price}])
                if dt_to_str == "2023-01-03": # Trading day
                    # For this test, the 'current_price' for execution will be what we provide in trades_to_test
                    # So, this can return a dummy or one of the trade prices.
                    # The _run_backtest_with_trades helper already handles overriding price for trade execution.
                    # So, this just needs to provide *some* valid data for the day if not covered by a trade.
                    return pd.DataFrame([{"time": "t", "close": limit_up_price - 0.01}])
            return default_price_data(ticker,date_from,date_to)

        self.mock_get_price_data.side_effect = custom_price_side_effect

        # Buy 1 share successfully first to have something for sell test later.
        initial_buy_trades = [("2023-01-03", ticker_symbol, "buy", 1, limit_up_price - 0.01)]
        self._run_backtest_with_trades(bt, initial_buy_trades)
        self.assertEqual(bt.portfolio["positions"][ticker_symbol]["long_shares"], 1, "Price Limit Test: Initial buy failed")

        # Test rejected buy at limit up
        rejected_buy_trades = [("2023-01-03", ticker_symbol, "buy", 10, limit_up_price)]
        # We need to reset portfolio cash for this part or track cash precisely
        # For simplicity, let's assume the backtester state (portfolio) persists.
        # The previous_day_close_prices should be set from the 2023-01-02 run.
        # No, run_backtest resets the portfolio unless we manage state carefully.
        # Let's re-initialize for clarity or make _run_backtest_with_trades more stateful / chainable.

        # Re-initialize backtester for a clean state for each sub-test or make helper more advanced.
        # For this test, let's focus on one scenario at a time.
        bt_buy_reject = Backtester(agent=dummy_agent, tickers=[ticker_symbol], start_date="2023-01-03", end_date="2023-01-03", initial_capital=self.initial_capital, limit_up_ratio=1.1, limit_down_ratio=0.9)
        self.mock_get_price_data.side_effect = custom_price_side_effect # Ensure mock is set for this instance too
        # Manually prime previous_day_close_prices for this specific test instance
        bt_buy_reject.previous_day_close_prices = {ticker_symbol: prev_close_price}

        self._run_backtest_with_trades(bt_buy_reject, [("2023-01-03", ticker_symbol, "buy", 10, limit_up_price)]) # Try to buy at limit
        self.assertEqual(bt_buy_reject.portfolio["positions"][ticker_symbol]["long_shares"], 0, "Price Limit Test: Buy at limit_up_price should be rejected.")

        self._run_backtest_with_trades(bt_buy_reject, [("2023-01-03", ticker_symbol, "buy", 10, limit_up_price + 1.0)]) # Try to buy above limit
        self.assertEqual(bt_buy_reject.portfolio["positions"][ticker_symbol]["long_shares"], 0, "Price Limit Test: Buy above limit_up_price should be rejected.")

        # --- Test Sell Limits ---
        # Need shares first, bought on a day *before* 2023-01-02 to be sellable on 2023-01-03 due to T+1
        # This test is getting complicated due to state interactions (T+1, prev_close).
        # Let's simplify: Assume we have shares and test the limit rejection.
        bt_sell_reject = Backtester(agent=dummy_agent, tickers=[ticker_symbol], start_date="2023-01-03", end_date="2023-01-03", initial_capital=self.initial_capital, limit_up_ratio=1.1, limit_down_ratio=0.9)
        # Give it some shares manually, as if bought long ago
        bt_sell_reject.portfolio["positions"][ticker_symbol]["long_shares"] = 100
        bt_sell_reject.portfolio["positions"][ticker_symbol]["long_lots"].append({"date_acquired": "2022-12-01", "quantity": 100, "price_per_share": prev_close_price - 10})
        bt_sell_reject.portfolio["positions"][ticker_symbol]["long_cost_basis"] = prev_close_price -10

        self.mock_get_price_data.side_effect = custom_price_side_effect
        bt_sell_reject.previous_day_close_prices = {ticker_symbol: prev_close_price}


        self._run_backtest_with_trades(bt_sell_reject, [("2023-01-03", ticker_symbol, "sell", 10, limit_down_price)])
        self.assertEqual(bt_sell_reject.portfolio["positions"][ticker_symbol]["long_shares"], 100, "Price Limit Test: Sell at limit_down_price should be rejected.")

        self._run_backtest_with_trades(bt_sell_reject, [("2023-01-03", ticker_symbol, "sell", 10, limit_down_price - 1.0)])
        self.assertEqual(bt_sell_reject.portfolio["positions"][ticker_symbol]["long_shares"], 100, "Price Limit Test: Sell below limit_down_price should be rejected.")

        # Test successful sell just above limit down
        bt_sell_ok = Backtester(agent=dummy_agent, tickers=[ticker_symbol], start_date="2023-01-03", end_date="2023-01-03", initial_capital=self.initial_capital, limit_up_ratio=1.1, limit_down_ratio=0.9)
        bt_sell_ok.portfolio["positions"][ticker_symbol]["long_shares"] = 100
        bt_sell_ok.portfolio["positions"][ticker_symbol]["long_lots"].append({"date_acquired": "2022-12-01", "quantity": 100, "price_per_share": prev_close_price - 10})
        bt_sell_ok.portfolio["positions"][ticker_symbol]["long_cost_basis"] = prev_close_price -10
        self.mock_get_price_data.side_effect = custom_price_side_effect
        bt_sell_ok.previous_day_close_prices = {ticker_symbol: prev_close_price}

        self._run_backtest_with_trades(bt_sell_ok, [("2023-01-03", ticker_symbol, "sell", 10, limit_down_price + 0.01)])
        self.assertEqual(bt_sell_ok.portfolio["positions"][ticker_symbol]["long_shares"], 90, "Price Limit Test: Sell just above limit_down_price should succeed.")

    def test_trading_calendar_skips_holidays(self):
        """ Test Trading Calendar: 节假日闭市逻辑验证 """
        print("\nRunning test_trading_calendar_skips_holidays...")
        # Using 2024-04-28 (Sun) to 2024-05-05 (Sun)
        # Mock holidays include 2024-05-01 (Wed) as Labour Day
        # Expected trading days: 2024-04-29 (Mon), 2024-04-30 (Tue), 2024-05-02 (Thu), 2024-05-03 (Fri)
        start_date = "2024-04-28"
        end_date = "2024-05-05"

        # Add a specific holiday for this test to MOCK_HOLIDAYS_SSE if not already there for clarity
        # MOCK_HOLIDAYS_SSE already includes 2024-05-01

        bt = Backtester(
            agent=dummy_agent,
            tickers=["SH600519"],
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            exchange="SSE"
        )

        # We need to check which dates the agent was called for, or which dates appear in portfolio_values
        # Patch the agent to record call dates
        called_dates_for_agent = []
        def recording_agent(tickers, start_date, end_date, portfolio, model_name, model_provider, selected_analysts):
            called_dates_for_agent.append(end_date) # end_date is current_date_str in backtester
            return {"decisions": {}, "analyst_signals": {}}
        bt.agent = MagicMock(side_effect=recording_agent)

        bt.run_backtest()

        # Dates in portfolio_values are pd.Timestamp, convert called_dates_for_agent for comparison
        processed_dates_in_portfolio = [val['Date'].strftime("%Y-%m-%d") for val in bt.portfolio_values]

        # Expected trading days based on our mock calendar (Mon, Tue, Thu, Fri)
        # 2024-04-28 Sun - Skip
        # 2024-04-29 Mon - Trade
        # 2024-04-30 Tue - Trade
        # 2024-05-01 Wed - Holiday (Labour Day) - Skip
        # 2024-05-02 Thu - Trade
        # 2024-05-03 Fri - Trade
        # 2024-05-04 Sat - Skip
        # 2024-05-05 Sun - Skip
        expected_trading_days = ["2024-04-29", "2024-04-30", "2024-05-02", "2024-05-03"]

        # The agent is called with end_date = current_date_str.
        # The portfolio_values also records 'Date' which is the pandas Timestamp for current_date.
        # The first entry in portfolio_values is the initial capital day, so skip it for agent call comparison.

        # Check dates agent was called for (actual trading activity)
        self.assertEqual(len(called_dates_for_agent), len(expected_trading_days),
                         f"Agent called on wrong number of days. Expected {len(expected_trading_days)}, Got {len(called_dates_for_agent)}. Called: {called_dates_for_agent}")
        for d in expected_trading_days:
            self.assertIn(d, called_dates_for_agent, f"Expected trading day {d} not processed by agent.")

        # Check dates in portfolio history (includes initial day + trading days)
        # The first day in `dates` from `get_trading_calendar` is the first actual trading day.
        # `portfolio_values` has an initial entry, then one for each day in `dates`.
        # So, len(bt.portfolio_values) should be 1 (initial) + len(expected_trading_days)
        # if the first day of the range is a trading day and also the start_date of backtest.
        # Or more simply, check the dates recorded in portfolio_values (excluding the initial setup one if any).

        # The `dates` variable in `run_backtest` will be exactly `expected_trading_days`.
        # `portfolio_values` will have an initial state, then one entry per day in `dates`.
        # So `len(bt.portfolio_values)` should be `1 + len(expected_trading_days)` if `dates` is not empty.
        if expected_trading_days: # if there are any trading days in range
             self.assertEqual(len(bt.portfolio_values), len(expected_trading_days) +1, "Trading Calendar: Incorrect number of entries in portfolio_values")
             portfolio_recorded_dates = [pv['Date'].strftime('%Y-%m-%d') for pv in bt.portfolio_values[1:]] # Skip initial capital entry
             self.assertListEqual(portfolio_recorded_dates, expected_trading_days, "Trading Calendar: Mismatch in processed dates in portfolio_values")

    def test_dividend_and_split_processing(self):
        """ Test Dividend and Split Processing (分红除权处理) """
        print("\nRunning test_dividend_and_split_processing...")
        ticker_div_split = "SH600519" # Using Moutai from mock adjustment data

        # Mock data for SH600519 from MOCK_ADJUSTMENT_DATA:
        # "2023-06-29": {"type": "dividend", "cash_per_share": 25.911}
        # "2022-05-10": {"type": "split", "factor": 2.0}

        # Scenario:
        # 1. Buy shares before split.
        # 2. Day of split: Check share quantity, cost basis, and prev_day_close adjustment.
        # 3. Hold shares through dividend ex-date.
        # 4. Day of dividend: Check cash increase.

        start_date = "2022-05-09"
        end_date = "2023-06-30" # Cover both events

        bt = Backtester(
            agent=dummy_agent,
            tickers=[ticker_div_split],
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            exchange="SSE"
        )

        initial_buy_price = 200.0
        initial_buy_quantity = 100

        trades = [
            (start_date, ticker_div_split, "buy", initial_buy_quantity, initial_buy_price),
        ]

        # We need to ensure prices are returned for the trade date, split date, and dividend date
        # And that prev_day_close is correctly set up for the split date.

        split_ex_date = "2022-05-10"
        dividend_ex_date = "2023-06-29"

        # Store specific prices for key dates
        price_map = {
            start_date: initial_buy_price, # Buy price
            # Price on split ex-date (before open, for prev_close_price adjustment)
            # The prev_close for 2022-05-10 should be the close of 2022-05-09.
            # Let's assume close of 2022-05-09 is initial_buy_price
            split_ex_date: initial_buy_price / 2.0, # Market price post-split
            dividend_ex_date: (initial_buy_price / 2.0) - 25.911 # Market price post-dividend (approx)
        }

        def price_side_effect_for_events(ticker, date_from, date_to):
            # date_to is current_date_str
            dt_to_str = pd.to_datetime(date_to).strftime("%Y-%m-%d")

            # For trade execution specified in `trades` list
            for trade_date, trade_ticker, _, _, trade_price_override in trades:
                if trade_date == dt_to_str and trade_ticker == ticker:
                    dt_to = pd.to_datetime(dt_to_str)
                    return pd.DataFrame([{
                        "time": dt_to.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "open": trade_price_override, "high": trade_price_override,
                        "low": trade_price_override, "close": trade_price_override, "volume": 1000
                    }])

            # For other key dates, provide prices from price_map
            if dt_to_str in price_map:
                price = price_map[dt_to_str]
                dt_to = pd.to_datetime(dt_to_str)
                return pd.DataFrame([{
                    "time": dt_to.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "open": price, "high": price, "low": price, "close": price, "volume": 1000
                }])

            # Fallback if no specific price for this date
            # This is important for days between events
            fallback_price = initial_buy_price / 2.0 # Assume it hovers around post-split price
            dt_to = pd.to_datetime(dt_to_str)
            return pd.DataFrame([{
                    "time": dt_to.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "open": fallback_price, "high": fallback_price, "low": fallback_price, "close": fallback_price, "volume": 1000
                }])

        original_agent = bt.agent
        self.mock_get_price_data.side_effect = price_side_effect_for_events

        # Run the backtest with the initial buy
        self._run_backtest_with_trades(bt, trades)

        # Check state after split (2022-05-10)
        # Portfolio values are recorded at the end of each day.
        # We need to inspect the internal state of the portfolio *after* the split adjustment logic ran.
        # The _run_backtest_with_trades helper runs the whole period.
        # We will check the final state and infer intermediate states based on logic.

        final_portfolio = bt.portfolio
        pos_after_all = final_portfolio["positions"][ticker_div_split]

        # Expected after split (on 2022-05-10):
        # Shares: 100 * 2 = 200
        # Price per share in lot: 200.0 / 2 = 100.0
        # Avg cost basis: 100.0
        self.assertEqual(pos_after_all["long_shares"], initial_buy_quantity * 2, "Split: Incorrect share quantity")
        self.assertEqual(len(pos_after_all["long_lots"]), 1, "Split: Incorrect number of lots")
        if pos_after_all["long_lots"]:
            self.assertEqual(pos_after_all["long_lots"][0]["quantity"], initial_buy_quantity * 2, "Split: Lot quantity incorrect")
            self.assertAlmostEqual(pos_after_all["long_lots"][0]["price_per_share"], initial_buy_price / 2.0, places=2, msg="Split: Lot price incorrect")
        self.assertAlmostEqual(pos_after_all["long_cost_basis"], initial_buy_price / 2.0, places=2, msg="Split: Avg cost basis incorrect")

        # Expected after dividend (on 2023-06-29):
        # Dividend per share: 25.911
        # Shares held pre-dividend: 200
        # Cash received: 25.911 * 200 = 5182.2
        # Initial cash: 100000
        # Cost of initial buy: 100 * 200 = 20000
        # Cash after buy: 80000
        # Expected cash after dividend: 80000 + 5182.2 = 85182.2

        expected_cash_after_dividend = self.initial_capital - (initial_buy_quantity * initial_buy_price) + (25.911 * initial_buy_quantity * 2)
        self.assertAlmostEqual(final_portfolio["cash"], expected_cash_after_dividend, places=2, msg="Dividend: Incorrect cash balance")

        # Restore agent if it was changed locally for more complex scenarios later
        bt.agent = original_agent

    @patch('builtins.print') # Mock print to capture log messages
    def test_st_delisting_warnings(self, mock_print):
        """ Test ST/Delisting Warnings (ST康美退市预警测试) """
        print("\nRunning test_st_delisting_warnings...") # This print won't be mocked

        st_ticker = "ST_KANGMEI" # From MOCK_STOCK_INFO, status: "ST"
        risk_ticker = "POTENTIAL_DELIST" # D/E ratio will be set to exceed threshold
        normal_ticker = "SH600519"

        # Update MOCK_STOCK_INFO for this test if needed, e.g. to make POTENTIAL_DELIST trigger rule
        original_potential_delist_info = MOCK_STOCK_INFO.get(risk_ticker, {}).copy()
        MOCK_STOCK_INFO[risk_ticker]["debt_to_equity_ratio"] = 0.7 # Exceeds default 0.6 threshold

        bt = Backtester(
            agent=dummy_agent,
            tickers=[st_ticker, risk_ticker, normal_ticker],
            start_date="2023-01-02",
            end_date="2023-01-03", # Run for a couple of days
            initial_capital=self.initial_capital,
            delisting_risk_threshold=0.6 # Explicitly set for clarity
        )

        bt.run_backtest()

        # Check mock_print calls for expected warnings
        log_messages = " ".join([call_args[0][0] for call_args in mock_print.call_args_list if call_args[0]]) # Get all print arguments

        # Expected warning for ST_KANGMEI
        self.assertIn(f"WARNING for {st_ticker}: Status is ST", log_messages)

        # Expected warning for POTENTIAL_DELIST due to debt_to_equity_ratio
        expected_risk_msg_part = f"WARNING for {risk_ticker}: Debt/Equity ratio (0.70) exceeds threshold (0.60)"
        self.assertIn(expected_risk_msg_part, log_messages)

        # Ensure no such warnings for the normal ticker
        self.assertNotIn(f"WARNING for {normal_ticker}", log_messages)

        # Restore MOCK_STOCK_INFO if changed
        if original_potential_delist_info:
            MOCK_STOCK_INFO[risk_ticker] = original_potential_delist_info
        else: # If it didn't exist before, remove it
            if risk_ticker in MOCK_STOCK_INFO:
                 del MOCK_STOCK_INFO[risk_ticker]


if __name__ == '__main__':
    unittest.main()
