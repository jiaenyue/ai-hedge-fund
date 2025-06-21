import unittest
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.agents.policy_analysis_agent import PolicyAnalysisAgent, policy_analysis_agent_node
from src.agents.industry_rotation_agent import IndustryRotationAgent, industry_rotation_agent_node
from src.agents.northbound_capital_agent import NorthboundCapitalAgent, northbound_capital_agent_node
from src.agents.st_stock_agent import STStockAgent, st_stock_agent_node

class TestAShareAgents(unittest.TestCase):

    def _create_mock_state(self, tickers=None, show_reasoning=False):
        if tickers is None:
            tickers = ["SH600000", "SZ000001"] # Default tickers
        return {
            "data": {
                "tickers": tickers,
                "portfolio": {}, # Simplified
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "analyst_signals": {}
            },
            "metadata": {
                "show_reasoning": show_reasoning,
                "model_name": "test_model",
                "model_provider": "test_provider"
            },
            "messages": [] # Simplified
        }

    def test_policy_analysis_agent_direct(self):
        agent = PolicyAnalysisAgent()
        result = agent.analyze_policies()
        self.assertIn("policy_heat_index", result)
        self.assertIsInstance(result["policy_heat_index"], int)
        self.assertTrue(0 <= result["policy_heat_index"] <= 100)
        self.assertEqual(result["agent_name"], "Policy Analysis Agent")

    def test_policy_analysis_agent_node(self):
        mock_state = self._create_mock_state()
        result_state = policy_analysis_agent_node(mock_state)
        self.assertIn("data", result_state)
        self.assertIn("analyst_signals", result_state["data"])
        self.assertIn("policy_analysis", result_state["data"]["analyst_signals"])
        policy_output = result_state["data"]["analyst_signals"]["policy_analysis"]
        self.assertIn("policy_heat_index", policy_output)

    def test_industry_rotation_agent_direct(self):
        agent = IndustryRotationAgent()
        result = agent.analyze_industry_momentum()
        self.assertIn("top_3_industries", result)
        self.assertIsInstance(result["top_3_industries"], list)
        self.assertEqual(len(result["top_3_industries"]), 3)
        self.assertEqual(result["agent_name"], "Industry Rotation Agent")

    def test_industry_rotation_agent_node(self):
        mock_state = self._create_mock_state()
        result_state = industry_rotation_agent_node(mock_state)
        self.assertIn("data", result_state)
        self.assertIn("analyst_signals", result_state["data"])
        self.assertIn("industry_rotation", result_state["data"]["analyst_signals"])
        industry_output = result_state["data"]["analyst_signals"]["industry_rotation"]
        self.assertIn("top_3_industries", industry_output)

    def test_northbound_capital_agent_direct(self):
        agent = NorthboundCapitalAgent()
        result = agent.monitor_capital_flow()
        self.assertIn("warning_signal", result)
        self.assertIsInstance(result["warning_signal"], str)
        self.assertIn("is_warning_triggered", result)
        self.assertIsInstance(result["is_warning_triggered"], bool)
        self.assertEqual(result["agent_name"], "Northbound Capital Agent")

    def test_northbound_capital_agent_node(self):
        mock_state = self._create_mock_state()
        result_state = northbound_capital_agent_node(mock_state)
        self.assertIn("data", result_state)
        self.assertIn("analyst_signals", result_state["data"])
        self.assertIn("northbound_capital", result_state["data"]["analyst_signals"])
        nb_output = result_state["data"]["analyst_signals"]["northbound_capital"]
        self.assertIn("warning_signal", nb_output)

    def test_st_stock_agent_direct(self):
        agent = STStockAgent()
        # Test with an ST stock
        st_stock = "ST600000"
        result_st = agent.assess_delisting_risk(stock_symbol=st_stock)
        self.assertIn("delisting_risk_score", result_st)
        self.assertIsInstance(result_st["delisting_risk_score"], float)
        self.assertTrue(0.0 <= result_st["delisting_risk_score"] <= 1.0)
        self.assertIn("cap_removal_probability", result_st)
        self.assertIsInstance(result_st["cap_removal_probability"], float)
        self.assertTrue(0.0 <= result_st["cap_removal_probability"] <= 1.0)
        self.assertEqual(result_st["stock_symbol_analyzed"], st_stock)
        self.assertEqual(result_st["agent_name"], "ST Stock Agent")

        # Test with a non-ST stock (agent should still run and note it)
        non_st_stock = "SH600001"
        result_non_st = agent.assess_delisting_risk(stock_symbol=non_st_stock)
        self.assertEqual(result_non_st["stock_symbol_analyzed"], non_st_stock)
        # The agent itself prints a warning, but the calculation proceeds with dummy values.

    def test_st_stock_agent_node_with_st_stock(self):
        mock_state = self._create_mock_state(tickers=["ST600123", "SH600000"])
        result_state = st_stock_agent_node(mock_state)
        self.assertIn("data", result_state)
        self.assertIn("analyst_signals", result_state["data"])
        self.assertIn("st_stock_analysis", result_state["data"]["analyst_signals"])
        st_output = result_state["data"]["analyst_signals"]["st_stock_analysis"]
        self.assertIn("delisting_risk_score", st_output)
        self.assertEqual(st_output["stock_symbol_analyzed"], "ST600123")

    def test_st_stock_agent_node_without_st_stock(self):
        # Node should pick the first ticker if no ST stock is found
        mock_state = self._create_mock_state(tickers=["SH600000", "SZ000001"])
        result_state = st_stock_agent_node(mock_state)
        self.assertIn("data", result_state)
        self.assertIn("analyst_signals", result_state["data"])
        self.assertIn("st_stock_analysis", result_state["data"]["analyst_signals"])
        st_output = result_state["data"]["analyst_signals"]["st_stock_analysis"]
        self.assertIn("delisting_risk_score", st_output)
        self.assertEqual(st_output["stock_symbol_analyzed"], "SH600000")

    def test_st_stock_agent_node_no_tickers(self):
        mock_state = self._create_mock_state(tickers=[])
        result_state = st_stock_agent_node(mock_state)
        self.assertIn("data", result_state)
        self.assertIn("analyst_signals", result_state["data"])
        self.assertIn("st_stock_analysis", result_state["data"]["analyst_signals"])
        st_output = result_state["data"]["analyst_signals"]["st_stock_analysis"]
        self.assertIn("error", st_output)
        self.assertIsNone(st_output["stock_symbol_analyzed"])

if __name__ == "__main__":
    unittest.main()
