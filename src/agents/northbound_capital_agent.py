import random

class NorthboundCapitalAgent:
    def __init__(self, llm=None, data_source: str = "HKEX / ChinaClear"):
        self.llm = llm # Language Model, if needed for interpreting flow data
        self.data_source = data_source
        self.name = "Northbound Capital Agent"
        self.description = "Monitors Shanghai-Shenzhen-Hong Kong Stock Connect capital flows and issues warnings for abnormal movements."

    def monitor_capital_flow(self, flow_data: dict = None) -> dict:
        """
        Monitors northbound capital flow data (沪深港通资金流向).
        In a real implementation, this would involve fetching and analyzing data from HKEX/ChinaClear
        to identify significant inflows, outflows, or changes in holding patterns.
        For now, it returns a dummy warning signal.

        Args:
            flow_data (dict, optional): A dictionary containing capital flow data. Defaults to None.
                                        Example: {"SH": {"inflow_ млрд": 1.5, "holding_change_%": 0.2},
                                                  "SZ": {"inflow_ млрд": -0.5, "holding_change_%": -0.1}}

        Returns:
            dict: A dictionary containing the agent's name, description, data_source and the 'warning_signal'.
        """
        print(f"[{self.name}] Monitoring northbound capital flow from {self.data_source}...")
        if flow_data:
            print(f"[{self.name}] Received flow data: {flow_data}")

        # Placeholder: Simulate analysis of capital flow and generating a warning
        possible_signals = [
            "No significant abnormal movements detected.",
            "High net inflow into Shanghai Stock Exchange.",
            "Sustained outflow from Shenzhen Stock Exchange.",
            "Unusual concentration in specific sectors/stocks.",
            "Foreign capital sentiment appears bullish.",
            "Foreign capital sentiment appears bearish."
        ]

        warning_signal = random.choice(possible_signals)
        is_warning = not ("No significant" in warning_signal or "bullish" in warning_signal) # Crude way to determine if it's a "warning"

        print(f"[{self.name}] Warning Signal: {warning_signal}")

        return {
            "agent_name": self.name,
            "description": self.description,
            "data_source": self.data_source,
            "warning_signal": warning_signal,
            "is_warning_triggered": is_warning
        }

# from src.graph.state import AgentState, show_agent_reasoning
# from src.utils.llm import get_llm

def northbound_capital_agent_node(state: dict) -> dict:
    """
    Wrapper function for the Northbound Capital Agent to be used as a node in LangGraph.
    """
    # print("---NORTHBOUND CAPITAL AGENT NODE---")
    data = state.get("data", {})
    metadata = state.get("metadata", {})

    # llm_instance = get_llm(model_name=metadata.get("model_name"), provider=metadata.get("model_provider"))
    agent = NorthboundCapitalAgent(llm=None) # LLM can be passed if agent uses it

    # Capital flow data could be fetched or passed in data if available
    # For this placeholder, monitor_capital_flow generates dummy data internally
    analysis_output = agent.monitor_capital_flow()

    if metadata.get("show_reasoning", False):
        # show_agent_reasoning(analysis_output, agent.name)
        print(f"\nReasoning from {agent.name}:")
        import json
        print(json.dumps(analysis_output, indent=2))

    current_signals = data.get("analyst_signals", {})
    updated_signals = {**current_signals, "northbound_capital": analysis_output}

    return {"data": {**data, "analyst_signals": updated_signals}}

if __name__ == '__main__':
    # Example Usage of the agent itself
    northbound_agent_instance = NorthboundCapitalAgent()
    example_sh_inflow_data = {
        "SH": {"inflow_CNY_billion": 2.5, "top_traded_stocks": ["KWEICHOW MOUTAI", "PING AN INSURANCE"]},
        "SZ": {"inflow_CNY_billion": 1.8, "top_traded_stocks": ["CONTEMPORARY AMPEREX TECHNOLOGY", "WULIANGYE YIBIN"]}
    }
    analysis_result_sh_direct = northbound_agent_instance.monitor_capital_flow(flow_data=example_sh_inflow_data)
    print("\nDirect Northbound Capital Flow Analysis Result (Example 1):")
    import json
    print(json.dumps(analysis_result_sh_direct, indent=2))

    example_sz_outflow_data = {
        "SH": {"inflow_CNY_billion": 0.1, "top_traded_stocks": ["CHINA MERCHANTS BANK", "LONGI GREEN ENERGY TECHNOLOGY"]},
        "SZ": {"inflow_CNY_billion": -1.2, "top_traded_stocks": ["BYD COMPANY", "LUXSHARE PRECISION INDUSTRY"]}
    }
    analysis_result_sz_direct = northbound_agent_instance.monitor_capital_flow(flow_data=example_sz_outflow_data)
    print("\nDirect Northbound Capital Flow Analysis Result (Example 2):")
    print(json.dumps(analysis_result_sz_direct, indent=2))

    general_analysis_result_direct = northbound_agent_instance.monitor_capital_flow()
    print("\nGeneral Northbound Capital Flow Analysis (Direct, no specific data):")
    print(json.dumps(general_analysis_result_direct, indent=2))

    # Example of how the node function might be called (simplified state)
    mock_state = {
        "data": {
            "tickers": ["SH600000", "SZ000001"], # Not directly used by this general agent
            "portfolio": {},
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "analyst_signals": {}
        },
        "metadata": {
            "show_reasoning": True,
            "model_name": "gpt-4",
            "model_provider": "OpenAI"
        },
        "messages": []
    }
    node_output = northbound_capital_agent_node(mock_state)
    print("\nNorthbound Capital Agent Node Output:")
    print(json.dumps(node_output, indent=2))
