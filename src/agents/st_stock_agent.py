import random

class STStockAgent:
    def __init__(self, llm=None, data_source: str = "Financial reports / Inquiry letters / Pledge data"):
        self.llm = llm # Language Model, if needed for analyzing text data like inquiry letters
        self.data_source = data_source
        self.name = "ST Stock Agent"
        self.description = "Implements a quantitative model to assess delisting risk for ST stocks and estimates ST '摘帽' (delisting cap removal) probability."

    def assess_delisting_risk(self, stock_symbol: str, financial_data: dict = None, inquiry_letters: list[str] = None, pledge_data: dict = None) -> dict:
        """
        Assesses the delisting risk for a given ST stock.
        In a real implementation, this would involve fetching financial reports, analyzing inquiry letters (possibly with NLP),
        and evaluating shareholder pledge data to feed into a quantitative risk model.
        For now, it returns a dummy delisting probability and '摘帽' (cap removal) probability.

        Args:
            stock_symbol (str): The symbol of the ST stock to analyze (e.g., "ST600000").
            financial_data (dict, optional): Parsed financial data for the stock.
            inquiry_letters (list[str], optional): Content of relevant inquiry letters.
            pledge_data (dict, optional): Data on major shareholder pledges.

        Returns:
            dict: A dictionary containing the agent's name, description, data_source, analyzed stock symbol,
                  'delisting_risk_score' (0-1, higher is riskier), and 'cap_removal_probability' (0-1).
        """
        if not stock_symbol or not stock_symbol.upper().startswith(("ST", "*ST")):
            # Adding *ST as it's another common prefix for higher-risk stocks
            print(f"[{self.name}] Warning: Stock symbol '{stock_symbol}' does not appear to be an ST or *ST stock. Analysis might be less relevant.")
            # return {
            #     "agent_name": self.name,
            #     "error": f"Stock symbol '{stock_symbol}' is not a valid ST stock format."
            # }

        print(f"[{self.name}] Assessing delisting risk for ST stock: {stock_symbol} using data from {self.data_source}...")
        if financial_data:
            print(f"[{self.name}] Using provided financial data: {list(financial_data.keys())}")
        if inquiry_letters:
            print(f"[{self.name}] Analyzing {len(inquiry_letters)} inquiry letters.")
        if pledge_data:
            print(f"[{self.name}] Considering shareholder pledge data: {list(pledge_data.keys())}")

        # Placeholder: Simulate quantitative modeling
        # Factors that might increase delisting risk / decrease cap removal probability:
        # - Negative net profit for multiple consecutive years
        # - Negative net assets
        # - Audit opinion issues (e.g., qualified opinion, disclaimer)
        # - High level of shareholder pledges, especially by controlling shareholders
        # - Regulatory inquiries indicating serious issues

        # Simulate a risk score and cap removal probability
        delisting_risk_score = random.uniform(0.05, 0.95) # 0 = no risk, 1 = certain delisting

        # Cap removal probability is often inversely related to delisting risk, but not perfectly.
        # A stock might have moderate delisting risk but also a moderate chance of improving.
        cap_removal_probability = random.uniform(max(0.0, 0.1 - delisting_risk_score/2), min(1.0, 1.0 - delisting_risk_score/1.5 + random.uniform(-0.2,0.2)))
        cap_removal_probability = max(0.0, min(1.0, cap_removal_probability)) # Ensure it's within [0,1]

        print(f"[{self.name}] Delisting Risk Score for {stock_symbol}: {delisting_risk_score:.2f}")
        print(f"[{self.name}] '摘帽' (Cap Removal) Probability for {stock_symbol}: {cap_removal_probability:.2f}")

        return {
            "agent_name": self.name,
            "description": self.description,
            "data_source": self.data_source,
            "stock_symbol_analyzed": stock_symbol,
            "delisting_risk_score": round(delisting_risk_score, 3),
            "cap_removal_probability": round(cap_removal_probability, 3)
        }

# from src.graph.state import AgentState, show_agent_reasoning
# from src.utils.llm import get_llm

def st_stock_agent_node(state: dict) -> dict:
    """
    Wrapper function for the ST Stock Agent to be used as a node in LangGraph.
    This agent analyzes a specific stock, so it needs a ticker.
    """
    # print("---ST STOCK AGENT NODE---")
    data = state.get("data", {})
    metadata = state.get("metadata", {})

    tickers = data.get("tickers", [])
    stock_to_analyze = None
    analysis_output = None

    # Attempt to find an ST or *ST stock from the list of tickers
    # In a real scenario, this agent might be triggered only if relevant stocks are present,
    # or it might iterate through all ST/*ST stocks. For simplicity, we pick the first.
    for ticker in tickers:
        if ticker.upper().startswith(("ST", "*ST")):
            stock_to_analyze = ticker
            break

    if not stock_to_analyze and tickers: # Fallback to first ticker if no ST stock, agent will warn
        stock_to_analyze = tickers[0]
    elif not tickers:
        print(f"[{STStockAgent().name}] No tickers provided for analysis.")
        analysis_output = {
            "agent_name": STStockAgent().name,
            "error": "No tickers provided for ST stock analysis.",
            "stock_symbol_analyzed": None,
            "delisting_risk_score": None,
            "cap_removal_probability": None
        }

    if stock_to_analyze:
        # llm_instance = get_llm(model_name=metadata.get("model_name"), provider=metadata.get("model_provider"))
        agent = STStockAgent(llm=None) # LLM can be passed if agent uses it

        # Financial data, inquiry letters, pledge data could be fetched based on stock_to_analyze
        # or passed in data if available for that specific stock.
        # For this placeholder, assess_delisting_risk generates dummy data.
        analysis_output = agent.assess_delisting_risk(stock_symbol=stock_to_analyze)

    if metadata.get("show_reasoning", False) and analysis_output:
        # show_agent_reasoning(analysis_output, agent.name if 'agent' in locals() else STStockAgent().name)
        print(f"\nReasoning from {analysis_output.get('agent_name', STStockAgent().name)}:")
        import json
        print(json.dumps(analysis_output, indent=2))

    current_signals = data.get("analyst_signals", {})
    # Store results under a specific key, perhaps per-stock if analyzing multiple STs,
    # or a general key if only one is focused on.
    # For now, using a generic key, assuming one primary ST stock or a general ST market view.
    updated_signals = {**current_signals, "st_stock_analysis": analysis_output}

    return {"data": {**data, "analyst_signals": updated_signals}}


if __name__ == '__main__':
    # Example Usage of the agent itself
    st_agent_instance = STStockAgent()

    stock_symbol_high_risk = "ST600123"
    financials_hr = {"net_profit_yoy": -0.5, "net_assets": -10000000, "audit_opinion": "Qualified"}
    letters_hr = ["Inquiry regarding sustained losses and debt repayment capabilities."]
    pledges_hr = {"controlling_shareholder_pledge_ratio": 0.9}
    analysis_result_hr_direct = st_agent_instance.assess_delisting_risk(
        stock_symbol_high_risk, financial_data=financials_hr, inquiry_letters=letters_hr, pledge_data=pledges_hr
    )
    print(f"\nDirect ST Stock Analysis Result for {stock_symbol_high_risk}:")
    import json
    print(json.dumps(analysis_result_hr_direct, indent=2))

    stock_symbol_hopeful = "*ST600456"
    analysis_result_h_direct = st_agent_instance.assess_delisting_risk(stock_symbol_hopeful)
    print(f"\nDirect ST Stock Analysis Result for {stock_symbol_hopeful}:")
    print(json.dumps(analysis_result_h_direct, indent=2))

    # Example of how the node function might be called
    mock_state_st_stock = {
        "data": {
            "tickers": ["ST600888", "SZ000001", "*ST000123"],
            "portfolio": {}, "start_date": "2023-01-01", "end_date": "2023-12-31", "analyst_signals": {}
        },
        "metadata": {"show_reasoning": True, "model_name": "gpt-4", "model_provider": "OpenAI"},
        "messages": []
    }
    node_output_st = st_stock_agent_node(mock_state_st_stock)
    print("\nST Stock Agent Node Output (ST stock found):")
    print(json.dumps(node_output_st, indent=2))

    mock_state_no_st_stock = {
        "data": {
            "tickers": ["SH600000", "SZ000001"],
            "portfolio": {}, "start_date": "2023-01-01", "end_date": "2023-12-31", "analyst_signals": {}
        },
        "metadata": {"show_reasoning": True, "model_name": "gpt-4", "model_provider": "OpenAI"},
        "messages": []
    }
    node_output_no_st = st_stock_agent_node(mock_state_no_st_stock)
    print("\nST Stock Agent Node Output (No ST stock found, uses first ticker):")
    print(json.dumps(node_output_no_st, indent=2))

    mock_state_no_tickers = {
        "data": {
            "tickers": [],
            "portfolio": {}, "start_date": "2023-01-01", "end_date": "2023-12-31", "analyst_signals": {}
        },
        "metadata": {"show_reasoning": True, "model_name": "gpt-4", "model_provider": "OpenAI"},
        "messages": []
    }
    node_output_no_tickers = st_stock_agent_node(mock_state_no_tickers)
    print("\nST Stock Agent Node Output (No tickers):")
    print(json.dumps(node_output_no_tickers, indent=2))
