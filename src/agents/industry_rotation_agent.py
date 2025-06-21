import random

class IndustryRotationAgent:
    def __init__(self, llm=None, data_source: str = "East Money / Tonghuashun"):
        self.llm = llm # Language Model, if needed for analysis
        self.data_source = data_source
        self.name = "Industry Rotation Agent"
        self.description = "Performs Shenwan Primary Industry Momentum Analysis and suggests TOP3 industries for allocation."

    def analyze_industry_momentum(self, industry_data: dict = None) -> dict:
        """
        Analyzes industry momentum based on provided data.
        In a real implementation, this would involve fetching and processing data from sources
        like East Money or Tonghuashun to calculate momentum scores for Shenwan primary industries.
        For now, it returns a dummy TOP3 industry suggestion.

        Args:
            industry_data (dict, optional): A dictionary containing industry performance data. Defaults to None.

        Returns:
            dict: A dictionary containing the agent's name, description, data_source and the 'top_3_industries'.
        """
        print(f"[{self.name}] Analyzing industry momentum from {self.data_source}...")
        if industry_data:
            print(f"[{self.name}] Received data for {len(industry_data)} industries.")

        # Placeholder: Simulate momentum analysis and selection of top 3 industries
        # Shenwan Primary Industries (example list, not exhaustive)
        shenwan_industries = [
            "计算机 (Computer)", "电子 (Electronics)", "医药生物 (Pharmaceuticals & Biotechnology)",
            "食品饮料 (Food & Beverage)", "电力设备 (Power Equipment)", "有色金属 (Non-ferrous Metals)",
            "基础化工 (Basic Chemicals)", "汽车 (Automotive)", "机械设备 (Machinery)",
            "国防军工 (Defense & Military)", "通信 (Telecommunications)", "传媒 (Media)",
            "银行 (Banking)", "非银金融 (Non-bank Financials)", "房地产 (Real Estate)",
            "建筑材料 (Construction Materials)", "建筑装饰 (Construction & Decoration)",
            "钢铁 (Steel)", "煤炭 (Coal)", "石油石化 (Oil & Petrochemicals)",
            "交通运输 (Transportation)", "农林牧渔 (Agriculture, Forestry, Animal Husbandry & Fishery)",
            "商贸零售 (Retail)", "社会服务 (Social Services)", "环保 (Environmental Protection)",
            "公用事业 (Utilities)", "纺织服饰 (Textiles & Apparel)", "轻工制造 (Light Manufacturing)",
            "美容护理 (Beauty & Personal Care)", "综合 (Comprehensive)"
        ]

        # Simulate selecting 3 random industries as "top"
        if len(shenwan_industries) >= 3:
            top_3_industries = random.sample(shenwan_industries, 3)
        else:
            top_3_industries = shenwan_industries # Should not happen with the above list

        print(f"[{self.name}] Top 3 Industries for allocation: {', '.join(top_3_industries)}")

        return {
            "agent_name": self.name,
            "description": self.description,
            "data_source": self.data_source,
            "top_3_industries": top_3_industries
        }

# from src.graph.state import AgentState, show_agent_reasoning
# from src.utils.llm import get_llm

def industry_rotation_agent_node(state: dict) -> dict:
    """
    Wrapper function for the Industry Rotation Agent to be used as a node in LangGraph.
    """
    # print("---INDUSTRY ROTATION AGENT NODE---")
    data = state.get("data", {})
    metadata = state.get("metadata", {})

    # llm_instance = get_llm(model_name=metadata.get("model_name"), provider=metadata.get("model_provider"))
    agent = IndustryRotationAgent(llm=None) # LLM can be passed if agent uses it

    # Industry data could be fetched or passed in data if available
    # For this placeholder, analyze_industry_momentum generates dummy data internally
    analysis_output = agent.analyze_industry_momentum()

    if metadata.get("show_reasoning", False):
        # show_agent_reasoning(analysis_output, agent.name)
        print(f"\nReasoning from {agent.name}:")
        import json
        print(json.dumps(analysis_output, indent=2))

    current_signals = data.get("analyst_signals", {})
    updated_signals = {**current_signals, "industry_rotation": analysis_output}

    return {"data": {**data, "analyst_signals": updated_signals}}

if __name__ == '__main__':
    # Example Usage of the agent itself
    industry_agent_instance = IndustryRotationAgent()
    example_data = {
        "计算机": {"momentum_score": 0.85},
        "医药生物": {"momentum_score": 0.72},
        "食品饮料": {"momentum_score": 0.91}
    }
    analysis_result_direct = industry_agent_instance.analyze_industry_momentum(industry_data=example_data)
    print("\nDirect Industry Rotation Analysis Result:")
    import json
    print(json.dumps(analysis_result_direct, indent=2))

    general_analysis_result_direct = industry_agent_instance.analyze_industry_momentum()
    print("\nGeneral Industry Rotation Analysis Result (Direct, no specific data):")
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
    node_output = industry_rotation_agent_node(mock_state)
    print("\nIndustry Rotation Agent Node Output:")
    print(json.dumps(node_output, indent=2))
