import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.email import EmailTools
from agno.tools.calculator import CalculatorTools
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define agents
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources",
    show_tool_calls=True,
    markdown=True,
)

business_agent = Agent(
    name="Business Agent",
    role="Leads business strategy and execution",
    model=OpenAIChat(id="gpt-4o"),
    tools=[EmailTools(), DuckDuckGoTools()],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_info=True), CalculatorTools(), DuckDuckGoTools()],
    instructions="Calculate financial ratios and metrics and get stock prices and financial data.",
    show_tool_calls=True,
    markdown=True,
)

market_sales_agent = Agent(
    name="Market & Sales Agent",
    role="Handles product availability and delivery",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_info=True), DuckDuckGoTools()],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

r_d_agent = Agent(
    name="R&D Agent",
    role="Innovates and improves product designs",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

supply_chain_agent = Agent(
    name="Manufacturing & Supply Chain Agent",
    role="Handles production and supply chain logistics",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

support_agent = Agent(
    name="Mentors & Advisors",
    role="Provide business guidance and strategy",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

# Create agent team
agent_team = Agent(
    team=[web_agent, finance_agent, supply_chain_agent, business_agent, market_sales_agent, r_d_agent, support_agent],
    model=OpenAIChat(id="gpt-4o"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit UI
st.title("Dairy Products Business Setup Assistant")
st.write("Enter your query to get detailed insights into starting a dairy business in India.")

query = st.text_area("Enter your question:", "I want to build a company related to dairy products in India. Give me all the details for my company.")

if st.button("Get Details"):
    with st.spinner("Fetching details..."):
        response = agent_team.print_response(query, stream=False)
        st.markdown(response)
