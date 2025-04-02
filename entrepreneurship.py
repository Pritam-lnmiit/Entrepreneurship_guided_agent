from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

from agno.tools.email import EmailTools
from agno.tools.calculator import CalculatorTools


import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources",
    show_tool_calls=True,
    markdown=True,
)
Business_agent = Agent(
    name="Business Agents",
    role="Initiates and leads the business and Develops the vision, strategy, and execution plan.",
    model=OpenAIChat(id="gpt-4o"),
    
    tools=[
        EmailTools(),
        DuckDuckGoTools(),
       
    ],
    show_tool_calls=True,

    instructions="Use tables to display data",

    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,stock_fundamentals=True,company_info=True),CalculatorTools(),DuckDuckGoTools()],
    instructions="Use to calculate financial ratios and metrics and to get stock prices and financial data.",
    show_tool_calls=True,
    markdown=True,
)

Market_Sales_agent = Agent(
    name=" Market & Sales Agents",
    role="Acts as a middleman between manufacturers and retailers and Ensures product availability and timely delivery.",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,stock_fundamentals=True,company_info=True),DuckDuckGoTools()],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

R_D_agent = Agent(
    name="R&D Teams",
    role="Innovate and improve product designs,Conduct experiments and test product feasibility and Work on technological advancements for the business.",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,stock_fundamentals=True,company_info=True),DuckDuckGoTools()],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

SupplyChain_Logistics_agents = Agent(
    name="Manufacturers & Suppliers",
    role="Produce goods and supply raw materials,Manage production schedules and quality control and Ensure timely delivery of productsa also Reduce waste and improve efficiency.",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)
Support_Advisory_agents = Agent(
    name="Mentors & Advisors",
    role="Provide business guidance and strategy.Share industry knowledge and experiences and Assist in decision-making.",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)


agent_team = Agent(
    team=[web_agent, finance_agent,SupplyChain_Logistics_agents ,Business_agent,Market_Sales_agent,R_D_agent,Support_Advisory_agents],
    model=OpenAIChat(id="gpt-4o"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

agent_team.print_response("i want to built a company related to dairy products in india give me all tha details for my company", stream=True)