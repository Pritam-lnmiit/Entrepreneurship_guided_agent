from fastapi import FastAPI, Body, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import io
import sys
import re
import os
import logging
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.email import EmailTools
from agno.tools.calculator import CalculatorTools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable not set.")
    sys.exit(1)
os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize FastAPI app
app = FastAPI(
    title="Dairy Business Agent API",
    description="An AI-powered API for dairy business operations using multiple specialized agents.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request and response
class QueryRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    response: str
    agent_name: str
    status: str
    error: Optional[str] = None

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
    role="Initiates and leads the business. Develops vision, strategy, and execution plan.",
    model=OpenAIChat(id="gpt-4o"),
    tools=[EmailTools(), DuckDuckGoTools()],
    show_tool_calls=True,
    instructions="Use tables to display data",
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Retrieve and analyze financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_info=True), CalculatorTools(), DuckDuckGoTools()],
    instructions="Calculate financial ratios and metrics. Include stock prices and financial data.",
    show_tool_calls=True,
    markdown=True,
)

market_sales_agent = Agent(
    name="Market & Sales Agent",
    role="Acts as a middleman between manufacturers and retailers. Ensures product availability and timely delivery.",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_info=True), DuckDuckGoTools()],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

r_d_agent = Agent(
    name="R&D Agent",
    role="Innovate and improve product designs. Conduct experiments and test feasibility.",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_info=True), DuckDuckGoTools()],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

supply_chain_agent = Agent(
    name="Supply Chain & Logistics Agent",
    role="Manage production, supply raw materials, and ensure timely delivery while reducing waste.",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

advisory_agent = Agent(
    name="Mentors & Advisors Agent",
    role="Provide business guidance, strategy, and industry knowledge.",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

# Create agent team
agent_team = Agent(
    team=[web_agent, finance_agent, supply_chain_agent, business_agent, market_sales_agent, r_d_agent, advisory_agent],
    model=OpenAIChat(id="gpt-4o"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Helper function to get agent response
def get_agent_response_string(agent: Agent, query: str) -> str:
    """
    Gets the agent response as a clean string.
    Tries agent.run() first, falls back to capturing print_response().
    """
    try:
        # Try direct method first
        if hasattr(agent, 'run') and callable(getattr(agent, 'run')):
            logger.info(f"Calling agent.run() for {agent.name}")
            response = agent.run(query)
            if isinstance(response, str):
                return response.strip()
            logger.warning(f"agent.run() for {agent.name} did not return a string, falling back.")

        # Fallback to capturing print_response
        logger.info(f"Capturing print_response for {agent.name}")
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        agent.print_response(query, stream=False)
        raw_response = new_stdout.getvalue()
        sys.stdout = old_stdout

        # Clean ANSI escape codes
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        cleaned_response = ansi_escape.sub('', raw_response).strip()
        return cleaned_response

    except Exception as e:
        logger.error(f"Error processing query for {agent.name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query for {agent.name}: {str(e)}")

# API Endpoints
@app.post("/api/v1/agent-response", response_model=AgentResponse)
async def get_team_response(request: QueryRequest):
    """Process a query through the agent team."""
    try:
        response = get_agent_response_string(agent_team, request.query)
        return AgentResponse(response=response, agent_name="Agent Team", status="success")
    except HTTPException as e:
        raise e
    except Exception as e:
        return AgentResponse(response="", agent_name="Agent Team", status="error", error=str(e))

@app.post("/api/v1/web-agent", response_model=AgentResponse)
async def web_agent_endpoint(request: QueryRequest):
    """Get response from the Web Agent."""
    response = get_agent_response_string(web_agent, request.query)
    return AgentResponse(response=response, agent_name="Web Agent", status="success")

@app.post("/api/v1/finance-agent", response_model=AgentResponse)
async def finance_agent_endpoint(request: QueryRequest):
    """Get response from the Finance Agent."""
    response = get_agent_response_string(finance_agent, request.query)
    return AgentResponse(response=response, agent_name="Finance Agent", status="success")

@app.post("/api/v1/business-agent", response_model=AgentResponse)
async def business_agent_endpoint(request: QueryRequest):
    """Get response from the Business Agent."""
    response = get_agent_response_string(business_agent, request.query)
    return AgentResponse(response=response, agent_name="Business Agent", status="success")

@app.post("/api/v1/market-sales-agent", response_model=AgentResponse)
async def market_sales_agent_endpoint(request: QueryRequest):
    """Get response from the Market & Sales Agent."""
    response = get_agent_response_string(market_sales_agent, request.query)
    return AgentResponse(response=response, agent_name="Market & Sales Agent", status="success")

@app.post("/api/v1/rd-agent", response_model=AgentResponse)
async def rd_agent_endpoint(request: QueryRequest):
    """Get response from the R&D Agent."""
    response = get_agent_response_string(r_d_agent, request.query)
    return AgentResponse(response=response, agent_name="R&D Agent", status="success")

@app.post("/api/v1/supply-chain-agent", response_model=AgentResponse)
async def supply_chain_agent_endpoint(request: QueryRequest):
    """Get response from the Supply Chain & Logistics Agent."""
    response = get_agent_response_string(supply_chain_agent, request.query)
    return AgentResponse(response=response, agent_name="Supply Chain & Logistics Agent", status="success")

@app.post("/api/v1/advisory-agent", response_model=AgentResponse)
async def advisory_agent_endpoint(request: QueryRequest):
    """Get response from the Mentors & Advisors Agent."""
    response = get_agent_response_string(advisory_agent, request.query)
    return AgentResponse(response=response, agent_name="Mentors & Advisors Agent", status="success")

@app.get("/api/v1/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "healthy"}

# Run the server
if __name__ == "__main__":
    logger.info("Starting Dairy Business Agent API on http://0.0.0.0:8000")
    logger.info("View API documentation at http://localhost:8000/docs")
    uvicorn.run(app, host="localhost", port=8000)