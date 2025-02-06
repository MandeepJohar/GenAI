import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os 
import phi 
from phi.playground import Playground, serve_playground_app
load_dotenv()

#Get API key from environment
openai.api.key=os.environ("OPENAI_API_KEY")
phi.ap = os.environ("PHI_APP_KEY")

#1st Agent 
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
    )


#2nd Agent 
finance_agent = Agent(
    name="Finance AI Agent",
    model=OpenAiChat(id="gpt-4o"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
            key_financial_ratios=True
        )
    ],
    instructions=["Use table to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Combining 2 agents using Playground
app = Playground(agents=[finance_agent, web_search_agent]).getapp()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
