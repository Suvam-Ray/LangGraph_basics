from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    client=MultiServerMCPClient(
        {
            "math":{
                "command":"python",
                "args":[r"module_19_1c_MCP_additional\math_server.py"], ## Ensure correct absolute path
                "transport":"stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",  # Ensure server is running here
                "transport": "streamable_http",
            }
        }
    )

    tools=await client.get_tools()
    model=ChatGroq(model="llama-3.3-70b-versatile")
    agent=create_react_agent(
        model,tools,prompt="Call one tool at a time. Never nest tool calls or use the output of one tool as an argument to another. Wait for each tool result before making the next call."
    )

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) * 12"}]}
    )

    print("Math response:", math_response['messages'][-1].content)

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in California?"}]}
    )
    
    print("Weather response:", weather_response['messages'][-1].content)

asyncio.run(main())
