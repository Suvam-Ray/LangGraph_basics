from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None
from dotenv import load_dotenv
import aiosqlite
import asyncio
import json
import sqlite3
import requests
import threading

load_dotenv()

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

def submit_async_task(coro):
    return _submit_async(coro)

llm = ChatGroq(model="qwen/qwen3-32b")

try:
    search_tool = DuckDuckGoSearchRun(region="us-en")
except ImportError:
    search_tool = None

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation: add, sub, mul, div."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol, e.g. AAPL or TSLA."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

client = MultiServerMCPClient(
    {
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
        }
    }
) if MultiServerMCPClient else None

def load_mcp_tools() -> list[BaseTool]:
    if client is None:
        return []
    future = None
    try:
        future = _submit_async(client.get_tools())
        return future.result(timeout=10)
    except Exception:
        if future:
            future.cancel()
        return []

mcp_tools = load_mcp_tools()
tools = [tool for tool in [search_tool, get_stock_price, calculator, *mcp_tools] if tool is not None]
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def normalize_tool_messages(messages):
    normalized_messages = []
    for message in messages:
        if isinstance(message, ToolMessage) and not isinstance(message.content, str):
            message = ToolMessage(
                content=json.dumps(message.content),
                tool_call_id=message.tool_call_id,
                name=message.name,
            )
        normalized_messages.append(message)
    return normalized_messages

async def chat_node(state: ChatState):
    messages = normalize_tool_messages(state['messages'])
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS thread_titles (
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL
)
""")
conn.commit()
# Checkpointer
async def _init_checkpointer():
    async_conn = await aiosqlite.connect(database='chatbot.db')
    return AsyncSqliteSaver(async_conn)

checkpointer = run_async(_init_checkpointer())

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

async def _alist_threads():
    rows = conn.execute("""
        SELECT thread_id
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY MAX(checkpoint_id)
    """).fetchall()
    return [row[0] for row in rows]

def retrieve_all_threads():
    return run_async(_alist_threads())

def save_thread_title(thread_id, title):
    conn.execute(
        """
        INSERT INTO thread_titles (thread_id, title)
        VALUES (?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET title = excluded.title
        """,
        (str(thread_id), title)
    )
    conn.commit()

def retrieve_all_thread_titles():
    rows = conn.execute("SELECT thread_id, title FROM thread_titles").fetchall()
    return {thread_id: title for thread_id, title in rows}

def delete_thread(thread_id):
    thread_id = str(thread_id)
    conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    conn.execute("DELETE FROM thread_titles WHERE thread_id = ?", (thread_id,))
    conn.commit()
