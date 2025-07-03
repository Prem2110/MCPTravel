import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)

# Load environment variables
load_dotenv()

@st.cache_resource
def setup_agent():
    client = MCPClient.from_config_file(
        os.path.join(os.path.dirname(__file__), "airbnb_mcp.json")
    )
    llm = ChatOpenAI(deployment_id=os.environ["LLM_DEPLOYMENT_ID"])
    agent = MCPAgent(llm=llm, client=client, max_steps=30, verbose=True)
    return agent, client

async def run_agent_query(agent, client, user_query):
    try:
        result = await agent.run(user_query, max_steps=30)
        return result
    finally:
        if client.sessions:
            await client.close_all_sessions()

# Async wrapper: compatible with running inside Streamlit
def run_async_task(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already inside an event loop (e.g., Streamlit), run with `ensure_future`
            return asyncio.ensure_future(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

# Streamlit UI
st.set_page_config(page_title="Airbnb Search with MCP", layout="centered")
st.title("🧭 Airbnb Search Agent")

query = st.text_input(
    "Enter your travel plan:",
    placeholder="e.g. Find me a nice place to stay in Chennai for 4 adults from July 1st to 4th"
)

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        agent, client = setup_agent()
        with st.spinner("Thinking..."):
            result_future = run_async_task(run_agent_query(agent, client, query))
            if isinstance(result_future, asyncio.Future):
                # Streamlit doesn't natively await asyncio Futures, so we need to wait for result synchronously
                result = asyncio.get_event_loop().run_until_complete(result_future)
            else:
                result = result_future
            st.success("Search completed!")
            st.markdown("### 🏡 Result:")
            st.write(result)
