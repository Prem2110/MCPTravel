import os
import asyncio
import warnings
from dotenv import load_dotenv
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

warnings.filterwarnings("ignore", category=ResourceWarning)

# Load environment variables
load_dotenv()

def setup_agent():
    client = MCPClient.from_config_file(
        os.path.join(os.path.dirname(__file__), "airbnb_mcp.json")
    )
    llm = ChatOpenAI(deployment_id=os.environ["LLM_DEPLOYMENT_ID"])
    agent = MCPAgent(llm=llm, client=client, max_steps=30, verbose=False)
    return agent, client

async def run_agent_query(agent, client, user_query):
    try:
        result = await agent.run(user_query, max_steps=30)
        return result
    finally:
        if client.sessions:
            await client.close_all_sessions()

def main():
    print("🧭 Airbnb Search Agent")
    query = input("Enter your travel plan (e.g. Find me a nice place to stay in Chennai for 4 adults from July 11th to 14th, 2025):\n> ").strip()

    if not query:
        print("⚠️ Please enter a valid query.")
        return

    agent, client = setup_agent()
    print("Thinking...")

    result = asyncio.run(run_agent_query(agent, client, query))

    print("\n✅ Search completed!")
    print("🏡 Result:\n")
    print(result)

if __name__ == "__main__":
    main()
