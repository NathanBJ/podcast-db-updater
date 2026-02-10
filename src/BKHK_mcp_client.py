#TODO: Import Agent, MCPClient and streamablehttp_client from the strands library
import os
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.tools.mcp.mcp_client import MCPClient
from strands.models.mistral import MistralModel

def main():
    MistralLabModel = MistralModel(
        api_key=os.getenv("MISTRAL_API_KEY"),
        # **model_config
        model_id="mistral-large-latest",
    )
    # Connect to the dice roll MCP server
    print("\nConnecting to BkHK MCP Server...")
    # TODO: Create a streamable http MCPClient connecting to "http://localhost:8080/mcp"
    def create_streamable_http_transport():
        return streamablehttp_client("http://localhost:8080/mcp")
    streamable_http_mcp_client = MCPClient(create_streamable_http_transport)
    try:
        # TODO: Use the MCP client in a context manager (with statement)
        with streamable_http_mcp_client:
            # TODO: Get available tools from MCP server using list_tools_sync()
            mcp_tools = streamable_http_mcp_client.list_tools_sync()
            print(f"Available tools: {[tool.tool_name for tool in mcp_tools]}")

            DESCRIPTION="""
            Specialized childhood and parenthood agent that provides clear explainations and advices for the parents to understand their child and their associated problems. 
            Queries the ChromaDB knowledge base containing indexed BKHK podcast content and returns clear, pedagological, podcast-referenced solutions for the parents to solves their problems. If you do not know the answer, just say to ask the BKHK coaching sessions.
            """

            SYSTEM_PROMPT="""
            You are an early childhood specialist. When asked about parenthood and child behavior, use the query_podcast_db tool once to find the relevant advices from the podcast knowledge base, 
            then provide a clear, concise answer with the source reference. Keep responses brief and focused on the specific childhood and parenthood questions. Always ends with the spotify url reference and redirect to BKHK coaching sessions. 
            """

            agent = Agent(
                # TODO: Configure the agent with:
                # - model: Optional
                # - tools: List containing the query_podcast_advices tool
                # - name: "BKHK teacher Agent"
                model=MistralLabModel,
                description= DESCRIPTION,
                system_prompt= SYSTEM_PROMPT,
                name="BKHK Podcast Teacher Agent",
                tools=[mcp_tools[0]]
            )
                        
            # Start interactive session
            print("\nHi I am your BKHK Podcast Teacher Agent. Ask me anything about early childhood and parenting!\n(Type 'exit', 'quit' or 'bye' to end the session)")
            
            while True:
                user_input = input("\n Your request: ")
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("🎭 Good luck with your child")
                    break
                
                print("\n Examinating the creature...\n")
                agent(user_input)
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Make sure the dice service is running: python dice_roll_mcp_server.py")

if __name__ == "__main__":
    main()