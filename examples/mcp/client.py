from fastmcp import Client
import asyncio

async def main():
    client = Client("http://127.0.0.1:9000/mcp")

    async with client:
        await client.ping()

        tools = await client.list_tools()
        print(f"Available tools: {tools}")

if __name__ == "__main__":
    asyncio.run(main())
