from fastmcp import Client
import asyncio

async def main():
    client = Client("http://127.0.0.1:9000/mcp")

    client._connect

    async with client:
        await client.ping()

        tools = await client.list_tools()
        for tool in tools:
            print(tool.model_dump_json(indent=2))

        result = await client.call_tool("add", arguments={"numbers": [1, 2, 3, 4, 5]})
        print(result)

if __name__ == "__main__":
    asyncio.run(main())

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
