# Travel Planner
# In this script, we build a multi-agent travel planner using LangChain and MCP.

import asyncio
from typing import Dict, Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent
from tavily import TavilyClient

load_dotenv()

RETRYABLE_MCP_CODES = {-32603}


def _print_mcp_error(request_name: str, exc: Exception, attempt: int, max_retries: int):
    print(f"[MCP interceptor] {type(exc).__name__} on {request_name} "
          f"(attempt {attempt+1}/{max_retries}): {exc}")


class RetryMCPInterceptor:
    """Intercept MCP tool calls: retry transient failures, surface all errors gracefully.

    - Retryable McpError codes (e.g. -32603): retry with exponential backoff.
    - Non-retryable McpError codes (e.g. -32602): return error message immediately.
    - Any other exception (fetch failed, network errors, etc.): retry then return error message.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                _print_mcp_error(request.name, exc, attempt, self.max_retries)
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Tool call failed (non-retryable): {exc}")],
                        isError=False,
                    )
            except Exception as exc:
                last_error = exc
                _print_mcp_error(request.name, exc, attempt, self.max_retries)

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        print(f"[MCP interceptor] all {self.max_retries} retries exhausted for {request.name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool call failed after {self.max_retries} attempts: {last_error}")],
            isError=False,
        )


# Globals that will be initialized in main()
travel_agent = None
lodging_agent = None
itinerary_agent = None
coordinator = None


tavily_client = TavilyClient()


@tool
def web_search(query: str, search_number: int, max_search_number: int) -> Dict[str, Any]:
    """Search the web for information. You must track your search count by providing
    search_number (starting at 1) and max_search_number on every call.
    Queries must use only plain text characters. Do not use accented or special characters
      (e.g., use 'capacite' instead of 'capacité').
    """
    if search_number > max_search_number:
        return {"message": "Search limit reached. Please summarize your findings and provide your final answer."}
    try:
        return tavily_client.search(query)
    except Exception as e:
        return {"error": str(e)}


class TravelState(AgentState):
    origin: str
    destination: str
    start_date: str
    end_date: str
    budget: str
    travel_style: str


@tool
async def search_flights(runtime: ToolRuntime) -> str:
    """Travel agent searches for flights to the desired destination."""
    origin = runtime.state["origin"]
    destination = runtime.state["destination"]
    start_date = runtime.state["start_date"]
    end_date = runtime.state["end_date"]
    response = await travel_agent.ainvoke(
        {
            "messages": [HumanMessage(content=f"Find flights from {origin} to {destination} departing {start_date} and returning {end_date}")]
        }
    )
    return response["messages"][-1].content


@tool
def search_lodging(runtime: ToolRuntime) -> str:
    """Lodging agent chooses the best accommodation for the destination travel plan."""
    destination = runtime.state["destination"]
    budget = runtime.state["budget"]
    travel_style = runtime.state["travel_style"]
    query = f"Find lodging options in {destination} for a {travel_style} traveler with a budget of {budget}"
    response = lodging_agent.invoke({"messages": [HumanMessage(content=query)]})
    return response["messages"][-1].content


@tool
def suggest_itinerary(runtime: ToolRuntime) -> str:
    """Itinerary agent curates the perfect travel plan for the destination and preferences."""
    destination = runtime.state["destination"]
    start_date = runtime.state["start_date"]
    end_date = runtime.state["end_date"]
    travel_style = runtime.state["travel_style"]
    query = f"Create a travel itinerary for {destination} from {start_date} to {end_date} for a {travel_style} traveler"
    response = itinerary_agent.invoke({"messages": [HumanMessage(content=query)]})
    return response["messages"][-1].content


@tool
def update_state(
    origin: str,
    destination: str,
    start_date: str,
    end_date: str,
    budget: str,
    travel_style: str,
    runtime: ToolRuntime,
) -> str:
    """Update the state when you know all of the values."""
    return Command(
        update={
            "origin": origin,
            "destination": destination,
            "start_date": start_date,
            "end_date": end_date,
            "budget": budget,
            "travel_style": travel_style,
            "messages": [ToolMessage("Successfully updated state", tool_call_id=runtime.tool_call_id)],
        }
    )


def prompt_user_for_trip() -> Dict[str, str]:
    print("Welcome to the Travel Planner!")
    print("Please enter your trip details below. Press Enter to keep a value empty.")
    origin = input("Origin city or airport: ").strip()
    destination = input("Destination city: ").strip()
    start_date = input("Start date (YYYY-MM-DD): ").strip()
    end_date = input("End date (YYYY-MM-DD): ").strip()
    budget = input("Budget (e.g. $3000): ").strip()
    travel_style = input("Travel style (e.g. cultural, adventure, luxury, budget): ").strip()
    return {
        "origin": origin,
        "destination": destination,
        "start_date": start_date,
        "end_date": end_date,
        "budget": budget,
        "travel_style": travel_style,
    }


def build_trip_prompt(trip: Dict[str, str]) -> str:
    return (
        f"Plan a complete trip for a traveler with the following details:\n"
        f"Origin: {trip['origin']}\n"
        f"Destination: {trip['destination']}\n"
        f"Start date: {trip['start_date']}\n"
        f"End date: {trip['end_date']}\n"
        f"Budget: {trip['budget']}\n"
        f"Travel style: {trip['travel_style']}\n"
        "Please provide flight recommendations, lodging options, and a brief itinerary. "
        "Focus on a well-balanced plan and keep the answer concise."
    )


async def main() -> None:
    global travel_agent, lodging_agent, itinerary_agent, coordinator

    client = MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com",
            }
        },
        tool_interceptors=[RetryMCPInterceptor()],
    )

    tools = await client.get_tools()

    travel_agent = create_agent(
        model="gpt-5-nano",
        tools=tools,
        system_prompt="""
        You are a flight specialist. Search for the best flights for the travel plan.
        You are not allowed to ask any more follow up questions, you must find flight options based on the following criteria:
        - Price (lowest, economy class)
        - Duration (shortest)
        - Dates provided by the traveler
        - Best balance of cost and convenience for the destination
        You may need to make multiple searches to iteratively find the best flight options.
        You will be given the origin, destination, and travel dates. It is your job to think critically and provide a shortlist of flight recommendations.
        If the MCP tool fails, returns malformed output, or does not give you usable flight results, try the tool again.
        Once you have found the best options, present a concise shortlist with price, airline, and timing.
        """,
    )

    lodging_agent = create_agent(
        model="gpt-5-nano",
        tools=[web_search],
        system_prompt="""
        You are a lodging specialist. Search for hotels, apartments, or stays in the destination.
        You are not allowed to ask any more follow up questions, you must find the best lodging options based on the following criteria:
        - Price (best value for the traveler's budget)
        - Location (central or convenient to main attractions)
        - Reviews or ratings (highest quality)
        You may need to make multiple searches to iteratively find the best options.
        You have a suggested limit of 12 web searches. Count every web_search call you make.
        After 12 searches, stop searching and summarize the best options you have found so far.
        """,
    )

    itinerary_agent = create_agent(
        model="gpt-5-nano",
        tools=[web_search],
        system_prompt="""
        You are an itinerary planner. Use web search to create an excellent travel itinerary for the destination.
        You are not allowed to ask any more follow up questions, you must design the best trip plan based on:
        - Traveler preferences and travel style
        - Number of days and budget
        - Key attractions, dining, and local experiences
        You may need to make multiple searches to iteratively build the itinerary.
        You have a suggested limit of 12 web searches. Count every web_search call you make.
        After 12 searches, stop searching and summarize the best itinerary options you have found so far.
        """,
    )

    coordinator = create_agent(
        model="gpt-5-nano",
        tools=[search_flights, search_lodging, suggest_itinerary, update_state],
        state_schema=TravelState,
        system_prompt="""
        You are a travel coordinator.
        First find all the information you need to update the state. When you have the information, update the state.
        Once that has completed and returned, delegate the tasks to your specialists for flights, lodging, and itinerary planning.
        Once you have received their answers, coordinate the best travel plan for me.
        """,
    )

    trip = prompt_user_for_trip()
    prompt = build_trip_prompt(trip)

    print("\nGenerating your travel plan, please wait...")
    response = await coordinator.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"tags": ["TP"], "recursion_limit": 40},
    )

    print("\n=== Travel Planner Results ===")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
