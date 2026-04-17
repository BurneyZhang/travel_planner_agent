# Travel Planner

A simple interactive travel planner script built with LangChain and MCP.

## Files

- `travel_planners.py` — main Python script for the travel planner.
- `.env` — local environment file with your credentials (ignored by Git).
- `requirements.txt` — Python dependencies for the project.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate
   ```
   On Windows PowerShell:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create `.env` and fill in your values:
   ```bash
   create .env
   ```

4. Run the travel planner:
   ```bash
   python travel_planners.py
   ```

## Code explanation

The main script `travel_planners.py` is built as an interactive travel planner with the following logic:

- Loads environment variables using `python-dotenv`.
- Creates a retry interceptor for MCP tool calls that retries transient failures and returns graceful error content for non-retryable errors.
- Defines a `web_search` tool backed by `TavilyClient`.
- Uses LangChain agent classes to define a `TravelState` schema for trip inputs.
- Defines tools for:
  - `search_flights` — asks a flight specialist agent to find flight recommendations.
  - `search_lodging` — asks a lodging specialist agent to find accommodation options.
  - `suggest_itinerary` — asks an itinerary specialist agent to build a trip plan.
  - `update_state` — updates the shared travel state for the coordinator agent.
- Builds an interactive prompt from user input with origin, destination, dates, budget, and travel style.
- Runs the coordinator agent with the generated prompt and prints the final travel plan.

## Environment variables

The project uses `python-dotenv` to load environment variables from `.env`.

Example values:

- `OPENAI_API_KEY` — API key for OpenAI if required by your LangChain setup.
- `MCP_URL` — MCP endpoint URL if your client requires a URL from env.
- `MCP_API_KEY` — API key or token for your MCP environment.
- `LANGSMITH_API_KEY` — API key or token for your LangSmith setup.

## Notes

- `.env` is ignored in version control by `.gitignore`.
- Keep your secret keys private and do not push them to GitHub.
- This project assumes the MCP service and your tool environment are configured correctly.
