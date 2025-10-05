import datetime
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

# ----------------------------
# Define Enhanced Utility Tools
# ----------------------------

@tool
def get_current_time() -> str:
    """
    Returns the current date and time formatted as a string.
    """
    # The current time is fetched based on the system's local time (IST).
    now = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p IST")
    return f"The current date and time is {now}."


@tool
def word_count(text: str) -> str:
    """
    Counts the number of words, characters, and sentences in a given block of text.
    """
    if not text or not text.strip():
        return "Error: The provided text is empty."

    words = len(text.split())
    characters = len(text)
    # A simple way to count sentences; may not be perfect for all cases.
    sentences = text.count('.') + text.count('!') + text.count('?')

    return f"The text contains {words} words, {characters} characters, and {sentences} sentences."


@tool
def search_web(query: str) -> str:
    """
    Performs a web search using the Tavily search engine to find up-to-date information.
    Use this for questions about current events, facts, or general knowledge.
    """
    # This is now a functional tool, not a placeholder.
    # Assumes you have TAVILY_API_KEY set in your environment.
    search = TavilySearch(max_results=3)
    results = search.invoke(query)
    return f"Here are the top web search results for '{query}': {results}"


@tool
def save_to_memory(key: str, value: str) -> str:
    """
    Saves a key-value pair to a temporary memory. (Simulation)
    """
    # In a real implementation, this could write to a database or a state management object.
    print(f"DEBUG: MEMORY_SAVE - Key: '{key}', Value: '{value}'")
    return f"Successfully saved the value for '{key}' to memory."


@tool
def weather_info(city: str) -> str:
    """
    Gets the current weather information for a specified city. (Placeholder)
    """
    # This remains a placeholder. You would integrate a real weather API here.
    return f"Weather in {city}: Sunny with scattered clouds, 31°C. (This is simulated data)"