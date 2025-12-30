"""Search functionality for agentu."""
import asyncio
from typing import List, Dict, Any, Optional
from duckduckgo_search import DDGS

from .agent import Agent
from .tools import Tool

async def search_duckduckgo(
    query: str,
    max_results: int = 3,
    region: str = "wt-wt",
    safesearch: str = "moderate"
) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo and return results (async).

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 3)
        region: Region for search results (default: "wt-wt" for worldwide)
        safesearch: SafeSearch setting ("on", "moderate", or "off", default: "moderate")

    Returns:
        List of dictionaries containing search results with title, link, and snippet
    """
    try:
        # Run blocking search in executor
        loop = asyncio.get_event_loop()
        def _search():
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results=max_results,
                    region=region,
                    safesearch=safesearch
                ))
                return [
                    {
                        "title": r["title"],
                        "link": r["link"],
                        "snippet": r["body"]
                    }
                    for r in results
                ]
        return await loop.run_in_executor(None, _search)
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]

# Create a preconfigured search tool
search_tool = Tool(
    name="web_search",
    description="Search the web using DuckDuckGo",
    function=search_duckduckgo,
    parameters={
        "query": "str: The search query",
        "max_results": "int: Maximum number of results (default: 3)",
        "region": "str: Region for search results (default: 'wt-wt' for worldwide)",
        "safesearch": "str: SafeSearch setting ('on', 'moderate', or 'off', default: 'moderate')"
    }
)

class SearchAgent(Agent):
    """A specialized agent for web searches using DuckDuckGo."""

    def __init__(
        self,
        name: str = "search_assistant",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_results: int = 3,
        **kwargs
    ):
        super().__init__(name, model, temperature, **kwargs)
        self.max_results = max_results
        self._add_tool_internal(search_tool)
        self.set_context(
            "You are a search assistant that helps find relevant information on the web. "
            "When searching, you should:\n"
            "1. Formulate clear and specific search queries\n"
            "2. Consider multiple aspects of the topic\n"
            "3. Filter and summarize the most relevant information\n"
            "4. Provide proper attribution for sources"
        )
    
    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        region: str = "wt-wt",
        safesearch: str = "moderate"
    ) -> List[Dict[str, str]]:
        """
        Perform a web search directly without LLM (async).

        Args:
            query: Search query string
            max_results: Maximum number of results (uses instance default if None)
            region: Region for search results
            safesearch: SafeSearch setting

        Returns:
            List of search results
        """
        if max_results is None:
            max_results = self.max_results

        return await search_duckduckgo(query, max_results, region, safesearch)