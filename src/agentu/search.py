"""Search functionality for AgentU."""
from typing import List, Dict, Any, Optional
from duckduckgo_search import DDGS

from .agent import Agent
from .tools import Tool

def search_duckduckgo(
    query: str,
    max_results: int = 3,
    region: str = "wt-wt",
    safesearch: str = "moderate"
) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo and return results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 3)
        region: Region for search results (default: "wt-wt" for worldwide)
        safesearch: SafeSearch setting ("on", "moderate", or "off", default: "moderate")
    
    Returns:
        List of dictionaries containing search results with title, link, and snippet
    """
    try:
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
        model: str = "llama2",
        temperature: float = 0.7,
        max_results: int = 3
    ):
        super().__init__(name, model, temperature)
        self.max_results = max_results
        self.add_tool(search_tool)
        self.set_context(
            "You are a search assistant that helps find relevant information on the web. "
            "When searching, you should:\n"
            "1. Formulate clear and specific search queries\n"
            "2. Consider multiple aspects of the topic\n"
            "3. Filter and summarize the most relevant information\n"
            "4. Provide proper attribution for sources"
        )
    
    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        region: str = "wt-wt",
        safesearch: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Perform a web search with the given query.
        
        This is a convenience method that wraps the process_input method
        with search-specific parameters.
        """
        if max_results is None:
            max_results = self.max_results
            
        return self.process_input(
            f"Search for: {query}"
        )