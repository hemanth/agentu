import requests
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Tool:
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]

class Agent:
    def __init__(self, name: str, model: str = "llama2", temperature: float = 0.7):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.tools: List[Tool] = []
        self.context = ""
        self.conversation_history = []
        
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent's toolkit."""
        self.tools.append(tool)
        logger.info(f"Added tool: {tool.name} to agent {self.name}")
        
    def set_context(self, context: str) -> None:
        """Set the context for the agent."""
        self.context = context
        
    def _format_tools_for_prompt(self) -> str:
        """Format tools into a string for the prompt."""
        tools_str = "Available tools:\n\n"
        for tool in self.tools:
            tools_str += f"Tool: {tool.name}\n"
            tools_str += f"Description: {tool.description}\n"
            tools_str += f"Parameters: {json.dumps(tool.parameters, indent=2)}\n\n"
        return tools_str

    def _call_ollama(self, prompt: str) -> str:
        """Make an API call to Ollama and handle streaming response."""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False  # Disable streaming for simpler handling
                },
                timeout=30
            )
            response.raise_for_status()
            
            # Handle the response
            full_response = ""
            response_json = response.json()
            
            if "error" in response_json:
                logger.error(f"Ollama API error: {response_json['error']}")
                raise Exception(response_json['error'])
                
            full_response = response_json.get("response", "")
            
            if not full_response:
                logger.error("Empty response from Ollama")
                raise Exception("Empty response from Ollama")
                
            return full_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            raise

    def evaluate_tool_use(self, user_input: str) -> Dict[str, Any]:
        """Evaluate which tool to use based on user input."""
        prompt = f"""Context: {self.context}

{self._format_tools_for_prompt()}

User Input: {user_input}

You are an AI assistant that helps determine which tool to use and how to use it.
Analyze the user input and available tools to determine the appropriate action.

Your response must be valid JSON in this exact format:
{{
    "selected_tool": "name_of_tool",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "reasoning": "Your explanation here"
}}

For the calculator tool, ensure numeric parameters are numbers, not strings.
Remember to match the parameter names exactly as specified in the tool description.

Example response for calculator:
{{
    "selected_tool": "calculator",
    "parameters": {{
        "x": 5,
        "y": 3,
        "operation": "multiply"
    }},
    "reasoning": "User wants to multiply 5 and 3"
}}"""

        try:
            response = self._call_ollama(prompt)
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Ollama response: {str(e)}")
            return {
                "selected_tool": None,
                "parameters": {},
                "reasoning": "Error parsing response"
            }

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a specific tool with given parameters."""
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    return tool.function(**parameters)
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {str(e)}")
                    raise
        raise ValueError(f"Tool {tool_name} not found")

    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and execute appropriate tool."""
        evaluation = self.evaluate_tool_use(user_input)
        
        if not evaluation["selected_tool"]:
            return {"error": "No appropriate tool found"}
            
        result = self.execute_tool(
            evaluation["selected_tool"],
            evaluation["parameters"]
        )
        
        response = {
            "tool_used": evaluation["selected_tool"],
            "parameters": evaluation["parameters"],
            "reasoning": evaluation["reasoning"],
            "result": result
        }
        
        self.conversation_history.append({
            "user_input": user_input,
            "response": response
        })
        
        return response