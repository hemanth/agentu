from typing import Dict, Any
import json

def validate_json_response(response: str) -> Dict[str, Any]:
    """Validate and parse JSON response from LLM."""
    try:
        data = json.loads(response)
        required_keys = {"selected_tool", "parameters", "reasoning"}
        if not all(key in data for key in required_keys):
            raise ValueError("Missing required keys in response")
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {str(e)}")
