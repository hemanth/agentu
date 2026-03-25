"""Serve agent as REST API."""
from agentu import Agent, serve


def get_status(order_id: str) -> dict:
    """Get order status."""
    return {"order_id": order_id, "status": "shipped"}


# Create agent
agent = Agent("support").with_tools([get_status])

# Serve on port 8000
# curl -X POST http://localhost:8000/execute \
#   -H "Content-Type: application/json" \
#   -d '{"tool_name": "get_status", "parameters": {"order_id": "12345"}}'
serve(agent, port=8000)
