"""DevOps monitoring agent API."""
from agentu import Agent, serve


def check_status(service: str) -> dict:
    """Check service status."""
    services = {
        "api": {"status": "running", "uptime": "99.9%"},
        "db": {"status": "running", "uptime": "99.8%"},
        "queue": {"status": "degraded", "uptime": "98.5%"}
    }
    return services.get(service, {"error": "Service not found"})


def restart(service: str) -> dict:
    """Restart a service."""
    return {"service": service, "restarted": True}


# Create agent
agent = Agent("devops", enable_memory=True) \
    .with_tools([check_status, restart])

# Serve API on port 8000
# Docs: http://localhost:8000/docs
# curl -X POST http://localhost:8000/execute \
#   -H "Content-Type: application/json" \
#   -d '{"tool_name": "check_status", "parameters": {"service": "api"}}'
serve(agent, port=8000, enable_cors=True)
