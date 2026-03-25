"""Customer support agent."""
import asyncio
from agentu import Agent


def lookup_order(order_id: str) -> dict:
    """Look up order details."""
    orders = {
        "ORD-12345": {"status": "shipped", "tracking": "1Z999AA10123456784"},
        "ORD-12346": {"status": "processing", "tracking": None}
    }
    return orders.get(order_id, {"error": "Order not found"})


def cancel_order(order_id: str, reason: str) -> dict:
    """Cancel an order."""
    return {"success": True, "order_id": order_id, "reason": reason}


async def main():
    # Create agent with tools and memory
    agent = Agent("support", enable_memory=True) \
        .with_tools([lookup_order, cancel_order])

    # Execute tools
    result = await agent.call("lookup_order", {"order_id": "ORD-12345"})
    print(f"Order lookup: {result}")

    result = await agent.call("cancel_order", {
        "order_id": "ORD-12346",
        "reason": "Changed mind"
    })
    print(f"Cancel order: {result}")

    # Store context
    agent.remember("Customer prefers email updates", memory_type="fact")

    # Recall
    memories = agent.recall(query="customer", limit=5)
    print(f"\nMemories: {len(memories)} found")


if __name__ == "__main__":
    asyncio.run(main())
