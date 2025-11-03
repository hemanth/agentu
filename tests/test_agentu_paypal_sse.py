"""Test AgentU with PayPal MCP using SSE transport."""
import pytest
from agentu import (
    Agent,
    MCPServerConfig,
    AuthConfig,
    TransportType,
)


def test_agentu_paypal_sse_integration():
    """Test AgentU integration with PayPal MCP via SSE transport."""

    print("\n" + "="*70)
    print("AgentU + PayPal MCP (SSE) Integration Test")
    print("="*70)

    # Configure PayPal MCP server with SSE transport
    token = "A21AAOKKlLHheEfxUHHn60npgltwQbBoscxruv9owjS1yiWXz7-FWc6SgY_axJWcfAOUSp6AtVhgvf2tIn-PUvjB-LNkqkqNQ"

    auth = AuthConfig.bearer_token(token)

    config = MCPServerConfig(
        name="paypal",
        transport_type=TransportType.SSE,  # Use SSE transport
        url="https://mcp.paypal.com/sse",
        auth=auth,
        timeout=30
    )

    # Create agent
    agent = Agent(name="paypal_agent")

    try:
        # Step 1: Load tools from PayPal MCP
        print("\n[1] Loading PayPal MCP tools...")
        tools = agent.add_mcp_server(config)

        print(f"\n✓ Loaded {len(tools)} tools from PayPal MCP")

        # Verify we got tools
        assert len(tools) > 0, "Expected at least one tool from PayPal MCP"

        # Display tools
        print("\nAvailable PayPal tools:")
        for i, tool in enumerate(tools[:5], 1):  # Show first 5
            print(f"  {i}. {tool.name}")
            if tool.description:
                desc = tool.description.replace('[MCP:paypal] ', '')
                print(f"     {desc[:80]}...")

        if len(tools) > 5:
            print(f"  ... and {len(tools) - 5} more")

        # Step 2: Find and test list_transactions tool
        print(f"\n[2] Testing list_transactions tool...")

        list_transactions_tool = None
        for tool in tools:
            if 'list_transactions' in tool.name:
                list_transactions_tool = tool
                break

        if list_transactions_tool:
            print(f"✓ Found tool: {list_transactions_tool.name}")

            # Execute the tool
            print("\n[3] Executing list_transactions...")
            result = agent.execute_tool(list_transactions_tool.name, {})

            print(f"✓ Tool executed successfully!")

            # Parse result
            if result:
                import json
                if isinstance(result, str):
                    try:
                        data = json.loads(result)
                        transactions = data.get('transaction_details', [])
                        print(f"\n✓ Retrieved {len(transactions)} transactions")

                        # Show first transaction
                        if transactions:
                            first_txn = transactions[0]['transaction_info']
                            print(f"\nSample transaction:")
                            print(f"  ID: {first_txn.get('transaction_id')}")
                            print(f"  Amount: {first_txn.get('transaction_amount', {}).get('value')} "
                                  f"{first_txn.get('transaction_amount', {}).get('currency_code')}")
                            print(f"  Status: {first_txn.get('transaction_status')}")
                            print(f"  Type: {first_txn.get('instrument_sub_type')}")

                    except json.JSONDecodeError:
                        print(f"Result (first 200 chars): {str(result)[:200]}")
                else:
                    print(f"Result: {result}")

            assert result is not None, "Expected non-null result from list_transactions"

        else:
            print("⚠ list_transactions tool not found")
            print("Available tools:")
            for tool in tools:
                print(f"  - {tool.name}")

        print(f"\n{'='*70}")
        print("✓ Test completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        # Cleanup
        agent.close_mcp_connections()
        print("\n✓ Connections closed")


def test_agentu_paypal_tool_discovery():
    """Test tool discovery from PayPal MCP."""

    token = "A21AAOKKlLHheEfxUHHn60npgltwQbBoscxruv9owjS1yiWXz7-FWc6SgY_axJWcfAOUSp6AtVhgvf2tIn-PUvjB-LNkqkqNQ"

    auth = AuthConfig.bearer_token(token)
    config = MCPServerConfig(
        name="paypal",
        transport_type=TransportType.SSE,
        url="https://mcp.paypal.com/sse",
        auth=auth
    )

    agent = Agent(name="discovery_agent")

    try:
        tools = agent.add_mcp_server(config)

        # Verify we got expected number of tools
        assert len(tools) >= 30, f"Expected at least 30 tools, got {len(tools)}"

        # Check for key tools
        tool_names = [t.name for t in tools]

        # Should have transaction tool
        assert any('list_transactions' in name for name in tool_names), \
            "Expected list_transactions tool"

        # Should have invoice tools
        assert any('invoice' in name for name in tool_names), \
            "Expected invoice tools"

        # Should have order tools
        assert any('order' in name for name in tool_names), \
            "Expected order tools"

        print(f"\n✓ Tool discovery test passed ({len(tools)} tools found)")

    finally:
        agent.close_mcp_connections()


if __name__ == "__main__":
    # Run the main integration test
    test_agentu_paypal_sse_integration()

    print("\n\n")

    # Run discovery test
    test_agentu_paypal_tool_discovery()
