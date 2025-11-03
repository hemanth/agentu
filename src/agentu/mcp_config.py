"""MCP server configuration management."""
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .mcp_transport import MCPServerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPConfigLoader:
    """Loader for MCP server configurations."""

    @staticmethod
    def load_from_file(config_path: str) -> Dict[str, MCPServerConfig]:
        """Load MCP server configurations from a JSON file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary mapping server names to MCPServerConfig objects
        """
        config_file = Path(config_path)

        if not config_file.exists():
            logger.warning(f"MCP config file not found: {config_path}")
            return {}

        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            return MCPConfigLoader.load_from_dict(config_data)

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing MCP config file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading MCP config: {str(e)}")
            raise

    @staticmethod
    def load_from_dict(config_data: Dict[str, Any]) -> Dict[str, MCPServerConfig]:
        """Load MCP server configurations from a dictionary.

        Args:
            config_data: Dictionary containing MCP server configurations

        Returns:
            Dictionary mapping server names to MCPServerConfig objects
        """
        servers = {}
        mcp_servers = config_data.get('mcp_servers', {})

        for server_name, server_data in mcp_servers.items():
            try:
                config = MCPServerConfig.from_dict(server_name, server_data)
                servers[server_name] = config
                logger.info(f"Loaded MCP server config: {server_name}")
            except Exception as e:
                logger.error(f"Error loading config for server {server_name}: {str(e)}")

        return servers

    @staticmethod
    def get_default_config_path() -> str:
        """Get the default configuration file path.

        Looks for mcp_config.json in:
        1. Current working directory
        2. User's home directory (.agentu/mcp_config.json)
        3. Environment variable MCP_CONFIG_PATH
        """
        # Check environment variable
        env_path = os.environ.get('MCP_CONFIG_PATH')
        if env_path and Path(env_path).exists():
            return env_path

        # Check current directory
        cwd_config = Path.cwd() / 'mcp_config.json'
        if cwd_config.exists():
            return str(cwd_config)

        # Check home directory
        home_config = Path.home() / '.agentu' / 'mcp_config.json'
        if home_config.exists():
            return str(home_config)

        # Return default path (may not exist)
        return str(cwd_config)

    @staticmethod
    def load_default_config() -> Dict[str, MCPServerConfig]:
        """Load MCP server configurations from the default location."""
        config_path = MCPConfigLoader.get_default_config_path()
        return MCPConfigLoader.load_from_file(config_path)

    @staticmethod
    def create_example_config(output_path: str) -> None:
        """Create an example MCP configuration file.

        Args:
            output_path: Path where the example config should be created
        """
        example_config = {
            "mcp_servers": {
                "example_server": {
                    "type": "http",
                    "url": "https://api.example.com/mcp",
                    "auth": {
                        "type": "bearer",
                        "headers": {
                            "Authorization": "Bearer YOUR_TOKEN_HERE"
                        }
                    },
                    "timeout": 30
                },
                "api_key_server": {
                    "type": "http",
                    "url": "https://api.example.com/v2/mcp",
                    "auth": {
                        "type": "apikey",
                        "headers": {
                            "X-API-Key": "YOUR_API_KEY_HERE"
                        }
                    },
                    "timeout": 45
                },
                "custom_headers_server": {
                    "type": "http",
                    "url": "https://custom.example.com/mcp",
                    "auth": {
                        "type": "custom",
                        "headers": {
                            "Authorization": "Custom YOUR_TOKEN",
                            "X-Custom-Header": "custom_value"
                        }
                    }
                },
                "no_auth_server": {
                    "type": "http",
                    "url": "https://public.example.com/mcp"
                }
            }
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(example_config, f, indent=2)

        logger.info(f"Created example MCP config at: {output_path}")


def load_mcp_servers(config_path: Optional[str] = None) -> Dict[str, MCPServerConfig]:
    """Convenience function to load MCP server configurations.

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        Dictionary mapping server names to MCPServerConfig objects
    """
    if config_path:
        return MCPConfigLoader.load_from_file(config_path)
    else:
        return MCPConfigLoader.load_default_config()
