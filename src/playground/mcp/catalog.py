"""The MCP servers exposed by the playground product.

The runtime owns the endpoint URLs and remote tool metadata.  The backend only
accepts these stable IDs so callers cannot provide arbitrary remote URLs.
"""

from collections.abc import Iterable

MCP_TOOL_PREFIX = "mcp_"
APPROVED_MCP_SERVER_IDS = frozenset(
    {
        "microsoft_learn",
        "deepwiki",
        "aws_knowledge",
    }
)


def mcp_tool_names(names: Iterable[str] | None) -> list[str]:
    """Return namespaced MCP tools while preserving caller order."""

    return [name for name in names or [] if name.startswith(MCP_TOOL_PREFIX)]
