# MCP support

The playground exposes a fixed catalog of public, unauthenticated MCP
servers. Clients send server IDs and namespaced tool names; they never send a
remote URL. The backend validates the IDs against the product allowlist and
uses runtime discovery to validate selected tools before saving a playground.

The approved server IDs are:

- `microsoft_learn`
- `deepwiki`
- `aws_knowledge`

The catalog is proxied through authenticated backend endpoints:

- `GET /mcp/servers`
- `GET /mcp/servers/{server_id}/tools`

The second endpoint performs a discovery check for that request. An
`unavailable` status does not represent a persistent connection; tool calls
open their own runtime connection.

`POST /playground` and `PATCH /playground/{id}` accept `mcp_servers`. The
field is always a list in playground responses and defaults to `[]`. The
existing `tools` list carries both builtin names and namespaced MCP names
(names beginning with `mcp_`).

When a model thread is created, the backend stores the selected MCP server IDs
and main-agent MCP tool names on the thread. It sends the server IDs and tool
list to the runtime when creating the runtime session. Those thread values are
immutable: changing playground defaults affects only future model threads.
Runtime session forks retain the same snapshot.

Per-turn tool semantics remain three-state:

- `tools: null` delegates the saved session selection to the runtime, including
  its frozen MCP snapshot.
- `tools: []` disables every tool for that turn, including MCP tools.
- A non-empty list selects builtin tools and automatically retains the
  thread's MCP snapshot. A namespaced MCP tool outside that snapshot is
  rejected.

For compare-mode fanout, the current playground defaults are used when a new
model thread is created. Existing threads keep their own MCP subset; newly
selected MCP names are filtered from that thread's turn while its frozen
subset and the requested builtin tools remain active. Single-thread
continuation and regeneration reject MCP additions outside the thread
snapshot.

Migration `010_playground_mcp` adds the non-null JSON columns and backfills
legacy playgrounds and threads with empty lists. Apply it through the normal
Alembic workflow; this change does not apply migrations to a live database.
