# P2 · homelab-mcp + agent evals

**Question:** how well does an operations agent do when replayed against its own history of real incidents?

**Status:** planned.

**Plan:**

- `homelab-mcp`: a read-only MCP server (FastMCP) for a homelab's knowledge base, tickets, containers and metrics, published as an installable package with tests and a documented security model.
- Replay eval over about 40 real incidents, labelled by hand.
- Tool-use eval over 25 tasks: success, right tool, steps, cost.

**Case study:** *Evaluating an ops agent on its own incident history.*
