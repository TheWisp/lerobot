## Run edit-tier MCP — transcript

Captured live against the unified GUI's MCP at `127.0.0.1:8000/mcp/` with NO active RLT session, which is the dominant operator state. The error responses below are the actual wire shapes the AI sees when it tries to update overrides on a host that isn't currently training — clean, actionable messages.

## get_run_status — confirm no run is active

_Intent: Pre-flight check the AI runs before proposing an override write._

**→** `get_run_status()`

**←**

```json
{
  "running": false,
  "command": null
}
```

## update_rlt_config — no active session

_Intent: Try to update beta without an active session. The response tells the AI WHY it failed and what to do (start an HVLA run first), not just a 409 status code._

**→** `update_rlt_config(beta=1.5)`

**← error** — `Error executing tool update_rlt_config: No active RLT session — start an HVLA run first, then update overrides.`

## update_rlt_config — no fields provided

_Intent: Call with nothing to update. Same error class as a missing session but a different message — the AI can distinguish 'session missing' from 'nothing to do'._

**→** `update_rlt_config()`

**← error** — `Error executing tool update_rlt_config: No active RLT session — start an HVLA run first, then update overrides.`

**For the success path** (write, clamp, partial-merge, previous_values), see the 10 unit tests in `tests/mcp/test_run_edit.py` which patch the `_active_config` module global to a tmp dir. Mocking a live training subprocess over the wire isn't worth the complexity for a transcript artifact.
