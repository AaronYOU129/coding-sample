#!/usr/bin/env bash
# Purpose:    Submit the AlphaGenome MR prompt to one of the two agents and save
#             its answer + plots. We drive the agent headlessly with `claude -p`
#             and an isolated, strict --mcp-config so a run depends only on the
#             agent we point at — not on whatever MCP servers are registered in
#             the user's global config. "local" = the agent we generated with
#             Paper2Agent; "remote" = the official hosted MCP.
# Inputs:     $1 agent: "local" | "remote"   (default: local)
#             run_instructions_<agent>.txt    the prompt actually sent. It wraps
#                 the bare task (prompt.txt) with operational notes: how to read
#                 the API key, resolve the rsID, and WHERE to write the report
#                 and plots. The report path therefore lives in that file, not
#                 in this script.
#             .env: ALPHAGENOME_API_KEY=<key>  (gitignored; required)
#             For "local": Paper2Agent/AlphaGenome_Agent/ must already be built
#                 (see README "Reproduce Path A — generate the agent").
# Outputs:    output/<agent>_run.log          raw run transcript (final summary)
#             output/pathA_local_response.md   (local)  full report, written by
#             output/pathB_remote_response.md  (remote) the agent per its
#                                              instruction file
#             plots/data under output/plots_pathA (local) or output/plots (remote)
# Key Steps:  1) load API key  2) write an isolated MCP config for the chosen
#             agent  3) send run_instructions_<agent>.txt headlessly  4) clean up.
# How to Run: bash run_agent.sh local
#             bash run_agent.sh remote
set -euo pipefail

AGENT="${1:-local}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PROMPT_FILE="run_instructions_${AGENT}.txt"
[[ -f .env ]] || { echo "ERROR: .env with ALPHAGENOME_API_KEY not found" >&2; exit 1; }
set -a; source .env; set +a
[[ -n "${ALPHAGENOME_API_KEY:-}" ]] || { echo "ERROR: ALPHAGENOME_API_KEY empty" >&2; exit 1; }
[[ -f "$PROMPT_FILE" ]] || { echo "ERROR: '$PROMPT_FILE' not found" >&2; exit 1; }

CONFIG="$ROOT/.mcp_run.json"

write_local_config() {
  # Locally generated agent: a stdio MCP server run by its own uv-built env.
  # Python auto-prepends the server file's dir to sys.path, so `from tools.x` resolves.
  local agent_dir="$ROOT/Paper2Agent/AlphaGenome_Agent"
  local server="$agent_dir/src/alphagenome_mcp.py"
  local py="$agent_dir/repo/alphagenome-env/bin/python"
  [[ -f "$server" && -x "$py" ]] || { echo "ERROR: local agent not built (see README)" >&2; exit 1; }
  cat > "$CONFIG" <<JSON
{ "mcpServers": { "alphagenome_local": {
  "command": "$py", "args": ["$server"],
  "env": { "PYTHONPATH": "$agent_dir/src", "ALPHAGENOME_API_KEY": "$ALPHAGENOME_API_KEY" } } } }
JSON
}

write_remote_config() {
  # Official hosted MCP — an HTTP server, no local env needed.
  cat > "$CONFIG" <<'JSON'
{ "mcpServers": { "alphagenome": {
  "type": "http", "url": "https://Paper2Agent-alphagenome-mcp.hf.space/mcp" } } }
JSON
}

case "$AGENT" in
  local)  write_local_config ;;
  remote) write_remote_config ;;
  *) echo "ERROR: agent must be 'local' or 'remote'" >&2; exit 1 ;;
esac

echo "Running MR prompt against '$AGENT' agent (see run_instructions_${AGENT}.txt)..." >&2
claude -p "$(cat "$PROMPT_FILE")" \
  --mcp-config "$CONFIG" --strict-mcp-config \
  --output-format text \
  --dangerously-skip-permissions \
  < /dev/null > "output/${AGENT}_run.log" 2>&1

rm -f "$CONFIG"
echo "Done. Transcript: output/${AGENT}_run.log  (report path is set inside the instruction file)" >&2
