# Task 4 — AI Agents (AlphaGenome via Paper2Agent)

Build an AI agent from the **AlphaGenome** paper using **Paper2Agent**
(`https://github.com/jmiao24/Paper2Agent`), submit the Mendelian-randomization
prompt about `rs11174281`, and save the output.

The assignment prompt is in [`prompt.txt`](prompt.txt). It asks the agent to look
up the variant, run variant-effect prediction across all modalities, rank brain
tissues by RNA-seq effect, find the top gene(s) and direction, check regulatory
overlap, save plots, and summarize the likely gene + mechanism.

## Two agents were used (both run the same prompt)

| | Path A — **local** | Path B — **remote** |
|---|---|---|
| What | The agent we **generated ourselves** by running `Paper2Agent.sh` on `google-deepmind/alphagenome` | The **official hosted** AlphaGenome MCP (`Paper2Agent-alphagenome-mcp.hf.space`) |
| Why | Faithful to "create an agent from the paper" | Fast cross-check / sanity baseline |
| Tools | 25 (7 tutorial modules) | 22 |
| Report | [`output/pathA_local_response.md`](output/pathA_local_response.md) | [`output/pathB_remote_response.md`](output/pathB_remote_response.md) |
| Plots | `output/plots_pathA/` (3 PNG + CSVs) | `output/plots/` (3 PNG) + CSVs in `output/` |

Both agents independently converge on the same answer (below), which is the main
evidence that the result is real and not an artifact of one toolchain.

## Result (what AlphaGenome concluded)

- **Variant:** `rs11174281` → **chr12:39,917,691 C>T** (GRCh38, confirmed via Ensembl
  REST + dbSNP), an **intron** variant in **`SLC2A13`**; common (MAF ≈ 0.145).
- **Class:** a **non-coding regulatory variant** — largest effects on chromatin
  accessibility (DNase/ATAC), TF binding (incl. **CTCF**), and histone marks
  (H3K4me1/H3K27ac), all top ~1–3% genome-wide. Direct RNA effect is tiny in
  magnitude but high-percentile. Splicing/coding effects negligible.
- **Brain:** the affected regulatory element is active in brain (cerebellum,
  prefrontal cortex, striatum, hippocampus, amygdala) — the LCU-relevant circuits.
- **Top target gene:** **`KIF21A`** (~474 kb upstream, in-window) — the T allele
  **raises KIF21A** consistently across brain RNA-seq tracks. The host gene
  `SLC2A13` responds weakly/mixed. Path A additionally flagged **`LINC02555` down**
  across all brain regions.
- **Best-guess mechanism:** a **distal brain-enhancer / CTCF variant** that modestly
  up-regulates KIF21A (T allele) in brain, especially cerebellum/cortex.

### Caveats carried in both reports (important for the MR)
1. **Risk-allele orientation** — directions are reported for the **T (minor) allele**;
   flip if your within-family GWAS coded **C** as the LCU-risk allele. The rare
   third allele **A** was not modeled.
2. **Predictions, not measurements** — triangulate the KIF21A link against brain
   eQTL/Hi-C (GTEx, PsychENCODE).
3. **Gene-vs-host mismatch / pleiotropy** — the strongest expression target (KIF21A)
   is not the host gene (SLC2A13), and multiple genes respond → a possible
   horizontal-pleiotropy flag; colocalize before assigning a single gene→LCU path.
4. Path A noted a **data artifact** in the combined all-scorers CSV (duplicated
   RNA_SEQ rows inflating some values, e.g. a spurious LRRK2 log2FC); it used the
   dedicated **GeneMaskLFC** scorer as authoritative instead.

## Reproduce

Prerequisites: Python ≥ 3.10, `git`, `uv`, the `claude` CLI (logged in), and an
**AlphaGenome API key** (free, non-commercial: <https://deepmind.google.com/science/alphagenome>).
Put it in a gitignored `.env`:

```bash
echo 'ALPHAGENOME_API_KEY=YOUR_KEY' > .env
pip install fastmcp alphagenome
```

### Run the prompt against an already-built agent
```bash
bash run_agent.sh remote   # uses the hosted MCP — no generation needed
bash run_agent.sh local    # uses the agent generated under Paper2Agent/ (below)
```

### Reproduce Path A — generate the agent
Two environment fixes were required on macOS (both already applied to the cloned
`Paper2Agent/`, documented here so the run is reproducible elsewhere):

1. **bash ≥ 4** — Paper2Agent uses `declare -A` / `mapfile`; macOS ships bash 3.2.
   Installed bash 5 and put it first on `PATH` so nested `bash scripts/*.sh` calls
   use it too: `conda install -n base -c conda-forge 'bash>=5'` then `conda activate base`.
2. **Model id** — the pipeline hard-codes `claude-sonnet-4-20250514`, which this
   account cannot access; patched all `scripts/*.sh` + `tools/*.py` to
   `claude-sonnet-4-6`.

```bash
cd Paper2Agent
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate base   # bash 5 on PATH
bash Paper2Agent.sh \
  --project_dir AlphaGenome_Agent \
  --github_url https://github.com/google-deepmind/alphagenome \
  --api "$ALPHAGENOME_API_KEY"
```

Notes:
- ~50 min wall-clock, ~$15 of Claude (Sonnet) credits here.
- `context7` (a docs MCP) was skipped because Node/`npx` isn't installed — non-fatal;
  the pipeline continued and still produced 25 tools.
- The pipeline's final auto-launch step fails on a path-naming bug (it looks for
  `AlphaGenome_Agent_mcp.py`; the file is `src/alphagenome_mcp.py`). `run_agent.sh`
  registers the correct file/env, so this is cosmetic.

## Files

```
task4_ai_agents/
├── prompt.txt                      # the assignment's MR prompt (bare task)
├── run_instructions_local.txt      # prompt + operational notes, sent to the local agent
├── run_instructions_remote.txt     # prompt + operational notes, sent to the remote agent
├── run_agent.sh                    # headless runner (local|remote), isolated --mcp-config
├── CLAUDE.md                       # analyst guidance loaded by the agent at runtime
├── .env                            # ALPHAGENOME_API_KEY (gitignored)
├── output/
│   ├── pathA_local_response.md     # Path A report (self-generated agent)
│   ├── pathB_remote_response.md    # Path B report (hosted agent)
│   ├── pathA_generation.log        # full Paper2Agent.sh generation log
│   ├── pathA_run.log / pathB_run.log
│   ├── plots_pathA/                # Path A plots (PNG) + analysis CSVs + script
│   ├── plots/                      # Path B plots (PNG)
│   └── *.csv, *.tsv                # Path B supporting data
└── Paper2Agent/                    # cloned framework + generated AlphaGenome_Agent/ (gitignored)
```

## MCP servers registered during this task (cleanup)
Building/validating registered two MCP servers in `~/.claude.json`. `run_agent.sh`
does **not** depend on them (it uses an isolated config), so they can be removed:
```bash
claude mcp remove --scope user alphagenome   # hosted MCP added for Path B validation
claude mcp remove context7                    # added by Paper2Agent (project scope)
```
