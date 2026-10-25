# neuron explorer

Steering-first neuron explorer for Qwen3.5-4B. Every MLP neuron in a layer is pushed up and down on a few prompts, every
rollout is kept, and the explorer lays the results out over a 3D view of the model.

![neuron explorer](explorer.png)

## Explorer

    uv sync
    uv run python explorer/server.py        # http://localhost:8791

Reads sweep runs from `runs/` and per-layer stats from `stats/`, both produced by the sweep below; the caches in
`explorer/cache/` ship with the repo. Loads Qwen3.5-4B locally (MPS, CUDA or CPU) for live
steered chat, lens, connections and live inference over your own text.

- **map**: the model's layers in perspective, attention and MLP per layer, one point per neuron. Hover shows a neuron's
  steer lens and logit lens; click opens it; click a layer's label to view its sheet face-on. `writes` places
  neurons with similar down-projection columns next to each other.
- **list**: units ranked by word-cluster mass against random-direction controls; `lens` searches the steer and logit
  lenses for a token, `fires` runs your text through the model and filters neurons by activation.
- **neuron**: steered rollouts per prompt and dose, a three-column multi-turn steered chat (−, 0, +), lens and
  connections, and a name.
- **embeddings**: UMAP of a layer's rollouts, coloured by neuron, prompt, dose or kind.
- **CLI assistant** (fourth neuron-panel tab, or the right-edge handle): choose **Claude** or **Codex** in the
  **CLI assistant** selector. Install and sign in to the selected CLI (`claude` or `codex`) first. Each uses
  the explorer as an MCP server (`explorer/agent_mcp.py`). It reads neurons, cones and lenses, runs the
  live model (steering, activations, next-token logprobs, ablations), moves the map, list and highlights, and has file
  tools and a shell. Its prompt is `explorer/agent_prompt.md` plus `explorer/agent_methodology.md`, and
  `explorer/agent_prompt.local.md` when present (untracked, for your own additions);
  `EXPLORER_CLAUDE_MODEL` or `EXPLORER_CODEX_MODEL` picks the corresponding model; otherwise the CLI's default applies.
  The server answers only pages and tools on this machine.

### Codex CLI

Install [Codex CLI](https://developers.openai.com/codex/cli), run `codex login`, then start the explorer and
select **Codex**. No API key or Python SDK is required when using your saved CLI login. The adapter uses
[`codex exec --json` and session resume](https://developers.openai.com/codex/noninteractive), and configures the
explorer MCP server for each invocation without changing your Codex configuration.

On Windows, the adapter also looks for Codex in its standard per-user installation folders when it is absent from
the server's `PATH`. You can select a specific executable before starting the server:

```powershell
$env:EXPLORER_CODEX_BIN = "C:\path\to\codex.exe"
python -m uv run python explorer/server.py
```

Use the executable's full path, without extra command arguments. `EXPLORER_CLAUDE_BIN` provides the same override
for Claude. Restart the server after changing either setting.

Claude and Codex keep separate conversations in the browser. Codex thread IDs and retained messages are saved under
`explorer/cache/agent/codex/`, so replies can continue after a server restart. **Edit** and **reroll** start a fresh
Codex thread with the earlier messages and replies as context; discarded turns and their tool traces are excluded.
Images are passed through `--image`; other uploads, including PDFs, are provided as local paths for the CLI's tools
(reading PDFs requires a suitable local reader). Codex shell writes stay in `explorer/cache/agent/` using the
`workspace-write` sandbox; network access is enabled so it can reach the explorer's local HTTP tools. Interactive
approval requests are disabled, so operations outside that sandbox can fail and be reported in the chat.

Adapter regression checks (no model download or CLI login needed):

    uv run python -m unittest discover -s tests -v

## Sweep

Runs on a CUDA 13 GPU pod with the steering fork of vLLM ([RhizoNymph/vllm](https://github.com/RhizoNymph/vllm)).

    bash sweep/setup_pod.sh                                   # fork, deps, model; writes /workspace/env.sh
    python sweep/neuron_stats.py --layers 22 --tokens 2000000 --out stats/q35-4b
    python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --design main_rel --dose-unit rel \
        --neurons all --random-controls 256 --run q35-4b-L22
    sweep/run_depth.sh "4 9 13 17" 384 64                     # random-subset sweep across several layers

`main_rel` doses are fractions of the median residual norm leaving the layer: {who, snake, orange} × {−0.7, +0.5, +0.8}
and {free} × {+3}, plus one unsteered baseline per prompt.

| script | output |
|---|---|
| `sweep/neuron_stats.py` | per-neuron activation extremes by sign regime, down columns, residual norms |
| `sweep/steer_sweep.py` | `runs/<run>.jsonl`: text, token ids, logprobs, top-k per step; optional activation captures |
| `sweep/rank.py` | `runs/<run>.units.jsonl`: distinctive words, cluster mass, percentile against controls |
| `sweep/steer_lens.py` | `stats/steer_lens_L{l}.npz`: next-token Δlogprob and KL for ±0.8 pushes on every neuron |
| `sweep/depth_report.py` | `runs/depth_report.json`: per-layer share of neurons above controls, coherence, top units |
| `sweep/shift.py` | word-shift score against controls, all words counted |
| `sweep/merge_runs.py` | joins sharded runs of one layer |
| `sweep/check_equivalence.py` | HF vs vLLM steering and capture agreement |
| `sweep/bench.py`, `sweep/peek.py` | throughput conditions; terminal views of a run |
| `explorer/build_embeddings.py` | `runs/<run>.umap.json` |
| `explorer/build_layout.py` | `explorer/cache/layout_writes.npz` |
| `explorer/build_connections.py` | `explorer/cache/connections.npz`: each neuron's 24 strongest direct-path links up and down |
| `sweep/capture_prefill.py` | `captures/<run>.mlp_in.npy`: MLP input at the last prompt token at the run's capture layers, steered |

`research_log.md` holds the findings and reasoning; `NOTES.md` the source notes.
