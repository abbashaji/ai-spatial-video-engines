# Archon Engine — Bedroom Causal Demo

Runs the built-in bedroom causal chain demo (`Ball → Glass → Cat`) to demonstrate multi-hop graph traversal and agentic four-node execution.

## Usage

```bash
python archon_engine.py \
  --api-key YOUR_GEMINI_KEY \
  --demo \
  --blender /path/to/blender
```

- Replace `YOUR_GEMINI_KEY` with your `GEMINI_API_KEY`.
- Replace `/path/to/blender` with the full path to your Blender 4.3+ executable (e.g. `/Applications/Blender.app/Contents/MacOS/Blender` on macOS).
- Omit `--blender` to run in graph-only mode (no rendering).

## What the demo does

1. **Node A — Perception Agent** scans the bedroom scene via Gemini ER 1.5 Pointing API.
2. **Node B — Grounding Agent** cross-references object positions against the Scene Hypergraph.
3. **Node C — Execution Agent** builds BPY proxy cubes with voxel-collision checks.
4. **Node D — Oracle Auditor** ghost-renders the scene and loops until drift < 2%.
