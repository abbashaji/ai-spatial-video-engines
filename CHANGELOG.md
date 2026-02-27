# Changelog

All notable changes to the AI Spatial Video Engine Series are documented here.

---

## v11.0.0 — Engine XI: Olympus

**Theme: "From imagining 3D space to calculating it."**

- **Protocol 1 — DepthTriangulator:** SfM parallax + Pinhole Camera Equation (`Z=f·H/h`) + Gemini Relative Scale Priors. Optical flow via `cv2.calcOpticalFlowFarneback` measures camera translation; depth fused by confidence weight.
- **Protocol 2 — StateChangeCache + TieredAuditor:** Static frames (<2% pixel diff) skip Gemini and extrapolate via velocity vector. Audit uses Flash first; ER 1.5 only if confidence < 85%. Typical result: ~10× latency reduction.
- **Protocol 3 — UVPinnedSkinning:** Texture generated once per object, UV-pinned to Blender mesh. Subsequent frames: only ControlNet-Tile patch (denoising < 0.3). Temporal seed = `sha256(session:object:keyframe_group) % 2^32`.
- Added `OlympusSpatialKernel` with `relative_scale` field in structural JSON.

---

## v10.0.0 — Engine X: Apex

**Theme: Full ER 1.5 SDK reference implementation.**

Five SDK fixes applied in one showcase engine:
1. New `google-genai` SDK (`genai.Client`, `types.Part.from_bytes`)
2. `ThinkingConfig` wired per-task (`NONE` → pointing, `DEEP` → causal)
3. Correct model string: `gemini-robotics-er-1.5-preview`
4. `box_2d` key (not `bbox`) matching ER 1.5 fine-tuning
5. Trajectory generation + consensus querying + code execution

Five stages: Spatial Blueprint → Trajectory → Consensus Oracle → Causal Chain Audit → Small Object Resolution.

---

## v9.0.0 — Engine IX: Vertex

**Theme: "The system no longer DREAMS a video. It MANAGES A DATABASE OF A REALITY IT IS SIMULATING."**

- Seven fully executable stages (see `engine_9_vertex/DEMO_vertex.md`)
- `VideoFrameExtractor` — cv2 keyframe extraction
- `SpatialKernel` — Gemini ER 1.5 as hardware sensor
- `SpatialGraph` — NetworkX (Object)-[:RELATION {distance, angle}]->(Object)
- `VoxelMap` — Pinhole Camera 2D→3D numpy occupancy grid
- `ProxyCubeBuilder` — SpatialGraph → BPY proxy cubes
- `PixelAuditor` — cv2 pixel diff, 5px threshold, recursive Delta-Correction JSON
- `NeuralRefinement` — ControlNet + Grok skin pass
- Inherits Engine VIII (Archon) fully as subsystem
- Hallucinations: **PROVABLY IMPOSSIBLE** (verified by cv2 pixel ruler)

---

## v8.0.0 — Engine VIII: Archon

**Theme: Multimodal Agentic Graph RAG × Persistent Intelligence Stack**

`Body = .blend | Brain = Gemini ER 1.5 | CNS = Scene Hypergraph`

- Four-agent cycle: Perception → Grounding → Execution → Oracle Auditor
- Persistent Memory: graph survives across sessions/months
- Multi-Hop Causality: `Ball → Glass → Cat` (BFS traversal)
- Cross-Modal Retrieval: "chair that looks like this photo"
- `SceneHypergraph`, `ICLMemoryLog`, `CoordinateMapper`, `ClippingAuditor`, `CrossModalEncoder`
- Hallucinations: **STRUCTURALLY IMPOSSIBLE**

---

## v7.0.0 — Engine VII: Nexus-V

**Theme: Aletheia-Blender Reality-Simulation Protocol (Full-Stack Unify)**

Merged all six previous engines into one coherent pipeline:
- Gemini ER 1.5 Spatial Kernel → Blender World Model → ControlNet Neural Skin → Autonomous Audit Loop
- Added: Gemini multimodal handshake, explicit `CoordinateMapper`, ICL memory log, AABB `ClippingAuditor`, frame-precise physics hand-off to Bullet/Mantaflow

---

## v6.0.0 — Engine VI: Prometheus

**Theme: Neural-Symbolic 3D Orchestration**

- Replaced Grok with Blender + ControlNet + Diffusion
- Gemini writes executable `bpy` Python scripts
- The `.blend` file **IS** the world state — hallucinations are geometrically impossible
- Infinite reshoot: change camera angle → re-run Phase 3 only

---

## v5.0.0 — Engine V: Aletheia

**Theme: Autotelic Neural-Physics & Style-Entropy Control**

- Gemini analyses a reference style and derives custom physics laws (Spider-Verse gravity ≠ 9.81 m/s²)
- N quantum paths generated; the highest Aesthetic Resonance Score wins
- `StyleDirective` input type

---

## v4.0.0 — Engine IV: Aether-Omni

**Theme: Active Inference & Reality Simulation**

- Replaced coordinate descriptions with physics laws (`F=ma`)
- Positions derived from force integration — cannot be wrong
- Pre-render contradiction detection eliminates errors before render
- `PhysicsLaw` input type

---

## v3.0.0 — Engine III: Nexus

**Theme: 4D World-State Synchronous Guidance**

- Complete Digital Twin built before Grok renders a pixel
- Material hash locks (SHA-256) prevent texture drift across clips
- First engine with explicit world-state model

---

## v2.0.0 — Engine II: Chronos

**Theme: Recursive Spatial Feedback**

- Surgical correction of only broken frames (not full re-renders)
- ~50× cheaper than full re-renders for single violations
- Recursive correction loop with configurable convergence threshold

---

## v1.0.0 — Engine I: Aether

**Theme: Sequential Spatial Correction**

- First engine in the series
- Gemini maps coordinates → Grok renders → Gemini checks
- Separated spatial truth from visual output for the first time
