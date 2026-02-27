"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AI SPATIAL VIDEO ENGINE SERIES                                      ║
║         Eleven Engines — Sequential Correction to Physics-Calculated Reality║
║                                                                              ║
║  "We stopped asking the AI to draw a video. We started asking it to build   ║
║   a world and record it. Then we connected that world to a real camera,     ║
║   a real graph database, and a real pixel ruler — and the system stopped    ║
║   dreaming. It started managing a database of a reality it was simulating.  ║
║   Then we taught it to calculate that reality from first principles."        ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT THIS SERIES IS
───────────────────
Eleven progressively sophisticated AI video orchestration engines, each one a
fundamental paradigm shift in how the hallucination problem is framed and solved.

The engines share a single mission: produce AI video where objects maintain
consistent positions, textures, and physics across every frame — without
drift, without warping, without the Goldfish Effect.

THE ELEVEN ENGINES AT A GLANCE
────────────────────────────────

  I    Aether      Sequential Spatial Correction
                   Gemini maps coordinates → Grok renders → Gemini checks.
                   The first engine to separate spatial truth from visual output.

  II   Chronos     Recursive Spatial Feedback
                   Surgically corrects only the broken frames (not the whole
                   video). 50× cheaper than full re-renders for single violations.

  III  Nexus       4D World-State Synchronous Guidance
                   Builds a complete Digital Twin before Grok renders a pixel.
                   Material hash locks (SHA-256) prevent texture drift across clips.

  IV   Aether-Omni Active Inference & Reality Simulation
                   Replaces coordinate descriptions with physics laws (F=ma).
                   Positions are derived from force integration — they cannot be wrong.
                   Pre-render contradiction detection eliminates errors before render.

  V    Aletheia    Autotelic Neural-Physics & Style-Entropy Control
                   Gemini analyses a reference style and derives custom physics laws
                   (Spider-Verse gravity ≠ 9.81 m/s²). N quantum paths generated;
                   the highest Aesthetic Resonance Score wins.

  VI   Prometheus  Neural-Symbolic 3D Orchestration
                   Replaces Grok with Blender + ControlNet + Diffusion.
                   Gemini writes executable bpy Python scripts. The .blend file IS
                   the world state. Hallucinations are geometrically impossible.
                   Infinite reshoot: change camera angle → re-run Phase 3 only.

  VII  Nexus-V     Aletheia-Blender Reality-Simulation Protocol (Full-Stack Unify)
                   Merges all six previous engines into one coherent pipeline:
                   Gemini ER 1.5 Spatial Kernel → Blender World Model →
                   ControlNet Neural Skin → Autonomous Audit Loop.
                   Adds: Gemini multimodal handshake, explicit CoordinateMapper,
                   ICL memory log on every Gemini call, AABB ClippingAuditor,
                   frame-precise physics hand-off to Bullet/Mantaflow.

  VIII Archon      Multimodal Agentic Graph RAG  ×  Persistent Intelligence Stack
                   Body = .blend  │  Brain = Gemini ER 1.5  │  CNS = Scene Hypergraph
                   Four-agent cycle:
                     Node A  Perception Agent    — Gemini ER 1.5 Pointing API
                     Node B  Grounding Agent     — Graph RAG cross-reference
                     Node C  Execution Agent     — BPY + Voxel-Collision-Check
                     Node D  Oracle Auditor      — Ghost render → drift < 2% → loop
                   Three game-changing capabilities:
                     • Persistent Memory      — graph survives across sessions/months
                     • Multi-Hop Causality    — Ball → Glass → Cat (BFS traversal)
                     • Cross-Modal Retrieval  — "chair that looks like this photo"
                   Hallucinations: STRUCTURALLY IMPOSSIBLE.

  IX   Vertex      Perception-to-Graph Production Pipeline
                   "The system no longer DREAMS a video.
                    It MANAGES A DATABASE OF A REALITY IT IS SIMULATING."
                   Seven stages (all executable today):
                     Stage 1  VideoFrameExtractor  — cv2 keyframe extraction
                     Stage 2  SpatialKernel        — Gemini ER 1.5 as hardware sensor
                     Stage 3  SpatialGraph         — NetworkX (Object)-[:RELATION
                                                      {distance, angle}]->(Object)
                     Stage 4  VoxelMap             — Pinhole Camera 2D→3D numpy grid
                     Stage 5  ProxyCubeBuilder     — SpatialGraph → BPY proxy cubes
                     Stage 6  PixelAuditor         — cv2 pixel diff, 5px threshold,
                                                      recursive Delta-Correction JSON
                     Stage 7  NeuralRefinement     — ControlNet + Grok skin pass
                   Archon (VIII) fully inherited as subsystem.
                   Hallucinations: PROVABLY IMPOSSIBLE (verified by cv2 pixel ruler).

  X    Apex        ER 1.5 Showcase Engine — Full SDK Migration
                   All five SDK fixes applied in one reference implementation:
                     Fix 1  New google-genai SDK (genai.Client, types.Part.from_bytes)
                     Fix 2  ThinkingConfig wired per-task (NONE→pointing, DEEP→causal)
                     Fix 3  Correct model string: gemini-robotics-er-1.5-preview
                     Fix 4  box_2d key (not bbox) matching ER 1.5 fine-tuning
                     Fix 5  Trajectory generation + consensus querying + code execution
                   Five stages: Spatial Blueprint → Trajectory → Consensus Oracle
                               → Causal Chain Audit → Small Object Resolution

  XI   Olympus     Physics-Grounded Spatial Video Engine
                   "From imagining 3D space to calculating it."
                   Three Gemini optimization protocols implemented:
                     Protocol 1  DepthTriangulator  — SfM parallax + Pinhole Camera
                                 Equation (Z=f·H/h) + Gemini Relative Scale Priors.
                                 Optical flow (cv2.calcOpticalFlowFarneback) measures
                                 camera translation; depth fused by confidence weight.
                     Protocol 2  StateChangeCache + TieredAuditor  — Static frames
                                 (<2% pixel diff) skip Gemini, extrapolate via velocity
                                 vector. Audit uses Flash first; ER 1.5 only if
                                 confidence < 85%. Typical: ~10× latency reduction.
                     Protocol 3  UVPinnedSkinning  — Texture generated ONCE per object,
                                 UV-pinned to Blender mesh. Subsequent frames: only
                                 ControlNet-Tile patch (denoising < 0.3). Temporal seed
                                 = sha256(session:object:keyframe_group) % 2^32.
                   Result: AI as Sensor + Refiner. Math as the unbreakable skeleton.


IMPORTS
───────
"""

# ── Engine I ──────────────────────────────────────────────────────────────────
try:
    from engine_1_aether.aether_engine import AetherEngine
except ImportError:
    AetherEngine = None  # type: ignore

# ── Engine II ─────────────────────────────────────────────────────────────────
try:
    from engine_2_chronos.chronos_engine import ChronosEngine
except ImportError:
    ChronosEngine = None  # type: ignore

# ── Engine III ────────────────────────────────────────────────────────────────
try:
    from engine_3_nexus.nexus_engine import NexusEngine
except ImportError:
    NexusEngine = None  # type: ignore

# ── Engine IV ─────────────────────────────────────────────────────────────────
try:
    from engine_4_aether_omni.aether_omni_engine import AetherOmniEngine, PhysicsLaw
except ImportError:
    AetherOmniEngine = None  # type: ignore
    PhysicsLaw = None  # type: ignore

# ── Engine V ──────────────────────────────────────────────────────────────────
try:
    from engine_5_aletheia.aletheia_engine import AletheiaEngine, StyleDirective
except ImportError:
    AletheiaEngine = None  # type: ignore
    StyleDirective = None  # type: ignore

# ── Engine VI ─────────────────────────────────────────────────────────────────
try:
    from engine_6_prometheus.prometheus_engine import PrometheusEngine, SceneDirective
except ImportError:
    PrometheusEngine = None  # type: ignore
    SceneDirective = None  # type: ignore

# ── Engine VII ────────────────────────────────────────────────────────────────
try:
    from engine_7_nexusv.nexusv_engine import (
        NexusVEngine,
        RealityDirective,
        AssetSpec,
        StoryBeat,
    )
except ImportError:
    NexusVEngine = None  # type: ignore
    RealityDirective = None  # type: ignore
    AssetSpec = None  # type: ignore
    StoryBeat = None  # type: ignore

# ── Engine VIII ───────────────────────────────────────────────────────────────
try:
    from engine_8_archon.archon_engine import (
        ArchonController,
        ArchonDirective,
        ArchonConfig,
        SceneHypergraph,
        GraphNode,
        GraphEdge,
        ICLMemoryLog,
        CoordinateMapper,
        ClippingAuditor,
        CrossModalEncoder,
        OracleVerdict,
    )
except ImportError:
    ArchonController = None  # type: ignore
    ArchonDirective = None  # type: ignore
    ArchonConfig = None  # type: ignore
    SceneHypergraph = None  # type: ignore
    GraphNode = None  # type: ignore
    GraphEdge = None  # type: ignore
    ICLMemoryLog = None  # type: ignore
    CoordinateMapper = None  # type: ignore
    ClippingAuditor = None  # type: ignore
    CrossModalEncoder = None  # type: ignore
    OracleVerdict = None  # type: ignore

# ── Engine IX ─────────────────────────────────────────────────────────────────
try:
    from engine_9_vertex.vertex_engine import (
        VertexController,
        VertexDirective,
        VertexConfig,
        SpatialGraph,
        SpatialNode,
        SpatialEdge,
        VoxelMap,
        SpatialKernel,
        SensorReading,
        VideoFrameExtractor,
        ExtractedFrame,
        ProxyCubeBuilder,
        PixelAuditor,
        PixelAuditReport,
        PixelDelta,
    )
except ImportError:
    VertexController = None  # type: ignore
    VertexDirective = None  # type: ignore
    VertexConfig = None  # type: ignore
    SpatialGraph = None  # type: ignore
    SpatialNode = None  # type: ignore
    SpatialEdge = None  # type: ignore
    VoxelMap = None  # type: ignore
    SpatialKernel = None  # type: ignore
    SensorReading = None  # type: ignore
    VideoFrameExtractor = None  # type: ignore
    ExtractedFrame = None  # type: ignore
    ProxyCubeBuilder = None  # type: ignore
    PixelAuditor = None  # type: ignore
    PixelAuditReport = None  # type: ignore
    PixelDelta = None  # type: ignore

# ── Engine X ──────────────────────────────────────────────────────────────────
try:
    from engine_10_apex.apex_engine import (
        ApexController,
        ApexDirective,
        ApexConfig,
        ApexResult,
        TrackedObject,
        CameraTrajectory,
        CausalChain,
        run_apex_demo,
    )
except ImportError:
    ApexController = None  # type: ignore
    ApexDirective = None  # type: ignore
    ApexConfig = None  # type: ignore
    ApexResult = None  # type: ignore
    TrackedObject = None  # type: ignore
    CameraTrajectory = None  # type: ignore
    CausalChain = None  # type: ignore
    run_apex_demo = None  # type: ignore

# ── Engine XI ─────────────────────────────────────────────────────────────────
try:
    from engine_11_olympus.olympus_engine import (
        OlympusController,
        OlympusConfig,
        DepthTriangulator,
        DepthEstimate,
        OpticalFlowResult,
        StateChangeCache,
        VelocityVector,
        TieredAuditor,
        UVPinnedSkinning,
        UVPinnedTexture,
        OlympusSpatialKernel,
        run_olympus_demo,
    )
except ImportError:
    OlympusController = None  # type: ignore
    OlympusConfig = None  # type: ignore
    DepthTriangulator = None  # type: ignore
    DepthEstimate = None  # type: ignore
    OpticalFlowResult = None  # type: ignore
    StateChangeCache = None  # type: ignore
    VelocityVector = None  # type: ignore
    TieredAuditor = None  # type: ignore
    UVPinnedSkinning = None  # type: ignore
    UVPinnedTexture = None  # type: ignore
    OlympusSpatialKernel = None  # type: ignore
    run_olympus_demo = None  # type: ignore


# ── Public API ────────────────────────────────────────────────────────────────

__version__ = "11.0.0"
__author__  = "AI Spatial Video Engine Series"

__all__ = [
    # I
    "AetherEngine",
    # II
    "ChronosEngine",
    # III
    "NexusEngine",
    # IV
    "AetherOmniEngine", "PhysicsLaw",
    # V
    "AletheiaEngine", "StyleDirective",
    # VI
    "PrometheusEngine", "SceneDirective",
    # VII
    "NexusVEngine", "RealityDirective", "AssetSpec", "StoryBeat",
    # VIII
    "ArchonController", "ArchonDirective", "ArchonConfig",
    "SceneHypergraph", "GraphNode", "GraphEdge",
    "ICLMemoryLog", "CoordinateMapper", "ClippingAuditor",
    "CrossModalEncoder", "OracleVerdict",
    # IX
    "VertexController", "VertexDirective", "VertexConfig",
    "SpatialGraph", "SpatialNode", "SpatialEdge",
    "VoxelMap", "SpatialKernel", "SensorReading",
    "VideoFrameExtractor", "ExtractedFrame",
    "ProxyCubeBuilder", "PixelAuditor", "PixelAuditReport", "PixelDelta",
    # X
    "ApexController", "ApexDirective", "ApexConfig", "ApexResult",
    "TrackedObject", "CameraTrajectory", "CausalChain", "run_apex_demo",
    # XI
    "OlympusController", "OlympusConfig",
    "DepthTriangulator", "DepthEstimate", "OpticalFlowResult",
    "StateChangeCache", "VelocityVector",
    "TieredAuditor",
    "UVPinnedSkinning", "UVPinnedTexture",
    "OlympusSpatialKernel", "run_olympus_demo",
]


# ── Engine Selection Helpers ──────────────────────────────────────────────────

SELECTION_GUIDE = """
ENGINE SELECTION GUIDE
══════════════════════

Use Case                                              Recommended Engine
──────────────────────────────────────────────────────────────────────────────
Real video + physics-accurate depth (SfM)           → XI   Olympus   ← NEW
Real video + low latency + stable textures          → XI   Olympus   ← NEW
Full ER 1.5 SDK showcase (trajectory, consensus)    → X    Apex
Input is a real video — measure from reality        → IX   Vertex
World must persist across sessions / months         → VIII Archon
Full stack: Gemini spatial + Blender + skin         → VII  Nexus-V
Studio-grade .blend world, infinite reshoot         → VI   Prometheus
Stylized animation (Spider-Verse, Ghibli, art)      → V    Aletheia
Physics events (collisions, shattering, fluids)     → IV   Aether-Omni
Long clip, character consistency across 30s+        → III  Nexus
Fast surgical frame correction, moderate scenes     → II   Chronos
Simple scene, text prompt, fast first iteration     → I    Aether

DECISION TREE:
  Need physics-accurate depth?     YES → Olympus (XI)   ← start here
  Texture flickering a problem?    YES → Olympus (XI)
  Latency is critical?             YES → Olympus (XI)
  Has a reference video?           YES → Vertex (IX) or Olympus (XI)
  Full ER 1.5 SDK demo needed?     YES → Apex (X)
  Needs cross-session memory?      YES → Archon (VIII)
  Needs full deterministic 3D?     YES → Nexus-V (VII)
  Stylized / non-realistic?        YES → Aletheia (V)
  Physics events required?         YES → Aether-Omni (IV)
  Long clip / identity drift?      YES → Nexus (III)
  Fast prototype?                  YES → Chronos (II) or Aether (I)
"""


def select_engine(
    has_reference_video: bool = False,
    needs_physics_depth: bool = False,
    needs_stable_textures: bool = False,
    low_latency: bool = False,
    needs_er15_showcase: bool = False,
    needs_persistence: bool = False,
    needs_full_stack: bool = False,
    stylized: bool = False,
    physics_events: bool = False,
    long_clip: bool = False,
) -> str:
    """
    Return the recommended engine name for the given production requirements.

    Example:
        select_engine(needs_physics_depth=True)   → 'Olympus (Engine XI)'
        select_engine(has_reference_video=True)   → 'Olympus (Engine XI)'
        select_engine(needs_er15_showcase=True)   → 'Apex (Engine X)'
        select_engine(needs_persistence=True)     → 'Archon (Engine VIII)'
        select_engine(physics_events=True)        → 'Aether-Omni (Engine IV)'
    """
    if needs_physics_depth or needs_stable_textures or low_latency:
        return "Olympus (Engine XI)"
    if has_reference_video:
        return "Olympus (Engine XI)"
    if needs_er15_showcase:
        return "Apex (Engine X)"
    if needs_persistence:
        return "Archon (Engine VIII)"
    if needs_full_stack:
        return "Nexus-V (Engine VII)"
    if stylized:
        return "Aletheia (Engine V)"
    if physics_events:
        return "Aether-Omni (Engine IV)"
    if long_clip:
        return "Nexus (Engine III)"
    return "Aether (Engine I) or Chronos (Engine II)"


def get_version_info() -> dict:
    """Return version and availability for all eleven engines."""
    return {
        "series_version": __version__,
        "engines": {
            "I    Aether":      {"class": "AetherEngine",      "available": AetherEngine is not None},
            "II   Chronos":     {"class": "ChronosEngine",     "available": ChronosEngine is not None},
            "III  Nexus":       {"class": "NexusEngine",       "available": NexusEngine is not None},
            "IV   Aether-Omni": {"class": "AetherOmniEngine",  "available": AetherOmniEngine is not None},
            "V    Aletheia":    {"class": "AletheiaEngine",    "available": AletheiaEngine is not None},
            "VI   Prometheus":  {"class": "PrometheusEngine",  "available": PrometheusEngine is not None},
            "VII  Nexus-V":     {"class": "NexusVEngine",      "available": NexusVEngine is not None},
            "VIII Archon":      {"class": "ArchonController",  "available": ArchonController is not None},
            "IX   Vertex":      {"class": "VertexController",  "available": VertexController is not None},
            "X    Apex":        {"class": "ApexController",    "available": ApexController is not None},
            "XI   Olympus":     {"class": "OlympusController", "available": OlympusController is not None},
        },
    }
