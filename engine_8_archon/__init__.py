"""
Archon Engine — Engine VIII
Multimodal Agentic Graph RAG × Deterministic 3D World

The Body (.blend) + The Brain (Gemini ER 1.5) + The CNS (Scene Hypergraph)

Four agents:
    Node A: PerceptionAgent  — Gemini ER 1.5 Pointing API
    Node B: GroundingAgent   — Graph RAG cross-reference
    Node C: ExecutionAgent   — BPY script generation
    Node D: OracleAuditor    — Closed-loop self-correction

Five phases (instruction 1.txt Oracle Protocol):
    Phase 0: Multimodal Extraction
    Phase 1: Graph-Grounded Build
    Phase 2: Oracle Verification
    Phase 3: Kinetic & Sim Baking
    Phase 4: Neural Refinement
"""
from .archon_engine import (
    ArchonController,
    ArchonDirective,
    ArchonConfig,
    AssetSpec,
    StoryBeat,
    StyleDirective,
    PhysicsLaw,
    SceneHypergraph,
    GraphNode,
    GraphEdge,
    PerceptionAgent,
    GroundingAgent,
    ExecutionAgent,
    OracleAuditor,
    CrossModalEncoder,
    CoordinateMapper,
)

__all__ = [
    "ArchonController",
    "ArchonDirective",
    "ArchonConfig",
    "AssetSpec",
    "StoryBeat",
    "StyleDirective",
    "PhysicsLaw",
    "SceneHypergraph",
    "GraphNode",
    "GraphEdge",
    "PerceptionAgent",
    "GroundingAgent",
    "ExecutionAgent",
    "OracleAuditor",
    "CrossModalEncoder",
    "CoordinateMapper",
]
