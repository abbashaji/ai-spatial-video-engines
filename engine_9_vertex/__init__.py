"""
Vertex Engine — Engine IX
Perception-to-Graph Production Pipeline

"From Theoretical Orchestration to Functional Execution"

The system no longer DREAMS a video.
It MANAGES A DATABASE OF A REALITY IT IS SIMULATING.

Seven stages:
    Stage 1: VideoFrameExtractor  — cv2 keyframe extraction from input_video.mp4
    Stage 2: SpatialKernel        — Gemini ER 1.5 as hardware-grade sensor
    Stage 3: SpatialGraph         — NetworkX graph (Object)-[:RELATION {dist,angle}]->(Object)
    Stage 4: VoxelMap             — 2D [y,x] → 3D voxel grid via Pinhole Camera Model
    Stage 5: ProxyCubeBuilder     — SpatialGraph → BPY proxy cube scene
    Stage 6: PixelAuditor         — cv2 pixel comparison → Delta-Correction JSON (<5px)
    Stage 7: NeuralRefinement     — Depth map → ControlNet → photorealistic skin

Inherits: ArchonController (Engine VIII) — full 4-agent cycle available
"""
from .vertex_engine import (
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

__all__ = [
    "VertexController",
    "VertexDirective",
    "VertexConfig",
    "SpatialGraph",
    "SpatialNode",
    "SpatialEdge",
    "VoxelMap",
    "SpatialKernel",
    "SensorReading",
    "VideoFrameExtractor",
    "ExtractedFrame",
    "ProxyCubeBuilder",
    "PixelAuditor",
    "PixelAuditReport",
    "PixelDelta",
]
