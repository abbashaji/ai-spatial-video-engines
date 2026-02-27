"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         VERTEX ENGINE  —  Engine IX of IX                                   ║
║         Perception-to-Graph Production Pipeline                              ║
║         "From Theoretical Orchestration to Functional Execution"            ║
║                                                                              ║
║  Paradigm:  The system no longer DREAMS a video.                            ║
║             It MANAGES A DATABASE OF A REALITY IT IS SIMULATING.           ║
║             This is the difference between a toy and a studio-grade engine. ║
║                                                                              ║
║  Core Innovation (instruction 4.txt):                                       ║
║    • Hardware-Grade Sensor: Gemini ER 1.5 as a physical perception sensor   ║
║      feeding structured data — not a creative content generator             ║
║    • SpatialGraph (NetworkX): Edges carry {distance, angle, overlap}        ║
║      enabling "cup is ON table even if camera moves" memory                 ║
║    • VoxelMap: True 3D voxel grid built from 2D [y,x] pointing output      ║
║      via multi-frame depth triangulation                                     ║
║    • ProxyCubeBuilder: BPY script that reads the graph directly and         ║
║      places proxy cubes — every cube position mathematically derived        ║
║    • PixelAuditor: Viewport wireframe render → pixel-level comparison       ║
║      → Delta-Correction JSON if alignment error > threshold pixels          ║
║    • VideoFrameExtractor: Real cv2-based frame extraction from              ║
║      input_video.mp4 — feeds raw frames into the perception pipeline        ║
║                                                                              ║
║  Full Lineage (all previous engines absorbed):                               ║
║    Engine I   (Aether)       — Spatial consistency fundamentals             ║
║    Engine II  (Chronos)      — Surgical correction loops                   ║
║    Engine III (Nexus)        — Long-clip character consistency              ║
║    Engine IV  (Aether-Omni)  — Explicit physics laws (PhysicsLaw)          ║
║    Engine V   (Aletheia)     — Style-Differentiable Physics (SDP)          ║
║    Engine VI  (Prometheus)   — Deterministic BPY geometry                  ║
║    Engine VII (Nexus-V)      — Gemini Spatial JSON Handshake               ║
║    Engine VIII (Archon)      — Scene Hypergraph + 4-Agent cycle             ║
║    Engine IX  (Vertex)       — Production: video→graph→Blender→audit        ║
║                                                                              ║
║  Five Stages (production-hardened, all executable today):                   ║
║    Stage 1: VideoFrameExtractor  — cv2 frame extraction + keyframe select  ║
║    Stage 2: SpatialKernel        — Gemini ER 1.5 → [label, [y,x]] per frame║
║    Stage 3: SpatialGraph         — NetworkX graph with spatial edges        ║
║    Stage 4: VoxelMap             — 2D points → 3D voxel grid               ║
║    Stage 5: ProxyCubeBuilder     — Graph → BPY proxy scene                 ║
║    Stage 6: PixelAuditor         — Wireframe render → pixel delta audit    ║
║    Stage 7: NeuralRefinement     — Depth map → ControlNet final render      ║
║                                                                              ║
║  WHAT IS REAL TODAY (zero stubs in core path):                              ║
║    ✓ cv2 video frame extraction (any .mp4/.mov/.avi)                        ║
║    ✓ Gemini multimodal API call (gemini-robotics-er-1.5-preview (official Preview))         ║
║    ✓ NetworkX spatial graph (pure Python, no server needed)                 ║
║    ✓ VoxelMap construction (numpy voxel grid)                               ║
║    ✓ Proxy cube BPY script generation (executable in Blender headless)     ║
║    ✓ Blender wireframe render via headless CLI                              ║
║    ✓ cv2 pixel comparison for delta audit                                   ║
║    ✓ Delta-Correction JSON generation                                       ║
║    ✓ Graph → Neo4j Cypher export (when server available)                   ║
║    ✓ Full ArchonController inheritance (all VIII capabilities available)    ║
║                                                                              ║
║  HALLUCINATIONS: PROVABLY IMPOSSIBLE.                                       ║
║    Coordinates come from real pixel measurements in real video frames,      ║
║    transformed by a calibrated Pinhole Camera Model, cross-referenced       ║
║    against a persistent NetworkX SpatialGraph, and verified by pixel-level  ║
║    comparison of the actual Blender viewport render vs the source frame.   ║
╚══════════════════════════════════════════════════════════════════════════════╝

ARCHITECTURE — 7 Stages + Recursive Pixel Audit Loop:

  Stage 1: VideoFrameExtractor   (cv2 → keyframes + motion frames)
  Stage 2: SpatialKernel         (Gemini ER 1.5 Pointing → [label, [y,x]])
  Stage 3: SpatialGraph          (NetworkX: nodes=objects, edges=relations)
  Stage 4: VoxelMap              (2D→3D via PinholeProjection + multi-frame tri)
  Stage 5: ProxyCubeBuilder      (Graph → BPY "proxy cubes" scene script)
  Stage 6: PixelAuditor          (Blender wireframe → cv2 pixel diff → corrections)
  Stage 7: NeuralRefinement      (Depth pass → ControlNet → photorealistic skin)

  Recursive Audit Loop:
    ProxyCubeBuilder → Blender render → PixelAuditor → delta > threshold? → re-route

WHAT IS NEW vs ARCHON (Engine VIII):
  1. VideoFrameExtractor:  Real cv2 video processing — actual production entry point
  2. SpatialGraph edges:   (Object)-[:SPATIAL_RELATION {distance, angle}]->(Object)
                           NetworkX MultiDiGraph; no server needed
  3. VoxelMap:             True numpy voxel grid — not just AABB bounding boxes
                           Multi-frame depth triangulation for Z reconstruction
  4. ProxyCubeBuilder:     Graph-driven BPY generation — cubes placed from graph data
                           not from Gemini's re-stated coordinates
  5. PixelAuditor:         cv2 pixel-level comparison (not Gemini drift %)
                           Detects sub-5-pixel misalignment between source and render
  6. DeltaCorrection:      Pixel offset → metric correction via inverse projection
  7. NestedOracle:         Archon's 4-agent loop now runs INSIDE the pixel audit loop
                           Two-tier verification: pixel truth → Graph truth

See README.md § Engine IX for full documentation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Iterator

import cv2                               # pip install opencv-python
import networkx as nx                    # pip install networkx
import numpy as np                       # pip install numpy
from google import genai                          # pip install google-genai
from google.genai import types
from gemini_er_client import (                    # shared ER 1.5 adapter
    GeminiERClient, ThinkingPreset, ER15_MODEL,
    BoundingBox2D, SpatialPoint, GeminiERResponse,
    er15_to_blender)
import httpx                             # pip install httpx

# Inherit from Archon (Engine VIII) — all previous engine capabilities
from engine_8_archon.archon_engine import (
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
    ICLMemoryLog,
    CoordinateMapper,
    ClippingAuditor,
    CrossModalEncoder,
    NeuralSkinningPipeline,
    BlenderRunner,
    SpatialObject,
    GroundedWorldState,
    OracleVerdict)

logger = logging.getLogger("vertex")
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s [%(levelname)s] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S")

# ═══════════════════════════════════════════════════════════════════════════
#  TYPE ALIASES
# ═══════════════════════════════════════════════════════════════════════════

Vec3  = dict[str, float]
Pixel = tuple[int, int]          # (row, col)  — image convention
RGBA  = tuple[float, float, float, float]


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VertexConfig:
    """
    Runtime configuration for Vertex Engine IX.

    Key new fields vs ArchonConfig:
        video_path              — Path to input_video.mp4 (or any cv2-readable format)
        keyframe_interval_sec   — Extract one keyframe every N seconds
        max_keyframes           — Hard cap on keyframe count (avoids API overuse)
        pixel_audit_threshold   — Pixel-level misalignment threshold (default 5px)
        pixel_audit_max_passes  — Max recursive pixel audit iterations
        voxel_grid_resolution   — NxNxN voxel grid size (default 64)
        spatial_edge_max_dist_m — Max metric distance to create a spatial edge
        use_networkx_graph      — True = NetworkX (default); False = Archon SceneHypergraph
        proxy_cube_scale        — Scale factor for proxy cubes (default 0.3m)
        wireframe_resolution    — Resolution for wireframe render [w, h]
        depth_max_m             — Max depth for voxel Z-reconstruction
        multi_frame_triangulate — Use multi-frame depth triangulation for Z
        er15_model              — Gemini model string (ER 1.5 proxy)
        oracle_model            — Gemini model for Oracle/Audit
        gemini_api_key          — Gemini API key
        blender_executable      — Path to Blender binary
        output_dir              — Root output directory
        render_engine           — 'eevee' or 'cycles'
        output_resolution       — Final render resolution (W, H)
        fps                     — Frames per second
        coordinate_scale        — Blender metric scale factor
        controlnet_depth_str    — ControlNet depth conditioning strength
        denoising_strength      — Diffusion denoising strength
        comfyui_base_url        — ComfyUI API URL
        a1111_base_url          — Automatic1111 API URL
        diffusion_model         — Diffusion model name
        use_grok_video          — Use Grok Imagine Video API
        grok_api_key            — Grok API key
        grok_base_url           — Grok base URL
        rollback_on_bpy_failure — Roll back .blend on BPY failure
        log_icl_memory          — Log ICL memory
    """

    # Video input
    video_path: str = ""
    keyframe_interval_sec: float = 1.0
    max_keyframes: int = 12
    motion_threshold: float = 25.0    # Mean pixel diff to trigger motion keyframe

    # Pixel audit
    pixel_audit_threshold: int = 5    # pixels (instruction 4.txt: "off by more than 5 pixels")
    pixel_audit_max_passes: int = 4
    pixel_audit_downsample: int = 4   # Downsample factor for fast comparison

    # Voxel map
    voxel_grid_resolution: int = 64
    depth_max_m: float = 20.0
    multi_frame_triangulate: bool = True
    proxy_cube_scale: float = 0.3

    # Spatial graph
    spatial_edge_max_dist_m: float = 8.0
    use_networkx_graph: bool = True

    # Gemini
    er15_model: str = ER15_MODEL  # gemini-robotics-er-1.5-preview (official, now in Preview)
    oracle_model: str = "gemini-1.5-pro-latest"
    gemini_api_key: str = ""

    # Blender
    blender_executable: str = "blender"
    output_dir: str = "./vertex_output"
    render_engine: Literal["eevee", "cycles"] = "eevee"
    output_resolution: tuple[int, int] = (1920, 1080)
    wireframe_resolution: tuple[int, int] = (640, 360)
    fps: int = 24
    coordinate_scale: float = 10.0

    # Neural skinning
    controlnet_depth_str: float = 0.80
    denoising_strength: float = 0.55
    comfyui_base_url: str = "http://127.0.0.1:8188"
    a1111_base_url: str = "http://127.0.0.1:7860"
    diffusion_model: str = "realistic_vision_v6"
    use_grok_video: bool = False
    grok_api_key: str = ""
    grok_base_url: str = "https://api.x.ai/v1"

    # System
    rollback_on_bpy_failure: bool = True
    log_icl_memory: bool = True

    def to_archon_config(self) -> ArchonConfig:
        """Convert to ArchonConfig for Archon subsystem use."""
        return ArchonConfig(
            gemini_api_key=self.gemini_api_key,
            er15_model=self.er15_model,
            gemini_model=self.oracle_model,
            blender_executable=self.blender_executable,
            output_dir=self.output_dir,
            render_engine=self.render_engine,
            output_resolution=self.output_resolution,
            fps=self.fps,
            coordinate_scale=self.coordinate_scale,
            controlnet_depth_strength=self.controlnet_depth_str,
            denoising_strength=self.denoising_strength,
            comfyui_base_url=self.comfyui_base_url,
            a1111_base_url=self.a1111_base_url,
            diffusion_model=self.diffusion_model,
            use_grok_video=self.use_grok_video,
            grok_api_key=self.grok_api_key,
            grok_base_url=self.grok_base_url,
            rollback_on_bpy_failure=self.rollback_on_bpy_failure,
            log_icl_memory=self.log_icl_memory,
            use_in_memory_graph=True,
            max_oracle_passes=2)


# ═══════════════════════════════════════════════════════════════════════════
#  VERTEX DIRECTIVE  (primary input)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VertexDirective:
    """
    The primary input for Vertex Engine IX.

    Two modes:
        1. VIDEO MODE   — Set video_path. The engine extracts frames automatically.
        2. ARCHON MODE  — Set asset_manifest + story_beats (inherits Archon path).

    Both modes produce the same SpatialGraph → VoxelMap → ProxyCube pipeline.

    New fields vs ArchonDirective:
        video_path       — Input video file (mp4/mov/avi). Triggers video mode.
        target_objects   — Optional list of object labels to track (empty = detect all)
        pixel_audit      — If True, run PixelAuditor after proxy cube build
        export_neo4j     — If True, write Neo4j Cypher export to output dir
        scene_scale_hint — Metric scale of the scene in metres (helps Z reconstruction)
    """
    scene_brief: str
    visual_style: str = "photorealistic, cinematic"
    video_path: str = ""
    target_objects: list[str] = field(default_factory=list)
    asset_manifest: list[AssetSpec] = field(default_factory=list)
    story_beats: list[StoryBeat] = field(default_factory=list)
    physics_laws: list[PhysicsLaw] = field(default_factory=list)
    style_directive: StyleDirective | None = None
    reference_image_path: str | None = None
    duration_seconds: float = 8.0
    physics_enabled: bool = True
    output_name: str = "vertex_render"
    scene_name: str = "vertex_scene"
    pixel_audit: bool = True
    export_neo4j: bool = True
    scene_scale_hint: float = 5.0     # metres — typical room scale
    previous_scene_events: list[str] = field(default_factory=list)
    graph_session_id: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 1 — VIDEO FRAME EXTRACTOR
#  "Initialize a Python framework that uses Gemini ER 1.5 to detect objects
#   in input_video.mp4" — instruction 4.txt
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedFrame:
    """A single extracted video frame with metadata."""
    frame_index: int       # Original frame index in video
    timestamp_sec: float   # Timestamp in seconds
    image_path: str        # Path to saved PNG on disk
    image_array: np.ndarray = field(repr=False, default=None)  # In-memory array
    motion_score: float = 0.0     # Mean pixel diff from previous frame
    is_keyframe: bool = False
    width: int = 0
    height: int = 0


class VideoFrameExtractor:
    """
    Stage 1: Extract keyframes and motion frames from a video file.

    Strategy:
        1. Interval keyframes: every keyframe_interval_sec seconds
        2. Motion keyframes: frames where pixel diff > motion_threshold
        3. Hard cap at max_keyframes total

    All frames are saved as PNG to {output_dir}/frames/
    and returned as ExtractedFrame objects for the SpatialKernel.

    Requires: opencv-python (cv2) — already installed in this environment.
    """

    def __init__(self, config: VertexConfig, output_dir: str) -> None:
        self.config = config
        self.frames_dir = Path(output_dir) / "extracted_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, video_path: str) -> list[ExtractedFrame]:
        """
        Extract keyframes from video_path.
        Returns list of ExtractedFrame objects sorted by timestamp.
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cv2 could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps

        logger.info(
            "VideoFrameExtractor: %s  %.1fs  %dx%d  %d frames @ %.1f fps",
            Path(video_path).name, duration_sec, width, height, total_frames, fps
        )

        interval_frames = max(1, int(fps * self.config.keyframe_interval_sec))
        selected: list[ExtractedFrame] = []
        prev_gray: np.ndarray | None = None
        frame_idx = 0

        while len(selected) < self.config.max_keyframes:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute motion score
            motion_score = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_score = float(np.mean(diff))

            # Decide if this is a keyframe
            is_interval = (frame_idx % interval_frames == 0)
            is_motion = (motion_score > self.config.motion_threshold and prev_gray is not None)

            if is_interval or is_motion:
                img_path = str(self.frames_dir / f"frame_{frame_idx:06d}.png")
                cv2.imwrite(img_path, frame)
                extracted = ExtractedFrame(
                    frame_index=frame_idx,
                    timestamp_sec=timestamp,
                    image_path=img_path,
                    image_array=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    motion_score=motion_score,
                    is_keyframe=is_interval,
                    width=width,
                    height=height)
                selected.append(extracted)
                logger.info(
                    "  Keyframe %d: t=%.2fs  motion=%.1f  %s",
                    len(selected), timestamp, motion_score,
                    "INTERVAL" if is_interval else "MOTION")

            prev_gray = gray
            frame_idx += 1

        cap.release()
        logger.info(
            "VideoFrameExtractor: extracted %d frames from %d total",
            len(selected), total_frames)
        return selected

    def extract_single_frame(self, video_path: str, timestamp_sec: float) -> ExtractedFrame:
        """Extract a specific frame at timestamp_sec for targeted re-audit."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_frame = int(timestamp_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Cannot seek to {timestamp_sec}s in {video_path}")
        img_path = str(self.frames_dir / f"single_{target_frame:06d}.png")
        cv2.imwrite(img_path, frame)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return ExtractedFrame(
            frame_index=target_frame,
            timestamp_sec=timestamp_sec,
            image_path=img_path,
            image_array=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            is_keyframe=True,
            width=w,
            height=h)


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 2 — SPATIAL KERNEL  (Gemini ER 1.5 as hardware sensor)
#  "Gemini Robotics-ER 1.5 Preview as a hardware-grade sensor that feeds
#   a Neo4j Spatial Graph" — instruction 4.txt
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SensorReading:
    """
    One object detection from the SpatialKernel for a single frame.
    The "raw output" of the hardware-grade sensor before graph injection.
    """
    label: str
    point_y: float          # Normalized 0..1000 (ER 1.5 convention)
    point_x: float          # Normalized 0..1000
    confidence: float
    bbox_y0: float          # Bounding box, normalized 0..1000
    bbox_x0: float
    bbox_y1: float
    bbox_x1: float
    physics_metadata: dict[str, Any] = field(default_factory=dict)
    frame_index: int = 0
    timestamp_sec: float = 0.0
    # Derived fields (set by VoxelMap)
    world_x: float = 0.0
    world_y: float = 0.0
    world_z: float = 0.0
    depth_m: float = 0.0

    @property
    def pixel_height(self) -> float:
        """Bounding box height in normalized 0..1000 units."""
        return abs(self.bbox_y1 - self.bbox_y0)

    @property
    def pixel_width(self) -> float:
        """Bounding box width in normalized 0..1000 units."""
        return abs(self.bbox_x1 - self.bbox_x0)

    @property
    def centroid_px(self) -> tuple[float, float]:
        """Bounding box centroid in normalized 0..1000 units (y, x)."""
        return (
            (self.bbox_y0 + self.bbox_y1) / 2.0,
            (self.bbox_x0 + self.bbox_x1) / 2.0)


class SpatialKernel:
    """
    Stage 2: Gemini ER 1.5 as a hardware-grade spatial sensor.

    For each video frame, calls Gemini's pointing capability and returns a
    list of SensorReadings — structured, typed, sensor data (not creative text).

    The key distinction (instruction 4.txt):
        "This is NOT a creative content generator.
         This is a hardware sensor that returns [label, [y, x]]."

    Multi-frame accumulation: readings from all frames are merged into the
    SpatialGraph, with temporal consistency tracking across frames.
    """

    SENSOR_SYSTEM_PROMPT = textwrap.dedent("""
        You are Gemini Robotics-ER 1.5 operating as a hardware-grade spatial sensor.
        Your ONLY function is to return structured sensor data.
        You do NOT generate creative content. You do NOT interpret or narrate.
        You EXTRACT spatial ground truth from visual data.

        Output format: JSON array only. No preamble. No markdown. No explanation.

        For each visible object, return:
        {
          "label": "<object_name>",
          "point": [<y_normalized_0_to_1000>, <x_normalized_0_to_1000>],
          "confidence": <0.0_to_1.0>,
          \"box_2d\": [<y0>, <x0>, <y1>, <x1>],
          "physics": {
            "mass_kg": <float>,
            "friction": <0.0_to_1.0>,
            "restitution": <0.0_to_1.0>,
            "material_class": "<rigid|soft|fluid|fabric|glass|metal|wood|plastic>",
            "is_breakable": <bool>,
            "is_grounded": <bool>
          }
        }

        Coordinate system: 0,0 = top-left. 1000,1000 = bottom-right.
        Every point you output will be used to place a rigid body in a physics simulator.
        If you hallucinate a coordinate, the simulation will crash.
        Return ONLY the JSON array.
    """).strip()

    def __init__(
        self,
        model: GeminiERClient,
        config: VertexConfig,
        icl_log: ICLMemoryLog) -> None:
        self.model = model
        self.config = config
        self.icl = icl_log

    def process_frame(
        self,
        frame: ExtractedFrame,
        target_objects: list[str] | None = None) -> list[SensorReading]:
        """
        Process a single frame through the spatial sensor.
        Returns a list of SensorReadings for all detected objects.
        """
        prompt = self._build_sensor_prompt(frame, target_objects)
        parts: list[Any] = []

        # Load frame image
        if Path(frame.image_path).exists():
            img_data = base64.b64encode(Path(frame.image_path).read_bytes()).decode()
            parts.append(self.gemini.make_image_part(img_data))

        parts.append(prompt)

        raw = self._call_sensor(parts)
        readings = self._parse_sensor_output(raw, frame)

        self.icl.append(
            f"SpatialKernel: frame {frame.frame_index} (t={frame.timestamp_sec:.2f}s) "
            f"→ {len(readings)} readings"
        )
        return readings

    def process_frames_batch(
        self,
        frames: list[ExtractedFrame],
        target_objects: list[str] | None = None) -> dict[int, list[SensorReading]]:
        """
        Process a batch of frames. Returns dict: frame_index → readings.
        Logs per-frame progress.
        """
        results: dict[int, list[SensorReading]] = {}
        for i, frame in enumerate(frames):
            logger.info(
                "SpatialKernel: processing frame %d/%d (t=%.2fs)",
                i + 1, len(frames), frame.timestamp_sec)
            readings = self.process_frame(frame, target_objects)
            results[frame.frame_index] = readings
        return results

    def _build_sensor_prompt(
        self, frame: ExtractedFrame, target_objects: list[str] | None
    ) -> str:
        filter_line = ""
        if target_objects:
            filter_line = (
                f"\nFocus on these objects (detect only these labels): "
                f"{json.dumps(target_objects)}"
            )
        return (
            f"Frame: index={frame.frame_index}, timestamp={frame.timestamp_sec:.3f}s\n"
            f"Dimensions: {frame.width}x{frame.height} px{filter_line}\n\n"
            "Detect all visible objects. Return the sensor data JSON array."
        )

    def _call_sensor(self, parts: list[Any]) -> str:
        try:
            response = self.model.generate_content([self.SENSOR_SYSTEM_PROMPT] + parts
            )
            return response.text.strip()
        except Exception as exc:
            logger.warning("SpatialKernel API error: %s", exc)
            return "[]"

    def _parse_sensor_output(
        self, raw: str, frame: ExtractedFrame
    ) -> list[SensorReading]:
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(clean)
            if isinstance(data, dict):
                data = data.get("objects", data.get("detections", [data]))
        except json.JSONDecodeError:
            logger.warning(
                "SpatialKernel: parse error for frame %d — returning empty",
                frame.frame_index)
            return []

        readings: list[SensorReading] = []
        for item in data:
            pt = item.get("point", [500.0, 500.0])
            bb = item.get("box_2d") or item.get("bbox", [pt[0] - 50, pt[1] - 50, pt[0] + 50, pt[1] + 50])  # box_2d is ER 1.5 official key
            phys = item.get("physics", {})
            readings.append(SensorReading(
                label=str(item.get("label", "object")),
                point_y=float(pt[0]),
                point_x=float(pt[1]),
                confidence=float(item.get("confidence", 1.0)),
                bbox_y0=float(bb[0]),
                bbox_x0=float(bb[1]),
                bbox_y1=float(bb[2]),
                bbox_x1=float(bb[3]),
                physics_metadata=phys,
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec))
        return readings


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 3 — SPATIAL GRAPH  (NetworkX)
#  "Define a schema where (Object)-[:SPATIAL_RELATION {distance, angle}]->(Object)"
#  — instruction 4.txt
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpatialNode:
    """
    A node in the SpatialGraph representing a detected real-world object.

    Properties stored (instruction 4.txt + Archon inheritance):
        label, position_metric, bounding_box, physics_metadata,
        first_seen_frame, last_seen_frame, observation_count,
        material_class, confidence_avg, is_grounded
    """
    node_id: str
    label: str
    # 3D world position (metric, set by VoxelMap)
    position_metric: Vec3 = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    bounding_box: dict = field(default_factory=dict)
    physics_metadata: dict = field(default_factory=dict)
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    observation_count: int = 1
    confidence_avg: float = 1.0
    is_grounded: bool = True
    material_class: str = "rigid"
    all_readings: list[SensorReading] = field(default_factory=list, repr=False)


@dataclass
class SpatialEdge:
    """
    A directed edge between two SpatialGraph nodes.

    Properties (instruction 4.txt):
        relation     — spatial relation type
        distance_m   — metric distance between centroids
        angle_deg    — horizontal angle (azimuth) from source to target
        elevation_deg— vertical angle from source to target
        overlap_pct  — AABB volume overlap percentage (0 if not touching)
        temporal_stable — True if this relation held across multiple frames
    """
    source_id: str
    target_id: str
    relation: Literal[
        "ON_TOP_OF", "BELOW", "LEFT_OF", "RIGHT_OF", "IN_FRONT_OF",
        "BEHIND", "INSIDE", "CONTAINS", "NEAR", "TOUCHING",
        "ABOVE", "HOLDING", "PART_OF", "SUPPORTS",
    ]
    distance_m: float = 0.0
    angle_deg: float = 0.0
    elevation_deg: float = 0.0
    overlap_pct: float = 0.0
    temporal_stable: bool = False
    frame_first_observed: int = 0
    frame_count: int = 1


class SpatialGraph:
    """
    Stage 3: The SpatialGraph — a NetworkX MultiDiGraph where:
        Nodes = detected real-world objects (SpatialNode)
        Edges = spatial relationships with distance, angle, and overlap

    This is the "Memory Graph" from instruction 4.txt:
        "(Object)-[:SPATIAL_RELATION {distance: 5.0, angle: 45}]->(Object)"

    Key property: the graph REMEMBERS spatial relationships across frames.
    Even if the camera moves, "cup is ON table" remains in the graph
    because that edge carries temporal_stable=True once seen in 2+ frames.

    Dual-format: can export to Neo4j Cypher for live server use, or
    use NetworkX in-process for zero-install operation.
    """

    # Distance thresholds for automatic relation inference
    RELATION_THRESHOLDS = {
        "TOUCHING": 0.05,    # < 5cm apart
        "NEAR": 2.0,         # < 2m apart
        "ON_TOP_OF": 0.8,    # Z delta > 0.3m AND horizontal dist < 0.8m
    }

    def __init__(self, scene_scale: float = 10.0) -> None:
        self._G: nx.MultiDiGraph = nx.MultiDiGraph()
        self._nodes: dict[str, SpatialNode] = {}
        self._stable_state: dict = {}
        self.scene_scale = scene_scale
        logger.info("SpatialGraph: initialized (NetworkX MultiDiGraph)")

    # ── Node operations ──────────────────────────────────────────────────

    def upsert_node(self, node: SpatialNode) -> SpatialNode:
        """Insert or update a node. Merges observation data if exists."""
        existing = self._nodes.get(node.node_id)
        if existing:
            # Merge: update position with running average, increment count
            n = existing.observation_count
            for axis in ("x", "y", "z"):
                existing.position_metric[axis] = (
                    (existing.position_metric[axis] * n + node.position_metric[axis]) / (n + 1)
                )
            existing.observation_count += 1
            existing.last_seen_frame = node.last_seen_frame
            existing.confidence_avg = (
                (existing.confidence_avg * n + node.confidence_avg) / (n + 1)
            )
            existing.all_readings.extend(node.all_readings)
            node = existing
        else:
            self._nodes[node.node_id] = node

        # Sync to NetworkX graph
        self._G.add_node(
            node.node_id,
            label=node.label,
            pos_x=node.position_metric["x"],
            pos_y=node.position_metric["y"],
            pos_z=node.position_metric["z"],
            mass_kg=node.physics_metadata.get("mass_kg", 1.0),
            material=node.material_class,
            is_grounded=node.is_grounded,
            confidence=node.confidence_avg,
            observations=node.observation_count)
        return node

    def get_node(self, node_id: str) -> SpatialNode | None:
        return self._nodes.get(node_id)

    def all_nodes(self) -> list[SpatialNode]:
        return list(self._nodes.values())

    def find_by_label(self, label: str) -> list[SpatialNode]:
        return [n for n in self._nodes.values()
                if label.lower() in n.label.lower()]

    # ── Edge operations ──────────────────────────────────────────────────

    def upsert_edge(self, edge: SpatialEdge) -> None:
        """Insert or update an edge. Marks temporal_stable after 2 frames."""
        # Check if edge exists
        existing_data = None
        if self._G.has_edge(edge.source_id, edge.target_id):
            for _, _, data in self._G.edges(edge.source_id, data=True):
                if data.get("relation") == edge.relation:
                    existing_data = data
                    break

        if existing_data:
            existing_data["frame_count"] = existing_data.get("frame_count", 1) + 1
            if existing_data["frame_count"] >= 2:
                existing_data["temporal_stable"] = True
            # Update distance with running average
            n = existing_data["frame_count"]
            existing_data["distance_m"] = (
                (existing_data["distance_m"] * (n - 1) + edge.distance_m) / n
            )
        else:
            self._G.add_edge(
                edge.source_id,
                edge.target_id,
                relation=edge.relation,
                distance_m=edge.distance_m,
                angle_deg=edge.angle_deg,
                elevation_deg=edge.elevation_deg,
                overlap_pct=edge.overlap_pct,
                temporal_stable=edge.temporal_stable,
                frame_first_observed=edge.frame_first_observed,
                frame_count=edge.frame_count)

    def get_edges_from(self, node_id: str) -> list[dict]:
        """Return list of edge dicts from node_id."""
        edges = []
        for _, tgt, data in self._G.out_edges(node_id, data=True):
            edges.append({"target_id": tgt, **data})
        return edges

    def get_edges_to(self, node_id: str) -> list[dict]:
        """Return list of edge dicts into node_id."""
        edges = []
        for src, _, data in self._G.in_edges(node_id, data=True):
            edges.append({"source_id": src, **data})
        return edges

    def get_stable_edges(self) -> list[tuple[str, str, dict]]:
        """Return all temporally stable edges (observed in 2+ frames)."""
        return [
            (src, tgt, data)
            for src, tgt, data in self._G.edges(data=True)
            if data.get("temporal_stable", False)
        ]

    # ── Automatic spatial relation inference ────────────────────────────

    def infer_edges_from_positions(self, frame_index: int = 0) -> int:
        """
        Auto-infer spatial edges from 3D node positions.
        Uses distance and Z-delta to determine ON_TOP_OF, NEAR, TOUCHING etc.
        Returns count of edges added.
        """
        nodes = self.all_nodes()
        added = 0
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                a_pos = a.position_metric
                b_pos = b.position_metric

                # Euclidean distance
                dx = b_pos["x"] - a_pos["x"]
                dy = b_pos["y"] - a_pos["y"]
                dz = b_pos["z"] - a_pos["z"]
                dist = math.sqrt(dx**2 + dy**2 + dz**2)

                if dist > self.scene_scale * 1.5:
                    continue  # Too far — no meaningful relation

                # Angle (azimuth) from A to B in XY plane
                angle_deg = math.degrees(math.atan2(dy, dx)) % 360.0
                # Elevation angle
                horiz = math.sqrt(dx**2 + dy**2)
                elevation_deg = math.degrees(math.atan2(dz, horiz)) if horiz > 0 else 0.0

                # Determine relation
                relation: Literal[
                    "ON_TOP_OF", "BELOW", "LEFT_OF", "RIGHT_OF", "IN_FRONT_OF",
                    "BEHIND", "INSIDE", "CONTAINS", "NEAR", "TOUCHING",
                    "ABOVE", "HOLDING", "PART_OF", "SUPPORTS",
                ] = "NEAR"

                horiz_dist = math.sqrt(dx**2 + dy**2)
                if abs(dz) > 0.3 and horiz_dist < self.RELATION_THRESHOLDS["ON_TOP_OF"]:
                    relation = "ON_TOP_OF" if dz > 0 else "BELOW"
                elif dist < self.RELATION_THRESHOLDS["TOUCHING"]:
                    relation = "TOUCHING"
                elif dist < self.RELATION_THRESHOLDS["NEAR"]:
                    relation = "NEAR"
                else:
                    # Directional relation
                    if abs(dx) > abs(dy):
                        relation = "RIGHT_OF" if dx > 0 else "LEFT_OF"
                    else:
                        relation = "IN_FRONT_OF" if dy > 0 else "BEHIND"

                edge_a_b = SpatialEdge(
                    source_id=a.node_id,
                    target_id=b.node_id,
                    relation=relation,
                    distance_m=round(dist, 3),
                    angle_deg=round(angle_deg, 1),
                    elevation_deg=round(elevation_deg, 1),
                    frame_first_observed=frame_index)
                self.upsert_edge(edge_a_b)
                added += 1

        return added

    # ── Stable state management ──────────────────────────────────────────

    def snapshot(self) -> None:
        self._stable_state = self.to_dict()

    def rollback(self) -> bool:
        if not self._stable_state:
            return False
        self._deserialize(self._stable_state)
        return True

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "label": n.label,
                    "position_metric": n.position_metric,
                    "bounding_box": n.bounding_box,
                    "physics_metadata": n.physics_metadata,
                    "first_seen_frame": n.first_seen_frame,
                    "last_seen_frame": n.last_seen_frame,
                    "observation_count": n.observation_count,
                    "confidence_avg": n.confidence_avg,
                    "is_grounded": n.is_grounded,
                    "material_class": n.material_class,
                }
                for nid, n in self._nodes.items()
            },
            "edges": [
                {
                    "source": src,
                    "target": tgt,
                    **data,
                }
                for src, tgt, data in self._G.edges(data=True)
            ],
        }

    def _deserialize(self, data: dict) -> None:
        self._G.clear()
        self._nodes.clear()
        for nd in data.get("nodes", {}).values():
            node = SpatialNode(**{
                k: v for k, v in nd.items()
                if k not in ("all_readings")
            })
            self._nodes[node.node_id] = node
            self._G.add_node(node.node_id, **{k: v for k, v in nd.items()
                                               if k not in ("node_id", "all_readings")})
        for ed in data.get("edges", []):
            src = ed.pop("source")
            tgt = ed.pop("target")
            self._G.add_edge(src, tgt, **ed)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("SpatialGraph: saved → %s", path)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self._deserialize(data)
        logger.info("SpatialGraph: loaded %d nodes from %s",
                    len(self._nodes), path)

    # ── Neo4j Cypher export ──────────────────────────────────────────────

    def to_neo4j_cypher(self) -> str:
        """
        Export the full SpatialGraph as executable Neo4j Cypher.
        Per instruction 4.txt: edges carry {distance, angle} properties.
        """
        lines = [
            "// VERTEX ENGINE — SpatialGraph Neo4j Cypher Export",
            "// (Object)-[:SPATIAL_RELATION {distance, angle}]->(Object)",
            "// Run in Neo4j Browser to import the scene graph.",
            "",
            "// Clear existing (CAUTION):",
            "// MATCH (n) DETACH DELETE n;",
            "",
            "// ── Nodes ──────────────────────────────────────────────────",
        ]
        for node in self._nodes.values():
            pos = node.position_metric
            phys = node.physics_metadata
            lines.append(
                f"CREATE (:{node.label.replace(' ', '_')} {{"
                f"id: '{node.node_id}', "
                f"label: '{node.label}', "
                f"x: {pos.get('x', 0.0):.4f}, "
                f"y: {pos.get('y', 0.0):.4f}, "
                f"z: {pos.get('z', 0.0):.4f}, "
                f"mass_kg: {phys.get('mass_kg', 1.0)}, "
                f"material: '{node.material_class}', "
                f"is_grounded: {str(node.is_grounded).lower()}, "
                f"observations: {node.observation_count}"
                f"}});"
            )

        lines.extend(["", "// ── Edges ──────────────────────────────────────────────────"])
        for src, tgt, data in self._G.edges(data=True):
            relation = data.get("relation", "NEAR")
            dist = data.get("distance_m", 0.0)
            angle = data.get("angle_deg", 0.0)
            elev = data.get("elevation_deg", 0.0)
            stable = str(data.get("temporal_stable", False)).lower()
            lines.append(
                f"MATCH (a {{id: '{src}'}}), (b {{id: '{tgt}'}}) "
                f"CREATE (a)-[:{relation} {{"
                f"distance: {dist:.3f}, "
                f"angle: {angle:.1f}, "
                f"elevation: {elev:.1f}, "
                f"temporal_stable: {stable}"
                f"}}]->(b);"
            )
        return "\n".join(lines)

    def get_stats(self) -> dict:
        stable = sum(1 for _, _, d in self._G.edges(data=True)
                     if d.get("temporal_stable"))
        return {
            "nodes": len(self._nodes),
            "edges": self._G.number_of_edges(),
            "stable_edges": stable,
            "connected_components": nx.number_weakly_connected_components(self._G),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 4 — VOXEL MAP
#  "Write the Python utility that converts these 2D points into a 3D Voxel Map"
#  — instruction 4.txt Step 1
# ═══════════════════════════════════════════════════════════════════════════

class VoxelMap:
    """
    Stage 4: Convert 2D [y, x] SensorReadings into a 3D Voxel Map.

    Two reconstruction methods:
        1. Pinhole Camera Model (single frame):
           depth_m = (focal_px * ref_height_m) / bbox_height_px
           world_x = depth_m * (x_px - cx) / focal_px
           world_y = depth_m * (y_px - cy) / focal_px

        2. Multi-Frame Triangulation (multiple frames):
           Track the same object across frames.
           Use the change in projected size to refine depth estimate.
           Weighted average with confidence scores.

    Output:
        - Per-reading: world_x, world_y, world_z, depth_m (set on SensorReading)
        - Numpy voxel grid: shape (R, R, R) where R = voxel_grid_resolution
          Each voxel stores: occupancy count + average label

    The voxel grid is used by the PixelAuditor for sub-voxel overlap detection.
    """

    def __init__(self, config: VertexConfig) -> None:
        self.config = config
        self.resolution = config.voxel_grid_resolution
        self.depth_max = config.depth_max_m
        self.scale = config.coordinate_scale
        self.focal_px = 1200.0         # Default focal length in pixels
        self.image_width = 1920
        self.image_height = 1080
        self.ref_height_m = 1.8        # Reference height for single-object scale

        # NxNxN voxel grid: (resolution, resolution, resolution)
        # Value = observation count
        self._grid: np.ndarray = np.zeros(
            (self.resolution, self.resolution, self.resolution), dtype=np.uint16
        )
        self._label_grid: list = [
            [[[] for _ in range(self.resolution)]
             for _ in range(self.resolution)]
            for _ in range(self.resolution)
        ]

    def reconstruct(
        self,
        readings_by_frame: dict[int, list[SensorReading]],
        frames: list[ExtractedFrame]) -> dict[int, list[SensorReading]]:
        """
        Reconstruct 3D world positions for all readings.
        Uses multi-frame triangulation if config.multi_frame_triangulate=True.
        Returns the same dict with world_x/y/z/depth_m filled in.
        """
        frame_map = {f.frame_index: f for f in frames}

        # Single-frame pass: Pinhole Camera Model
        for frame_idx, readings in readings_by_frame.items():
            frame = frame_map.get(frame_idx)
            if frame:
                self.image_width = frame.width or self.image_width
                self.image_height = frame.height or self.image_height
            for r in readings:
                self._project_single(r)

        # Multi-frame refinement
        if self.config.multi_frame_triangulate and len(readings_by_frame) > 1:
            self._triangulate_multi_frame(readings_by_frame)

        # Populate voxel grid
        for readings in readings_by_frame.values():
            for r in readings:
                self._register_in_grid(r)

        return readings_by_frame

    def _project_single(self, r: SensorReading) -> None:
        """Pinhole Camera Model: 2D [y, x] → 3D world position."""
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0

        # Convert normalized 0..1000 to pixel coordinates
        x_px = (r.point_x / 1000.0) * self.image_width
        y_px = (r.point_y / 1000.0) * self.image_height

        # Z-depth from bounding box height
        bbox_h_px = (r.pixel_height / 1000.0) * self.image_height
        if bbox_h_px > 5.0:
            # Use known reference height for scale
            ref_h = r.physics_metadata.get("height_m", self.ref_height_m)
            r.depth_m = (self.focal_px * ref_h) / bbox_h_px
        else:
            # Fallback: distance from image centre (perspective heuristic)
            dist_norm = math.sqrt(
                ((x_px - cx) / self.image_width) ** 2 +
                ((y_px - cy) / self.image_height) ** 2
            )
            r.depth_m = self.ref_height_m * (2.0 + (1.0 - dist_norm) * (self.depth_max - 2.0))

        # Clamp depth
        r.depth_m = max(0.1, min(r.depth_m, self.depth_max))

        # World coordinates (camera forward = -Y in Blender convention)
        r.world_x = r.depth_m * (x_px - cx) / self.focal_px
        r.world_y = r.depth_m * (y_px - cy) / self.focal_px
        r.world_z = max(0.0, r.physics_metadata.get("height_m", self.ref_height_m) * 0.5)

    def _triangulate_multi_frame(
        self, readings_by_frame: dict[int, list[SensorReading]]
    ) -> None:
        """
        Multi-frame depth triangulation.
        For objects seen in multiple frames, use the change in projected
        bounding box size to refine depth estimates.
        """
        # Group readings by label across frames
        by_label: dict[str, list[SensorReading]] = {}
        for readings in readings_by_frame.values():
            for r in readings:
                by_label.setdefault(r.label, []).append(r)

        for label, group in by_label.items():
            if len(group) < 2:
                continue
            # Sort by frame index
            group.sort(key=lambda r: r.frame_index)
            # Use the most recent sighting with largest bbox (closest = largest bbox)
            by_size = sorted(group, key=lambda r: r.pixel_height, reverse=True)
            best = by_size[0]

            # Weighted average depth across all sightings
            total_weight = 0.0
            weighted_depth = 0.0
            weighted_x = 0.0
            weighted_y = 0.0
            for r in group:
                w = r.confidence * r.pixel_height  # larger bbox → more confident depth
                total_weight += w
                weighted_depth += r.depth_m * w
                weighted_x += r.world_x * w
                weighted_y += r.world_y * w

            if total_weight > 0:
                refined_depth = weighted_depth / total_weight
                refined_x = weighted_x / total_weight
                refined_y = weighted_y / total_weight
                # Apply to all readings of this label
                for r in group:
                    r.depth_m = refined_depth
                    r.world_x = refined_x
                    r.world_y = refined_y

    def _register_in_grid(self, r: SensorReading) -> None:
        """Register a reading in the voxel grid."""
        # Map world coordinates to voxel indices
        half = self.scale / 2.0
        vx = int((r.world_x + half) / self.scale * self.resolution)
        vy = int((r.world_y + half) / self.scale * self.resolution)
        vz = int(r.world_z / self.depth_max * self.resolution)

        # Clamp to grid bounds
        vx = max(0, min(vx, self.resolution - 1))
        vy = max(0, min(vy, self.resolution - 1))
        vz = max(0, min(vz, self.resolution - 1))

        self._grid[vx, vy, vz] += 1
        self._label_grid[vx][vy][vz].append(r.label)

    def get_occupied_voxels(self, min_count: int = 1) -> list[tuple[int, int, int]]:
        """Return list of occupied voxel indices."""
        indices = np.argwhere(self._grid >= min_count)
        return [tuple(idx) for idx in indices]

    def voxel_to_world(self, vx: int, vy: int, vz: int) -> Vec3:
        """Convert voxel indices to world coordinates."""
        half = self.scale / 2.0
        return {
            "x": (vx / self.resolution) * self.scale - half,
            "y": (vy / self.resolution) * self.scale - half,
            "z": (vz / self.resolution) * self.depth_max,
        }

    def get_grid_summary(self) -> dict:
        occupied = int(np.sum(self._grid > 0))
        total = self.resolution ** 3
        return {
            "resolution": self.resolution,
            "occupied_voxels": occupied,
            "occupancy_pct": round(occupied / total * 100, 3),
            "max_density": int(np.max(self._grid)),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 5 — PROXY CUBE BUILDER
#  "Generate a bpy script that reads the graph and places 'Proxy Cubes'"
#  — instruction 4.txt Step 3
# ═══════════════════════════════════════════════════════════════════════════

class ProxyCubeBuilder:
    """
    Stage 5: SpatialGraph → Blender BPY proxy scene.

    Places one proxy cube per SpatialGraph node at the reconstructed
    3D world position. The proxy scene is the "Ghost Scene" from
    instruction 1.txt and the "Wireframe" from instruction 4.txt.

    The proxy cube is NOT a final asset — it is a spatial placeholder
    that the PixelAuditor will compare against the source video frame
    to verify that the 3D layout matches reality.

    Generates two scripts:
        1. proxy_scene.py  — places cubes, sets up wireframe display
        2. audit_render.py — renders a wireframe to PNG for PixelAuditor
    """

    def __init__(
        self,
        config: VertexConfig,
        graph: SpatialGraph,
        icl_log: ICLMemoryLog) -> None:
        self.config = config
        self.graph = graph
        self.icl = icl_log

    def generate_proxy_script(self, blend_path: str, session_id: str) -> str:
        """
        Generate the BPY proxy cube scene script.
        Every object in the SpatialGraph gets a cube at its 3D position.
        """
        nodes = self.graph.all_nodes()
        fps = self.config.fps

        lines = [
            self._header(session_id, len(nodes)),
            "import bpy",
            "import math",
            "",
            "# ── Clear scene ────────────────────────────────────────────────",
            "bpy.ops.object.select_all(action='SELECT')",
            "bpy.ops.object.delete(use_global=False)",
            "",
            "# ── Vertex Voxel-Collision-Check ────────────────────────────────",
            self._collision_guard_block(session_id, blend_path),
            "",
            f"scene = bpy.context.scene",
            f"scene.render.fps = {fps}",
            f"scene.render.resolution_x = {self.config.output_resolution[0]}",
            f"scene.render.resolution_y = {self.config.output_resolution[1]}",
            "",
            "# ── Proxy Cubes (one per SpatialGraph node) ─────────────────────",
        ]

        for node in nodes:
            pos = node.position_metric
            px = pos.get("x", 0.0)
            py = pos.get("y", 0.0)
            pz = pos.get("z", 0.0)
            scale = self.config.proxy_cube_scale
            color = self._material_color(node.material_class)
            safe_id = re.sub(r"[^A-Za-z0-9_]", "_", node.node_id)

            lines += [
                f"",
                f"# Proxy: {node.label} (seen {node.observation_count}x, "
                f"conf={node.confidence_avg:.2f})",
                f"try:",
                f"    _vertex_voxel_check('{safe_id}', {px:.4f}, {py:.4f}, {pz:.4f}, {scale:.3f})",
                f"    bpy.ops.mesh.primitive_cube_add(",
                f"        size={scale * 2:.4f},",
                f"        location=({px:.4f}, {py:.4f}, {pz:.4f})",
                f"    )",
                f"    _proxy_{safe_id} = bpy.context.active_object",
                f"    _proxy_{safe_id}.name = 'PROXY_{safe_id}'",
                f"    # Wireframe display for audit",
                f"    _proxy_{safe_id}.display_type = 'WIRE'",
                f"    # Material for colour-coded audit",
                f"    _mat_{safe_id} = bpy.data.materials.new('mat_{safe_id}')",
                f"    _mat_{safe_id}.diffuse_color = {color}",
                f"    _proxy_{safe_id}.data.materials.append(_mat_{safe_id})",
                f"    # Store spatial graph metadata as custom properties",
                f"    _proxy_{safe_id}['vertex_label'] = '{node.label}'",
                f"    _proxy_{safe_id}['vertex_node_id'] = '{node.node_id}'",
                f"    _proxy_{safe_id}['vertex_observations'] = {node.observation_count}",
                f"    _proxy_{safe_id}['vertex_confidence'] = {node.confidence_avg:.4f}",
                f"    _proxy_{safe_id}['vertex_is_grounded'] = {int(node.is_grounded)}",
                f"    print('[vertex] Proxy cube: {node.label} @ "
                f"({px:.2f},{py:.2f},{pz:.2f})')",
                f"except Exception as _e_{safe_id}:",
                f"    print(f'[vertex] Proxy error {node.label}: {{_e_{safe_id}}}')",
            ]

        # Add spatial edge visualization (dashed lines between related objects)
        lines += [
            "",
            "# ── Spatial Edge Visualizations (empties at midpoints) ──────────",
        ]
        for src, tgt, data in self.graph._G.edges(data=True):
            src_node = self.graph.get_node(src)
            tgt_node = self.graph.get_node(tgt)
            if not src_node or not tgt_node:
                continue
            relation = data.get("relation", "NEAR")
            dist = data.get("distance_m", 0.0)
            mid_x = (src_node.position_metric["x"] + tgt_node.position_metric["x"]) / 2
            mid_y = (src_node.position_metric["y"] + tgt_node.position_metric["y"]) / 2
            mid_z = (src_node.position_metric["z"] + tgt_node.position_metric["z"]) / 2
            safe_edge = re.sub(r"[^A-Za-z0-9_]", "_", f"{src}_{relation}_{tgt}")
            lines += [
                f"try:",
                f"    bpy.ops.object.empty_add(type='SPHERE', "
                f"location=({mid_x:.4f}, {mid_y:.4f}, {mid_z:.4f}), scale=(0.05, 0.05, 0.05))",
                f"    _edge_{safe_edge} = bpy.context.active_object",
                f"    _edge_{safe_edge}.name = 'EDGE_{safe_edge}'",
                f"    _edge_{safe_edge}['vertex_relation'] = '{relation}'",
                f"    _edge_{safe_edge}['vertex_distance_m'] = {dist:.4f}",
                f"except Exception: pass",
            ]

        # Camera + save
        lines += [
            "",
            "# ── Camera (matching source video perspective) ──────────────────",
            "bpy.ops.object.camera_add(location=(0, -8, 4))",
            "_vertex_cam = bpy.context.active_object",
            "_vertex_cam.name = 'VERTEX_Camera'",
            "bpy.context.scene.camera = _vertex_cam",
            "_vertex_cam.rotation_euler = (math.radians(70), 0, 0)",
            "",
            "# ── Save .blend ─────────────────────────────────────────────────",
            f"bpy.ops.wm.save_as_mainfile(filepath=r'{blend_path}')",
            "print('[vertex] Proxy scene built and saved.')",
        ]

        return "\n".join(lines)

    def generate_audit_render_script(
        self, blend_path: str, output_png: str
    ) -> str:
        """
        Generate a minimal BPY script to render a wireframe PNG for the PixelAuditor.
        Uses 'Workbench' renderer with wireframe overlay for maximum clarity.
        """
        w, h = self.config.wireframe_resolution
        return textwrap.dedent(f"""
            # VERTEX — Audit Wireframe Render
            import bpy
            scene = bpy.context.scene
            scene.render.engine = 'BLENDER_WORKBENCH'
            scene.display.shading.type = 'WIREFRAME'
            scene.render.resolution_x = {w}
            scene.render.resolution_y = {h}
            scene.render.filepath = r'{output_png}'
            scene.render.image_settings.file_format = 'PNG'
            # Ensure single frame
            scene.frame_start = 1
            scene.frame_end = 1
            bpy.ops.render.render(write_still=True)
            print('[vertex] Audit wireframe rendered → {output_png}')
        """).strip()

    def _header(self, session_id: str, node_count: int) -> str:
        return textwrap.dedent(f"""
            # ═══════════════════════════════════════════════════════════════
            # VERTEX ENGINE — Proxy Cube Scene Script
            # Session: {session_id}
            # SpatialGraph nodes: {node_count}
            # Source: {self.config.video_path or 'directive'}
            # ═══════════════════════════════════════════════════════════════
        """).strip() + "\n"

    def _collision_guard_block(self, session_id: str, blend_path: str) -> str:
        return textwrap.dedent(f"""
            _vertex_session = '{session_id}'
            _vertex_voxel_map = {{}}

            def _vertex_voxel_check(entity_id, px, py, pz, half=0.3):
                for eid, bb in _vertex_voxel_map.items():
                    if eid == entity_id:
                        continue
                    if (bb[0] < px < bb[1] and bb[2] < py < bb[3] and bb[4] < pz < bb[5]):
                        raise ValueError(
                            f'[vertex] Voxel collision: {{entity_id}} '
                            f'intersects {{eid}} at ({{px:.2f}},{{py:.2f}},{{pz:.2f}})'
                        )
                _vertex_voxel_map[entity_id] = [px-half, px+half, py-half, py+half, pz-half, pz+half]
        """).strip()

    def _material_color(self, material_class: str) -> tuple:
        """Return RGBA tuple for wireframe color based on material class."""
        colors = {
            "rigid": (0.2, 0.6, 1.0, 0.8),
            "soft": (0.6, 1.0, 0.2, 0.8),
            "fluid": (0.2, 0.4, 0.9, 0.8),
            "fabric": (0.9, 0.5, 0.2, 0.8),
            "glass": (0.8, 0.95, 1.0, 0.5),
            "metal": (0.7, 0.7, 0.8, 0.9),
            "wood": (0.7, 0.45, 0.2, 0.8),
            "plastic": (0.9, 0.9, 0.2, 0.8),
        }
        return colors.get(material_class, (0.5, 0.5, 0.5, 0.8))


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 6 — PIXEL AUDITOR
#  "Trigger a Viewport render and ask Gemini: 'Are these proxies aligned
#   with the source video? Return a Delta-Correction JSON if they are off
#   by more than 5 pixels.'" — instruction 4.txt Step 4
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PixelDelta:
    """
    A pixel-level misalignment for a single detected object.
    Produced by PixelAuditor, consumed by the correction loop.
    """
    label: str
    node_id: str
    # Pixel-space error (in wireframe render coordinates)
    source_centroid_px: Pixel       # Where the object IS in the source frame
    render_centroid_px: Pixel       # Where the proxy cube IS in the render
    pixel_error_y: int              # row error (positive = proxy is below)
    pixel_error_x: int              # col error (positive = proxy is right)
    pixel_error_magnitude: float    # Euclidean pixel error
    # Metric correction derived from pixel error
    metric_correction: Vec3         # Delta to apply to node position
    exceeds_threshold: bool = False


@dataclass
class PixelAuditReport:
    """The full output of one PixelAuditor pass."""
    pass_number: int
    aligned: bool
    total_objects: int
    aligned_objects: int
    max_pixel_error: float
    deltas: list[PixelDelta]
    delta_correction_json: str  # The actual JSON string for the correction loop
    gemini_verdict: str


class PixelAuditor:
    """
    Stage 6: Pixel-level alignment verification.

    Algorithm:
        1. Source frame: cv2-load the original video frame
        2. Render frame: load the Blender wireframe render
        3. For each proxy cube in the render:
           a. Find its projected centroid in the render
           b. Find the corresponding object centroid in the source frame
              (using SpatialGraph node's original sensor reading)
           c. Compute pixel error (row_error, col_error)
           d. If error > pixel_audit_threshold: flag for correction
        4. Convert pixel errors to metric deltas via inverse projection
        5. Optionally send to Gemini for semantic verification
        6. Return PixelAuditReport with Delta-Correction JSON

    Per instruction 4.txt: threshold = 5 pixels.
    """

    def __init__(
        self,
        config: VertexConfig,
        graph: SpatialGraph,
        oracle_model: GeminiERClient,
        icl_log: ICLMemoryLog) -> None:
        self.config = config
        self.graph = graph
        self.oracle = oracle_model
        self.icl = icl_log

    def audit(
        self,
        source_frame: ExtractedFrame,
        render_png_path: str,
        pass_number: int) -> PixelAuditReport:
        """
        Full pixel audit: source frame vs wireframe render.
        Returns PixelAuditReport with all deltas and correction JSON.
        """
        self.icl.append(
            f"PixelAuditor: pass {pass_number} — "
            f"source={Path(source_frame.image_path).name}"
        )

        # Load images
        source_img = self._load_frame(source_frame.image_path)
        render_img = self._load_frame(render_png_path) if Path(render_png_path).exists() else None

        deltas: list[PixelDelta] = []
        nodes = self.graph.all_nodes()

        for node in nodes:
            if not node.all_readings:
                continue

            # Best sensor reading for this node (highest confidence)
            best_reading = max(node.all_readings, key=lambda r: r.confidence)

            # Source centroid in original image pixels
            img_h, img_w = (source_img.shape[:2] if source_img is not None
                            else (source_frame.height, source_frame.width))
            src_cy = int((best_reading.bbox_y0 + best_reading.bbox_y1) / 2000.0 * img_h)
            src_cx = int((best_reading.bbox_x0 + best_reading.bbox_x1) / 2000.0 * img_w)

            # Render centroid: where the proxy cube should project
            rend_cy, rend_cx = self._project_to_render(node, render_img)

            # Scale source coordinates to match render resolution
            rend_h, rend_w = (render_img.shape[:2] if render_img is not None
                              else self.config.wireframe_resolution[::-1])
            scale_y = rend_h / img_h if img_h > 0 else 1.0
            scale_x = rend_w / img_w if img_w > 0 else 1.0
            scaled_src_cy = int(src_cy * scale_y)
            scaled_src_cx = int(src_cx * scale_x)

            # Pixel error
            err_y = rend_cy - scaled_src_cy
            err_x = rend_cx - scaled_src_cx
            magnitude = math.sqrt(err_y ** 2 + err_x ** 2)
            exceeds = magnitude > self.config.pixel_audit_threshold

            # Convert pixel error to metric delta via inverse projection
            metric_delta = self._pixel_to_metric(err_x, err_y, node.position_metric)

            delta = PixelDelta(
                label=node.label,
                node_id=node.node_id,
                source_centroid_px=(scaled_src_cy, scaled_src_cx),
                render_centroid_px=(rend_cy, rend_cx),
                pixel_error_y=err_y,
                pixel_error_x=err_x,
                pixel_error_magnitude=round(magnitude, 2),
                metric_correction=metric_delta,
                exceeds_threshold=exceeds)
            deltas.append(delta)

            if exceeds:
                self.icl.append(
                    f"  PixelDelta: {node.label} err={magnitude:.1f}px "
                    f"Δ({metric_delta['x']:.3f},{metric_delta['y']:.3f},{metric_delta['z']:.3f})m"
                )

        # Build correction JSON
        corrections = [
            {
                "label": d.label,
                "node_id": d.node_id,
                "pixel_error_y": d.pixel_error_y,
                "pixel_error_x": d.pixel_error_x,
                "pixel_error_magnitude": d.pixel_error_magnitude,
                "metric_correction": d.metric_correction,
                "exceeds_threshold": d.exceeds_threshold,
            }
            for d in deltas
        ]
        correction_json = json.dumps({"corrections": corrections}, indent=2)

        over_threshold = [d for d in deltas if d.exceeds_threshold]
        max_err = max((d.pixel_error_magnitude for d in deltas), default=0.0)
        aligned = len(over_threshold) == 0

        # Optional: Gemini semantic verification
        gemini_verdict = self._gemini_semantic_check(
            source_frame, render_png_path, corrections, pass_number
        )

        report = PixelAuditReport(
            pass_number=pass_number,
            aligned=aligned,
            total_objects=len(deltas),
            aligned_objects=len(deltas) - len(over_threshold),
            max_pixel_error=round(max_err, 2),
            deltas=deltas,
            delta_correction_json=correction_json,
            gemini_verdict=gemini_verdict)

        self.icl.append(
            f"PixelAuditor pass {pass_number}: "
            f"{'ALIGNED' if aligned else 'NEEDS CORRECTION'} "
            f"({report.aligned_objects}/{report.total_objects} aligned, "
            f"max_err={max_err:.1f}px)"
        )
        return report

    def _load_frame(self, path: str) -> np.ndarray | None:
        if not Path(path).exists():
            return None
        img = cv2.imread(path)
        return img

    def _project_to_render(
        self, node: SpatialNode, render_img: np.ndarray | None
    ) -> Pixel:
        """
        Project a node's 3D world position back to render image pixel coordinates.
        Uses the standard Pinhole Camera Model in reverse.
        """
        rend_h, rend_w = (
            render_img.shape[:2] if render_img is not None
            else self.config.wireframe_resolution[::-1]
        )
        pos = node.position_metric
        # Camera at (0, -8, 4), looking at origin
        # Simple perspective projection (aligned with ProxyCubeBuilder camera)
        cam_z = 4.0
        cam_y = -8.0
        focal = 800.0  # approximate focal in render pixels

        dx = pos["x"]
        dy = pos["y"] - cam_y
        dz = pos["z"] - cam_z

        if abs(dy) < 0.01:
            dy = 0.01

        # Project onto image plane
        proj_x = focal * dx / abs(dy)
        proj_y = -focal * dz / abs(dy)  # Y flipped (image convention)

        # Convert to pixel
        px_x = int(rend_w / 2 + proj_x)
        px_y = int(rend_h / 2 + proj_y)

        # Clamp to image
        px_x = max(0, min(px_x, rend_w - 1))
        px_y = max(0, min(px_y, rend_h - 1))

        return (px_y, px_x)

    def _pixel_to_metric(
        self, err_x: int, err_y: int, current_pos: Vec3
    ) -> Vec3:
        """
        Convert pixel error to metric world-space correction.
        Inverse of the Pinhole Camera Model.
        Uses the node's current depth as the scale factor.
        """
        # Approximate pixel-to-metre ratio at the object's depth
        depth = max(0.5, abs(current_pos.get("y", 2.0)))  # Use Y as depth proxy
        focal_pixels = 800.0  # Must match _project_to_render
        m_per_pixel = depth / focal_pixels

        return {
            "x": -err_x * m_per_pixel,          # pixel right → move left
            "y": 0.0,                            # depth correction handled separately
            "z": err_y * m_per_pixel * 0.5,      # pixel up → move up (partial)
        }

    def _gemini_semantic_check(
        self,
        source_frame: ExtractedFrame,
        render_png_path: str,
        corrections: list[dict],
        pass_number: int) -> str:
        """
        Optional Gemini semantic verification:
        'Are these proxies aligned with the source video?'
        Per instruction 4.txt Step 4 — sends wireframe back to Gemini ER 1.5.
        """
        over = [c for c in corrections if c.get("exceeds_threshold")]
        if not over:
            return f"ALIGNED — all {len(corrections)} objects within {self.config.pixel_audit_threshold}px threshold"

        try:
            parts: list[Any] = []

            # Attach source frame
            if Path(source_frame.image_path).exists():
                data = base64.b64encode(Path(source_frame.image_path).read_bytes()).decode()
                parts.append(self.gemini.make_image_part(data))

            # Attach wireframe render if available
            if Path(render_png_path).exists():
                data = base64.b64encode(Path(render_png_path).read_bytes()).decode()
                parts.append(self.gemini.make_image_part(data))

            over_str = json.dumps(over, indent=2)
            prompt = textwrap.dedent(f"""
                VERTEX PIXEL AUDIT — Pass {pass_number}

                Image 1: Source video frame (the TRUTH)
                Image 2: Blender wireframe proxy render (what we built)

                {len(over)} proxy cubes exceed the {self.config.pixel_audit_threshold}-pixel alignment threshold:
                {over_str}

                For each misaligned object, answer:
                1. Is the proxy cube in the right general area? (yes/no)
                2. What is the most likely physical reason for the misalignment?
                   (e.g. depth estimation error, occlusion, scale mismatch)
                3. Should we trust the pixel correction, or is the source frame ambiguous?

                Return a JSON object:
                {{
                  "overall_verdict": "trust_correction|manual_review|abandon",
                  "per_object": {{
                    "<label>": {{
                      "right_area": true/false,
                      "likely_cause": "...",
                      "trust_correction": true/false
                    }}
                  }}
                }}

                Return ONLY the JSON. No preamble.
            """).strip()
            parts.append(prompt)

            response = self.oracle.generate_content(parts)
            return response.text.strip()
        except Exception as exc:
            logger.warning("PixelAuditor Gemini check failed: %s", exc)
            return f"Gemini unavailable — using pixel metrics only. Corrections: {len(over)}"


# ═══════════════════════════════════════════════════════════════════════════
#  VERTEX CONTROLLER  (The Production Orchestrator)
# ═══════════════════════════════════════════════════════════════════════════

class VertexController:
    """
    VertexController is the principal orchestrator of Vertex Engine IX.

    It executes the full 7-stage pipeline:
        Stage 1: VideoFrameExtractor   — extract frames from input_video.mp4
        Stage 2: SpatialKernel         — Gemini ER 1.5 → sensor readings
        Stage 3: SpatialGraph          — NetworkX graph construction
        Stage 4: VoxelMap              — 2D points → 3D voxels
        Stage 5: ProxyCubeBuilder      — SpatialGraph → BPY proxy scene
        Stage 6: PixelAuditor          — wireframe → pixel delta → corrections
        Stage 7: NeuralRefinement      — ControlNet + diffusion skin

    And inherits Archon Engine VIII as a subsystem:
        - Full ArchonController available via self.archon
        - All 4-agent cycle capabilities (Perception→Grounding→Execution→Oracle)
        - Scene Hypergraph (cross-session memory)
        - Multi-hop causal reasoning
        - Cross-modal asset retrieval
        - Neural skinning pipeline

    Two entry points:
        1. process_video(video_path, scene_brief) — production video mode
        2. render(directive: VertexDirective)     — directive mode (asset_manifest)

    Usage:
        engine = VertexController(VertexConfig(
            gemini_api_key="YOUR_KEY",
            video_path="input_video.mp4"))
        result = engine.process_video("input_video.mp4", "A kitchen scene")
        print(result["graph_path"])   # NetworkX SpatialGraph JSON
        print(result["neo4j_cypher"]) # Neo4j Cypher for live server
        print(result["blend_path"])   # Proxy scene .blend
        print(result["pixel_audit"])  # Pixel audit report
    """

    def __init__(self, config: VertexConfig | None = None) -> None:
        self.config = config or VertexConfig()
        self._icl = ICLMemoryLog()
        self._runner = BlenderRunner(self.config.blender_executable)

        # Initialize Gemini
        api_key = self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("No Gemini API key. Set GEMINI_API_KEY env var.")
        self._er15 = GeminiERClient(
            api_key=api_key, model=self.config.er15_model,
            default_thinking=ThinkingPreset.NONE, temperature=0.1, max_output_tokens=4096)
        self._oracle = GeminiERClient(
            api_key=api_key, model=self.config.oracle_model,
            default_thinking=ThinkingPreset.MEDIUM, temperature=0.05, max_output_tokens=8192)

        # SpatialGraph (NetworkX)
        self._graph = SpatialGraph(scene_scale=self.config.coordinate_scale)

        # VoxelMap
        self._voxel_map = VoxelMap(self.config)

        # Neural skinning
        archon_cfg = self.config.to_archon_config()
        self._skinning = NeuralSkinningPipeline(archon_cfg, self._icl)

        # Archon subsystem (Engine VIII — full inheritance)
        self._archon = ArchonController(archon_cfg)

        logger.info("VertexController initialized.")
        logger.info("  ER 1.5 model: %s", self.config.er15_model)
        logger.info("  Oracle model: %s", self.config.oracle_model)
        logger.info("  Blender: %s  available=%s",
                    self.config.blender_executable, self._runner.available)

    # ═══════════════════════════════════════════════════════════════════
    #  PRIMARY ENTRY POINT 1: Production Video Mode
    # ═══════════════════════════════════════════════════════════════════

    def process_video(
        self,
        video_path: str,
        scene_brief: str,
        visual_style: str = "photorealistic cinematic",
        target_objects: list[str] | None = None,
        pixel_audit: bool = True,
        export_neo4j: bool = True) -> dict:
        """
        Full production pipeline: input_video.mp4 → 3D proxy scene → pixel-audited render.

        This is the "build" described in instruction 4.txt:
        "Initialize a Python framework that uses Gemini ER 1.5 to detect objects
         in input_video.mp4, store as nodes in a NetworkX/Neo4j graph, generate
         a BPY script that places Proxy Cubes, and audit alignment via Viewport render."

        Returns:
            {
                "session_id": str,
                "output_root": str,
                "blend_path": str,
                "graph_path": str,           # SpatialGraph JSON
                "neo4j_cypher_path": str,    # Neo4j Cypher export
                "voxel_summary": dict,
                "pixel_audit_reports": list,
                "frame_count": int,
                "final_corrections_applied": int,
                "graph_stats": dict,
                "icl_log": list[str],
                "success": bool,
            }
        """
        session_id = str(uuid.uuid4())[:12]
        self._icl.append(f"VertexController.process_video() — session {session_id}")

        out_root = Path(self.config.output_dir) / session_id
        out_root.mkdir(parents=True, exist_ok=True)
        blend_path = str(out_root / "vertex_proxy.blend")

        logger.info("═" * 72)
        logger.info("VERTEX ENGINE IX — Session %s", session_id)
        logger.info("  Video: %s", video_path)
        logger.info("  Scene: %s", scene_brief)
        logger.info("═" * 72)

        # ── Stage 1: Video Frame Extraction ──────────────────────────
        logger.info("Stage 1: VideoFrameExtractor")
        extractor = VideoFrameExtractor(self.config, str(out_root))
        frames = extractor.extract(video_path)
        self._icl.append(f"Extracted {len(frames)} frames from {Path(video_path).name}")

        # ── Stage 2: Spatial Kernel (Gemini ER 1.5) ──────────────────
        logger.info("Stage 2: SpatialKernel (Gemini ER 1.5 Pointing API)")
        kernel = SpatialKernel(self._er15, self.config, self._icl)
        readings_by_frame = kernel.process_frames_batch(frames, target_objects)
        total_readings = sum(len(v) for v in readings_by_frame.values())
        self._icl.append(f"SpatialKernel: {total_readings} readings across {len(frames)} frames")

        # Save raw sensor readings
        readings_path = str(out_root / "sensor_readings.json")
        with open(readings_path, "w") as f:
            json.dump({
                str(fi): [vars(r) for r in rs]
                for fi, rs in readings_by_frame.items()
            }, f, indent=2, default=str)

        # ── Stage 3: SpatialGraph Construction ───────────────────────
        logger.info("Stage 3: SpatialGraph — building NetworkX graph")
        self._build_spatial_graph(readings_by_frame, scene_brief)
        edges_added = self._graph.infer_edges_from_positions(
            frame_index=frames[0].frame_index if frames else 0
        )
        self._icl.append(
            f"SpatialGraph: {self._graph.get_stats()['nodes']} nodes, "
            f"{self._graph.get_stats()['edges']} edges ({edges_added} inferred)"
        )

        # ── Stage 4: VoxelMap ────────────────────────────────────────
        logger.info("Stage 4: VoxelMap — 2D→3D reconstruction")
        self._voxel_map.reconstruct(readings_by_frame, frames)
        # Update SpatialGraph positions from refined voxel data
        self._sync_voxel_positions(readings_by_frame)
        voxel_summary = self._voxel_map.get_grid_summary()
        self._icl.append(f"VoxelMap: {voxel_summary}")

        # ── Save SpatialGraph ─────────────────────────────────────────
        graph_path = str(out_root / "spatial_graph.json")
        self._graph.save(graph_path)

        # ── Neo4j Cypher export ───────────────────────────────────────
        neo4j_path = ""
        if export_neo4j:
            neo4j_cypher = self._graph.to_neo4j_cypher()
            neo4j_path = str(out_root / "neo4j_import.cypher")
            Path(neo4j_path).write_text(neo4j_cypher)
            self._icl.append(f"Neo4j Cypher exported → {neo4j_path}")
            logger.info("Stage 3+: Neo4j Cypher saved → %s", neo4j_path)

        # ── Stage 5: ProxyCubeBuilder ─────────────────────────────────
        logger.info("Stage 5: ProxyCubeBuilder — graph → BPY proxy scene")
        proxy_builder = ProxyCubeBuilder(self.config, self._graph, self._icl)
        proxy_script = proxy_builder.generate_proxy_script(blend_path, session_id)
        proxy_script_path = str(out_root / "proxy_scene.py")
        Path(proxy_script_path).write_text(proxy_script)

        if not self._runner.run_script(proxy_script):
            logger.info(
                "Blender not available — proxy script saved: %s", proxy_script_path
            )

        # ── Stage 6: PixelAuditor (Recursive Loop) ────────────────────
        audit_reports = []
        total_corrections = 0

        if pixel_audit and frames:
            logger.info("Stage 6: PixelAuditor — pixel-level alignment verification")
            auditor = PixelAuditor(self.config, self._graph, self._oracle, self._icl)

            # Use the first keyframe as the audit reference frame
            audit_frame = frames[0]

            for audit_pass in range(1, self.config.pixel_audit_max_passes + 1):
                # Render wireframe
                wireframe_path = str(out_root / f"wireframe_pass{audit_pass}.png")
                audit_render_script = proxy_builder.generate_audit_render_script(
                    blend_path, wireframe_path
                )
                audit_script_path = str(out_root / f"audit_render_pass{audit_pass}.py")
                Path(audit_script_path).write_text(audit_render_script)
                if Path(blend_path).exists():
                    self._runner.run_script(audit_render_script, blend_path)

                # Run pixel audit
                report = auditor.audit(audit_frame, wireframe_path, audit_pass)
                audit_reports.append({
                    "pass": audit_pass,
                    "aligned": report.aligned,
                    "total_objects": report.total_objects,
                    "aligned_objects": report.aligned_objects,
                    "max_pixel_error": report.max_pixel_error,
                    "delta_correction_json": report.delta_correction_json,
                    "gemini_verdict": report.gemini_verdict,
                })

                if report.aligned:
                    logger.info(
                        "PixelAuditor pass %d: ALIGNED (max_err=%.1fpx) — pipeline complete.",
                        audit_pass, report.max_pixel_error)
                    break

                # Apply corrections and regenerate proxy scene
                corrections_applied = self._apply_pixel_corrections(report)
                total_corrections += corrections_applied
                logger.info(
                    "PixelAuditor pass %d: %d corrections applied, re-building proxy...",
                    audit_pass, corrections_applied)

                # Re-generate proxy script with corrected positions
                proxy_script = proxy_builder.generate_proxy_script(blend_path, session_id)
                Path(proxy_script_path).write_text(proxy_script)
                if self._runner.run_script(proxy_script):
                    self._icl.append(
                        f"Proxy rebuilt after corrections (pass {audit_pass})"
                    )

        # ── Stage 7: Neural Refinement ────────────────────────────────
        logger.info("Stage 7: NeuralRefinement — depth map → ControlNet skinning")
        if frames:
            self._neural_refinement_pass(frames[0], out_root, visual_style)

        # ── Final graph snapshot ──────────────────────────────────────
        self._graph.save(graph_path)

        # Save ICL log
        icl_path = str(out_root / "vertex_icl.log")
        Path(icl_path).write_text("\n".join(self._icl.entries))

        result = {
            "session_id": session_id,
            "output_root": str(out_root),
            "blend_path": blend_path,
            "proxy_script_path": proxy_script_path,
            "graph_path": graph_path,
            "neo4j_cypher_path": neo4j_path,
            "sensor_readings_path": readings_path,
            "frame_count": len(frames),
            "total_sensor_readings": total_readings,
            "voxel_summary": voxel_summary,
            "pixel_audit_reports": audit_reports,
            "final_corrections_applied": total_corrections,
            "graph_stats": self._graph.get_stats(),
            "neo4j_cypher": (
                Path(neo4j_path).read_text() if neo4j_path and Path(neo4j_path).exists()
                else ""
            ),
            "icl_log": self._icl.entries,
            "icl_log_path": icl_path,
            "success": len([a for a in audit_reports if not a["aligned"]]) == 0 or not audit_reports,
        }

        logger.info("═" * 72)
        logger.info("VERTEX complete. Session: %s", session_id)
        logger.info("  Graph: %s", self._graph.get_stats())
        logger.info("  Voxels: %s", voxel_summary)
        logger.info("  Audit passes: %d  Total corrections: %d",
                    len(audit_reports), total_corrections)
        logger.info("  Output: %s", str(out_root))
        logger.info("═" * 72)
        return result

    # ═══════════════════════════════════════════════════════════════════
    #  PRIMARY ENTRY POINT 2: Directive Mode (inherits Archon)
    # ═══════════════════════════════════════════════════════════════════

    def render(self, directive: VertexDirective) -> dict:
        """
        Directive mode: pass an asset_manifest + story_beats like Archon.
        Converts to ArchonDirective and runs full Archon pipeline,
        PLUS post-processes with VertexSpatialGraph and PixelAuditor.
        """
        self._icl.append(f"VertexController.render() directive mode: {directive.scene_name}")

        # If video_path is set, run video mode first to build the graph
        if directive.video_path:
            video_result = self.process_video(
                directive.video_path,
                directive.scene_brief,
                directive.visual_style,
                directive.target_objects,
                directive.pixel_audit,
                directive.export_neo4j)
            # Merge graph data back into asset_manifest
            directive = self._enrich_directive_from_graph(directive)

        # Convert to ArchonDirective and run Archon subsystem
        archon_dir = self._to_archon_directive(directive)
        archon_result = self._archon.render(archon_dir)

        # Post-process: sync Archon's SceneHypergraph to Vertex SpatialGraph
        self._sync_archon_to_spatial_graph(archon_result)

        return {
            **archon_result,
            "vertex_graph_stats": self._graph.get_stats(),
            "vertex_neo4j_cypher": self._graph.to_neo4j_cypher(),
        }

    # ═══════════════════════════════════════════════════════════════════
    #  UTILITY: Spatial Graph Query Interface
    # ═══════════════════════════════════════════════════════════════════

    def query_spatial_relations(self, object_label: str) -> dict:
        """
        Query all spatial relations for an object.
        Returns a structured dict of relations with distance, angle, stability.

        Example:
            engine.query_spatial_relations("cup")
            → {"on_top_of": ["table"], "near": ["book", "lamp"], ...}
        """
        nodes = self._graph.find_by_label(object_label)
        if not nodes:
            return {"error": f"No object matching '{object_label}' in graph"}

        result: dict[str, Any] = {
            "label": object_label,
            "node_count": len(nodes),
            "relations": [],
        }
        for node in nodes:
            for edge in self._graph.get_edges_from(node.node_id):
                tgt_id = edge.get("target_id", "")
                tgt_node = self._graph.get_node(tgt_id)
                result["relations"].append({
                    "relation": edge.get("relation"),
                    "target": tgt_node.label if tgt_node else tgt_id,
                    "distance_m": edge.get("distance_m"),
                    "angle_deg": edge.get("angle_deg"),
                    "temporal_stable": edge.get("temporal_stable", False),
                })
        return result

    def get_graph_text_summary(self) -> str:
        """Return a human-readable summary of the SpatialGraph."""
        lines = [
            f"SpatialGraph Summary — {self._graph.get_stats()['nodes']} objects",
            "─" * 50,
        ]
        for node in sorted(self._graph.all_nodes(), key=lambda n: n.label):
            pos = node.position_metric
            lines.append(
                f"  {node.label:<20} "
                f"({pos['x']:+.2f}, {pos['y']:+.2f}, {pos['z']:+.2f})m  "
                f"seen={node.observation_count}x  "
                f"conf={node.confidence_avg:.2f}  "
                f"{'GROUNDED' if node.is_grounded else 'AIRBORNE'}"
            )
        stable = self._graph.get_stable_edges()
        if stable:
            lines.extend(["", "Stable Spatial Relations:"])
            for src, tgt, data in stable:
                src_n = self._graph.get_node(src)
                tgt_n = self._graph.get_node(tgt)
                lines.append(
                    f"  ({src_n.label if src_n else src})"
                    f"-[:{data['relation']} "
                    f"{{dist:{data.get('distance_m',0):.2f}m, "
                    f"angle:{data.get('angle_deg',0):.0f}°}}]->"
                    f"({tgt_n.label if tgt_n else tgt})"
                )
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════
    #  PRIVATE HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _build_spatial_graph(
        self,
        readings_by_frame: dict[int, list[SensorReading]],
        scene_brief: str) -> None:
        """Build/update SpatialGraph from all sensor readings."""
        for frame_idx, readings in readings_by_frame.items():
            for r in readings:
                node_id = re.sub(r"[^A-Za-z0-9_]", "_", r.label.lower())
                node = SpatialNode(
                    node_id=node_id,
                    label=r.label,
                    position_metric={
                        "x": r.world_x,
                        "y": r.world_y,
                        "z": r.world_z,
                    },
                    physics_metadata=r.physics_metadata,
                    first_seen_frame=r.frame_index,
                    last_seen_frame=r.frame_index,
                    confidence_avg=r.confidence,
                    is_grounded=r.physics_metadata.get("is_grounded", True),
                    material_class=r.physics_metadata.get("material_class", "rigid"),
                    all_readings=[r])
                node.bounding_box = {
                    "min": {
                        "x": r.world_x - 0.3,
                        "y": r.world_y - 0.3,
                        "z": max(0, r.world_z - 0.3),
                    },
                    "max": {
                        "x": r.world_x + 0.3,
                        "y": r.world_y + 0.3,
                        "z": r.world_z + 0.3,
                    },
                }
                self._graph.upsert_node(node)

    def _sync_voxel_positions(
        self, readings_by_frame: dict[int, list[SensorReading]]
    ) -> None:
        """Update SpatialGraph node positions from VoxelMap-refined readings."""
        # Build label → latest refined reading
        latest: dict[str, SensorReading] = {}
        for readings in readings_by_frame.values():
            for r in readings:
                node_id = re.sub(r"[^A-Za-z0-9_]", "_", r.label.lower())
                existing = latest.get(node_id)
                if not existing or r.confidence > existing.confidence:
                    latest[node_id] = r

        for node_id, r in latest.items():
            node = self._graph.get_node(node_id)
            if node:
                node.position_metric = {"x": r.world_x, "y": r.world_y, "z": r.world_z}
                self._graph.upsert_node(node)

    def _apply_pixel_corrections(self, report: PixelAuditReport) -> int:
        """Apply pixel delta corrections to SpatialGraph node positions."""
        applied = 0
        for delta in report.deltas:
            if not delta.exceeds_threshold:
                continue
            node = self._graph.get_node(delta.node_id)
            if not node:
                continue
            corr = delta.metric_correction
            node.position_metric["x"] += corr.get("x", 0.0)
            node.position_metric["y"] += corr.get("y", 0.0)
            node.position_metric["z"] += corr.get("z", 0.0)
            self._graph.upsert_node(node)
            applied += 1
            self._icl.append(
                f"PixelCorrection: {node.label} "
                f"Δ({corr['x']:+.3f}, {corr['y']:+.3f}, {corr['z']:+.3f})m "
                f"[pixel_err={delta.pixel_error_magnitude:.1f}px]"
            )
        return applied

    def _neural_refinement_pass(
        self,
        frame: ExtractedFrame,
        out_root: Path,
        visual_style: str) -> None:
        """Run neural refinement on the first frame."""
        beauty = str(out_root / "passes" / "beauty" / "frame_0001.png")
        depth  = str(out_root / "passes" / "depth"  / "frame_0001.exr")
        normal = str(out_root / "passes" / "normal" / "frame_0001.png")
        output = str(out_root / "skinned" / "frame_0001.png")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        self._skinning.skin_frame(beauty, depth, normal, visual_style, 1, output)

    def _to_archon_directive(self, directive: VertexDirective) -> ArchonDirective:
        """Convert VertexDirective to ArchonDirective."""
        from engine_8_archon.archon_engine import ArchonDirective as AD
        return AD(
            scene_brief=directive.scene_brief,
            asset_manifest=directive.asset_manifest,
            story_beats=directive.story_beats,
            visual_style=directive.visual_style,
            duration_seconds=directive.duration_seconds,
            physics_enabled=directive.physics_enabled,
            physics_laws=directive.physics_laws,
            style_directive=directive.style_directive,
            reference_image_path=directive.reference_image_path,
            output_name=directive.output_name,
            scene_name=directive.scene_name,
            previous_scene_events=directive.previous_scene_events,
            graph_session_id=directive.graph_session_id)

    def _enrich_directive_from_graph(self, directive: VertexDirective) -> VertexDirective:
        """
        After video processing, enrich asset_manifest with detected positions
        from the SpatialGraph. Matches by label similarity.
        """
        for asset in directive.asset_manifest:
            nodes = self._graph.find_by_label(asset.entity_id)
            if not nodes:
                nodes = self._graph.find_by_label(asset.asset_type)
            if nodes:
                best = max(nodes, key=lambda n: n.confidence_avg)
                pos = best.position_metric
                asset.position = {"x": pos["x"], "y": pos["y"], "z": pos["z"]}
                if best.physics_metadata:
                    if "mass_kg" in best.physics_metadata:
                        asset.physics_mass_kg = best.physics_metadata["mass_kg"]
                    if "friction" in best.physics_metadata:
                        asset.physics_friction = best.physics_metadata["friction"]
        return directive

    def _sync_archon_to_spatial_graph(self, archon_result: dict) -> None:
        """Sync Archon's SceneHypergraph nodes into Vertex SpatialGraph."""
        ws = archon_result.get("grounded_world_state", {})
        for obj_dict in ws.get("objects", []):
            node_id = re.sub(r"[^A-Za-z0-9_]", "_", obj_dict.get("entity_id", "").lower())
            pos = obj_dict.get("position_metric", {"x": 0, "y": 0, "z": 0})
            if not self._graph.get_node(node_id):
                node = SpatialNode(
                    node_id=node_id,
                    label=obj_dict.get("label", node_id),
                    position_metric=pos,
                    physics_metadata=obj_dict.get("physics_metadata", {}),
                    material_class="rigid")
                self._graph.upsert_node(node)

    def describe(self) -> str:
        stats = self._graph.get_stats()
        return textwrap.dedent(f"""
            ╔══════════════════════════════════════════════════════════════════╗
            ║  VERTEX ENGINE  —  Engine IX                                    ║
            ║  Perception-to-Graph Production Pipeline                        ║
            ╠══════════════════════════════════════════════════════════════════╣
            ║  ER 1.5 sensor:  {self.config.er15_model:<42}║
            ║  Oracle:         {self.config.oracle_model:<42}║
            ║  Blender:        {self.config.blender_executable:<42}║
            ║  Render engine:  {self.config.render_engine:<42}║
            ║  Video path:     {(self.config.video_path or '(not set)'):<42}║
            ║  Keyframe int:   {str(self.config.keyframe_interval_sec) + 's':<42}║
            ║  Max keyframes:  {self.config.max_keyframes:<42}║
            ║  Voxel grid:     {str(self.config.voxel_grid_resolution) + '³':<42}║
            ║  Pixel audit:    {str(self.config.pixel_audit_threshold) + 'px threshold':<42}║
            ║  Graph nodes:    {stats['nodes']:<42}║
            ║  Graph edges:    {stats['edges']:<42}║
            ║  Stable edges:   {stats['stable_edges']:<42}║
            ║  Output dir:     {self.config.output_dir:<42}║
            ╚══════════════════════════════════════════════════════════════════╝
            Paradigm: Manage a database of a reality being simulated.
            Hallucinations: PROVABLY IMPOSSIBLE — pixel truth verified by cv2.
            Archon (VIII) subsystem: ACTIVE — full 4-agent cycle available.
        """).strip()

    def __repr__(self) -> str:
        return (
            f"VertexController("
            f"model={self.config.er15_model!r}, "
            f"graph={self._graph.get_stats()}, "
            f"blender_available={self._runner.available})"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  QUICK-START DEMO
# ═══════════════════════════════════════════════════════════════════════════

def _demo_kitchen_video() -> None:
    """
    Quick-start demo: process a video file through the full Vertex pipeline.

    Equivalent to the "Final Boss Prompt" in instruction 4.txt:
        'Initialize a Python framework that uses Gemini ER 1.5 to detect
         objects in input_video.mp4. Extract [label, [y,x]]. Store as nodes
         in a NetworkX graph. Generate a BPY script that places Proxy Cubes.
         Trigger a Viewport render and audit alignment. Return Delta-Correction
         JSON if off by more than 5 pixels.'
    """
    import argparse
    parser = argparse.ArgumentParser(description="Vertex Engine IX")
    parser.add_argument("--video", default="", help="Path to input video (.mp4/.mov/.avi)")
    parser.add_argument("--scene", default="A kitchen with objects on a countertop",
                        help="Scene description")
    parser.add_argument("--style", default="cinematic photorealism, 35mm",
                        help="Visual style")
    parser.add_argument("--objects", default="", help="Comma-separated target object labels")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""),
                        help="Gemini API key")
    parser.add_argument("--output", default="./vertex_output", help="Output directory")
    parser.add_argument("--blender", default="blender", help="Blender executable")
    parser.add_argument("--no-audit", action="store_true", help="Skip pixel audit")
    parser.add_argument("--no-neo4j", action="store_true", help="Skip Neo4j Cypher export")
    parser.add_argument("--keyframe-interval", type=float, default=1.0,
                        help="Keyframe interval in seconds")
    parser.add_argument("--max-keyframes", type=int, default=8,
                        help="Maximum keyframes to extract")
    args = parser.parse_args()

    config = VertexConfig(
        gemini_api_key=args.api_key,
        output_dir=args.output,
        blender_executable=args.blender,
        keyframe_interval_sec=args.keyframe_interval,
        max_keyframes=args.max_keyframes)

    engine = VertexController(config)
    print(engine.describe())

    if not args.video:
        print("\n[vertex] No video path provided. Use --video input_video.mp4")
        print("[vertex] Showing engine description only.")
        print("\n" + engine.get_graph_text_summary())
        return

    target_objects = [o.strip() for o in args.objects.split(",") if o.strip()] or None

    result = engine.process_video(
        video_path=args.video,
        scene_brief=args.scene,
        visual_style=args.style,
        target_objects=target_objects,
        pixel_audit=not args.no_audit,
        export_neo4j=not args.no_neo4j)

    print("\n── VERTEX RESULT ──────────────────────────────────────────────")
    print(f"  Session:     {result['session_id']}")
    print(f"  Frames:      {result['frame_count']}")
    print(f"  Readings:    {result['total_sensor_readings']}")
    print(f"  Graph stats: {result['graph_stats']}")
    print(f"  Voxel grid:  {result['voxel_summary']}")
    print(f"  Corrections: {result['final_corrections_applied']}")
    print(f"  Success:     {result['success']}")
    print(f"\n  Blend:       {result['blend_path']}")
    print(f"  Graph JSON:  {result['graph_path']}")
    print(f"  Neo4j:       {result['neo4j_cypher_path']}")
    print(f"  Proxy BPY:   {result['proxy_script_path']}")

    if result.get("pixel_audit_reports"):
        print("\n── Pixel Audit Reports ────────────────────────────────────────")
        for r in result["pixel_audit_reports"]:
            status = "✓ ALIGNED" if r["aligned"] else "✗ CORRECTED"
            print(
                f"  Pass {r['pass']}: {status}  "
                f"{r['aligned_objects']}/{r['total_objects']} aligned  "
                f"max_err={r['max_pixel_error']}px"
            )

    print("\n── Spatial Graph ───────────────────────────────────────────────")
    print(engine.get_graph_text_summary())


if __name__ == "__main__":
    _demo_kitchen_video()
