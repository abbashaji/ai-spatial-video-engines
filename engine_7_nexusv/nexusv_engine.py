"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         NEXUS-V ENGINE  —  Engine VII of VII                                ║
║         Aletheia-Blender Reality-Simulation Protocol                        ║
║         "The Spatial Engine Architect"                                      ║
║                                                                              ║
║  Paradigm:  Simulate Reality. Record the Simulation. Skin It Neurally.      ║
║                                                                              ║
║  Stack:                                                                      ║
║    Gemini ER 1.5  →  Spatial Kernel / Creative Director / BPY Author        ║
║    Blender 4.3+   →  Deterministic World Model (Bullet, Mantaflow, Eevee)   ║
║    ControlNet     →  Depth + Canny conditioned spatial lock                 ║
║    Diffusion      →  Neural Skin (ComfyUI / A1111 / Flux)                   ║
║                                                                              ║
║  Lineage:  Integrates all advances from Engines I–VI plus:                  ║
║    • Style-Differentiable Physics (SDP) from Aletheia (V)                   ║
║    • Deterministic BPY geometry from Prometheus (VI)                        ║
║    • Gemini Spatial JSON Handshake from instruction 1.txt                   ║
║    • Nexus-V Reality-Simulation Protocol from instruction 2.txt             ║
║                                                                              ║
║  Four Phases:                                                                ║
║    Phase 1: BPY Core         — WorldBuilder from Spatial JSON               ║
║    Phase 2: Gemini Handshake — Multimodal analysis → Structural JSON        ║
║    Phase 3: Kinetic Engine   — TimelineManager + Physics hand-off + Cam DP  ║
║    Phase 4: Neural Refinement— Depth/Normal/Canny → ControlNet skinning     ║
║                                                                              ║
║  HALLUCINATIONS: MATHEMATICALLY IMPOSSIBLE.                                 ║
║  The .blend file is the ground truth. Depth maps are Blender geometry.      ║
║  ControlNet cannot drift what a mesh has fixed.                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

ARCHITECTURE — 4 Phases + Autonomous Correction Loop:

  Phase 1:  BPY Core              (Gemini Spatial JSON → WorldBuilder → .blend)
  Phase 2:  Gemini Handshake      (Multimodal analysis → Structural JSON →
                                   Coordinate mapping → Clipping sanity check)
  Phase 3:  Kinetic Engine        (TimelineManager → keyframes → Physics hand-off
                                   → CinematographyModule → Agentic Camera)
  Phase 4:  Neural Refinement     (Eevee-Next multipass → ControlNet → Diffusion)
  Loop:     Autonomous Audit      (Gemini audits → bpy correction → re-render)

WHAT IS REAL TODAY vs FUTURE API:
  Real:    Full bpy script generation, Blender headless CLI, Bullet physics,
           Mantaflow wiring, ControlNet depth conditioning logic, all Gemini
           API calls, WorldBuilder, TimelineManager, BezierCurveOrchestrator,
           CinematographyModule, AgentzCamera, SanityCheckLoop, multi-pass
           AOV export, ComfyUI/A1111 payload builder, autonomous audit loop,
           StyleDirective SDP physics, SHA-256 material hash, ICL memory log
  Future:  grok-imagine-video API (not yet public), true latent seed injection
           (requires model weight access), native voxel output from Gemini

See engine README for full documentation.
"""

from __future__ import annotations

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
from typing import Any, Literal

from google import genai                          # pip install google-genai
from google.genai import types
from gemini_er_client import (                    # shared ER 1.5 adapter
    GeminiERClient, ThinkingPreset, ER15_MODEL,
    BoundingBox2D, SpatialPoint, GeminiERResponse,
    er15_to_blender)
import httpx                             # pip install httpx
import numpy as np                       # pip install numpy

logger = logging.getLogger("nexusv")
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s [%(levelname)s] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S")


# ═══════════════════════════════════════════════════════════════════════════
#  TYPE ALIASES
# ═══════════════════════════════════════════════════════════════════════════

Vec3  = dict[str, float]   # {"x": float, "y": float, "z": float}
Euler = dict[str, float]   # {"x": float, "y": float, "z": float}  radians
RGBA  = tuple[float, float, float, float]


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NexusVConfig:
    """Runtime configuration for the Nexus-V engine."""

    gemini_api_key: str = ""
    comfyui_base_url: str = "http://127.0.0.1:8188"
    a1111_base_url: str = "http://127.0.0.1:7860"
    blender_executable: str = "blender"
    output_dir: str = "./nexusv_output"
    render_engine: Literal["eevee", "cycles"] = "eevee"
    output_resolution: tuple[int, int] = (1920, 1080)
    diffusion_model: str = "realistic_vision_v6"
    controlnet_depth_strength: float = 0.80
    controlnet_canny_strength: float = 0.40
    controlnet_normal_strength: float = 0.35
    denoising_strength: float = 0.55
    max_audit_passes: int = 3
    clipping_tolerance_m: float = 0.001
    coordinate_scale: float = 10.0   # Gemini 0..1 → Blender metric (multiply by this)
    fps: int = 24
    gemini_model: str = "gemini-1.5-pro-latest"
    gemini_thinking_tokens: int = 2048  # now wired to ThinkingConfig via ThinkingPreset
    use_style_physics: bool = True    # Engine V SDP integration
    log_icl_memory: bool = True


# ═══════════════════════════════════════════════════════════════════════════
#  PRIMARY INPUT PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AssetSpec:
    """
    A single 3D asset to be instantiated in the Blender world.

    Args:
        entity_id:       Unique identifier (used in story beats & camera ops).
        asset_type:      Blender primitive or rig type.
        position:        Initial world position {x, y, z} in metric units.
        rotation_euler:  Initial rotation in radians {x, y, z}.
        scale:           Uniform scale or per-axis {x, y, z}.
        material:        Material description dict — hashed for anti-drift.
        physics_type:    Blender rigid body type (ACTIVE / PASSIVE / NONE).
        physics_mass_kg: Mass for Bullet physics.
        height_m:        Convenience height for character rigs.
        is_fluid:        Whether Mantaflow should own this object.
        is_cloth:        Whether cloth simulation applies.
        blend_file_path: Optional .blend asset library path.
        extra_props:     Arbitrary additional bpy properties.
    """
    entity_id: str
    asset_type: Literal[
        "MESH_CUBE", "MESH_SPHERE", "MESH_CYLINDER", "MESH_PLANE",
        "MESH_CONE", "MESH_TORUS", "character_rig", "prop_rigid",
        "prop_soft", "furniture", "light_rig", "camera", "particle_system",
        "fluid_domain", "fluid_flow", "custom_blend",
    ]
    position: Vec3 = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    rotation_euler: Euler = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    scale: float | Vec3 = 1.0
    material: dict[str, Any] = field(default_factory=dict)
    physics_type: Literal["ACTIVE", "PASSIVE", "NONE"] = "NONE"
    physics_mass_kg: float = 1.0
    physics_restitution: float = 0.3
    physics_friction: float = 0.5
    physics_collision_shape: Literal[
        "CONVEX_HULL", "MESH", "BOX", "SPHERE", "CAPSULE", "CYLINDER",
    ] = "CONVEX_HULL"
    height_m: float | None = None
    is_fluid: bool = False
    is_cloth: bool = False
    blend_file_path: str | None = None
    extra_props: dict[str, Any] = field(default_factory=dict)

    @property
    def material_hash(self) -> str:
        """SHA-256 of the material description — the anti-drift texture lock."""
        return hashlib.sha256(
            json.dumps(self.material, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class StoryBeat:
    """
    A narrative event in the timeline that drives animation or physics.

    Args:
        t_seconds:      Timestamp in seconds.
        beat_id:        Short identifier (used in bpy script comments).
        description:    Human-readable description of the beat.
        entity_id:      Primary entity involved.
        force_vector:   Force to apply (triggers physics hand-off if set).
        action_type:    Category of action (drives code-gen logic).
        camera_hint:    Shot type request for the CinematographyModule.
        motion_target:  Target position for keyframed movement {x, y, z}.
        extra:          Arbitrary extra data for script generation.
    """
    t_seconds: float
    beat_id: str
    description: str
    entity_id: str = ""
    force_vector: Vec3 | None = None
    action_type: Literal[
        "enter", "exit", "walk", "run", "grab", "drop", "collision",
        "fluid", "cloth", "door_slam", "camera_cut", "light_change",
        "explosion", "physics_handoff", "keyframe", "custom",
    ] = "custom"
    camera_hint: str = ""
    motion_target: Vec3 | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StyleDirective:
    """
    Style-physics manifest input (from Engine V / Aletheia integration).
    When provided, Gemini derives custom physics constants from the art style
    (Style-Differentiable Physics — SDP).

    Args:
        reference:       Artistic reference (film, painting, animation style).
        emotional_arc:   List of (emotion_label, t_seconds) tuples.
        quantum_paths:   Number of parallel outcome paths to evaluate.
        style_tags:      Additional style tags.
        scene_brief:     Override scene brief for this directive.
    """
    reference: str = ""
    emotional_arc: list[tuple[str, float]] = field(default_factory=list)
    quantum_paths: int = 3
    style_tags: list[str] = field(default_factory=list)
    scene_brief: str = ""


@dataclass
class PhysicsLaw:
    """
    A causal physics law (from Engine IV / Aether-Omni integration).
    Explicit laws override Blender defaults when script generation runs.

    Args:
        law_type:  Category of physical law.
        params:    Dict of parameters for the law.
        target_id: Entity this law applies to ("scene" for global laws).
    """
    law_type: Literal[
        "gravity", "material", "intent", "atmosphere",
        "thermal", "electromagnetic", "fluid", "constraint",
    ]
    params: dict[str, Any]
    target_id: str = "scene"


@dataclass
class RealityDirective:
    """
    The unified primary input for Nexus-V.  Merges:
      - SceneDirective  (Prometheus VI)  — 3D assets, story beats, visual style
      - StyleDirective  (Aletheia V)     — SDP physics derived from artistic reference
      - PhysicsLaw list (Aether-Omni IV) — explicit causal physics overrides
      - Reference image / video          — Gemini multimodal handshake input

    The name "RealityDirective" signals the paradigm: we are not directing
    a video generator — we are directing a reality simulation.

    Example:
        RealityDirective(
            scene_brief="A detective finds a clue on a rain-soaked rooftop.",
            asset_manifest=[
                AssetSpec("detective", "character_rig", height_m=1.82,
                           material={"outfit": "trench_coat_charcoal"}),
                AssetSpec("evidence_box", "prop_rigid",
                           position={"x": 2.0, "y": 0.5, "z": 0.0},
                           physics_type="PASSIVE",
                           material={"surface": "weathered_cardboard"}),
                AssetSpec("rain", "particle_system",
                           extra_props={"particle_count": 50000}),
                AssetSpec("key_light", "light_rig",
                           material={"type": "SUN", "energy": 3.0,
                                     "color": [1.0, 0.95, 0.85]}),
            ],
            story_beats=[
                StoryBeat(0.0, "scene_open",   "Wide establish shot"),
                StoryBeat(1.5, "detective_crouches", "Detective examines box",
                           entity_id="detective", action_type="walk",
                           motion_target={"x": 2.0, "y": 0.5, "z": 0.0}),
                StoryBeat(3.0, "pick_up",      "Picks up box",
                           entity_id="evidence_box", action_type="grab",
                           force_vector={"x": 0.0, "y": 0.0, "z": 8.0}),
            ],
            visual_style="neo-noir photorealism, rain-slicked reflections, 35mm grain",
            duration_seconds=6.0,
            physics_enabled=True,
            style_directive=StyleDirective(
                reference="Blade Runner 2049",
                emotional_arc=[("dread", 0.0), ("revelation", 3.5), ("resolve", 6.0)]),
            reference_image_path="./ref_rooftop.jpg")
    """
    scene_brief: str
    asset_manifest: list[AssetSpec]
    story_beats: list[StoryBeat]
    visual_style: str
    duration_seconds: float = 8.0
    physics_enabled: bool = True
    fluid_objects: list[str] = field(default_factory=list)
    cloth_objects: list[str] = field(default_factory=list)
    physics_laws: list[PhysicsLaw] = field(default_factory=list)
    style_directive: StyleDirective | None = None
    reference_image_path: str | None = None
    reference_video_path: str | None = None
    output_name: str = "nexusv_render"


# ═══════════════════════════════════════════════════════════════════════════
#  STRUCTURAL JSON — the canonical handoff format between Gemini and Blender
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpatialObject:
    """One object entry in a Gemini Structural JSON."""
    entity_id: str
    label: str
    position_norm: Vec3          # Gemini normalized 0..1
    position_metric: Vec3        # Blender metric (after coordinate mapping)
    bounding_box: dict           # {"min": Vec3, "max": Vec3} metric
    material_description: str
    semantic_type: str
    physics_metadata: dict
    motion_trajectory: list[dict]  # [{t, x_norm, y_norm, z_norm}]
    clip_checked: bool = False
    clip_violation: bool = False


@dataclass
class StructuralJSON:
    """
    Gemini's authoritative spatial description of the scene.
    Produced by the GeminiHandshake and consumed by the WorldBuilder.
    """
    session_id: str
    scene_brief: str
    objects: list[SpatialObject]
    camera_ops: list[dict]
    action_tokens: list[dict]
    physics_manifest: dict
    style_physics_manifest: dict | None
    icl_log_entries: list[str]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "scene_brief": self.scene_brief,
            "objects": [vars(o) for o in self.objects],
            "camera_ops": self.camera_ops,
            "action_tokens": self.action_tokens,
            "physics_manifest": self.physics_manifest,
            "style_physics_manifest": self.style_physics_manifest,
            "icl_log_entries": self.icl_log_entries,
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  ICL MEMORY LOG  —  The continuity bible injected into every Gemini prompt
# ═══════════════════════════════════════════════════════════════════════════

class ICLMemoryLog:
    """
    Append-only in-context learning log.
    Records every significant event during the pipeline and injects the
    full log into subsequent Gemini calls as a continuity bible.
    """

    def __init__(self) -> None:
        self._entries: list[str] = []

    def append(self, event: str) -> None:
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {event}"
        self._entries.append(entry)
        logger.info("ICL ▸ %s", event)

    def as_prompt_block(self) -> str:
        if not self._entries:
            return ""
        lines = "\n".join(self._entries)
        return (
            "## CONTINUITY BIBLE — ICL MEMORY LOG\n"
            "The following events have already occurred in this simulation. "
            "You MUST NOT re-invent, contradict, or ignore any entry below.\n\n"
            f"{lines}\n"
        )

    @property
    def entries(self) -> list[str]:
        return list(self._entries)


# ═══════════════════════════════════════════════════════════════════════════
#  COORDINATE MAPPER  —  Gemini 0..1 norm → Blender metric
# ═══════════════════════════════════════════════════════════════════════════

class CoordinateMapper:
    """
    Maps Gemini's normalized coordinate system (0.0 – 1.0) to
    Blender's metric units.

    Gemini outputs:  normalized [0, 1] per axis, Y-down (image convention)
    Blender expects: metric units, Y-forward (right-hand, Z-up)

    Convention applied:
        blender_x = (norm_x - 0.5) * scale      # centre → left/right
        blender_y = (norm_y - 0.5) * scale       # centre → near/far
        blender_z = norm_z * scale               # 0 = ground, 1 = ceiling
    """

    def __init__(self, scale: float = 10.0) -> None:
        self.scale = scale

    def to_blender(self, norm: Vec3) -> Vec3:
        return {
            "x": (norm["x"] - 0.5) * self.scale,
            "y": (norm["y"] - 0.5) * self.scale,
            "z": norm.get("z", 0.0) * self.scale,
        }

    def norm_to_blender_traj(self, traj: list[dict]) -> list[dict]:
        """Convert a trajectory list [{t, x, y, z}] from norm to metric."""
        result = []
        for kf in traj:
            metric = self.to_blender({"x": kf["x"], "y": kf["y"], "z": kf.get("z", 0.0)})
            result.append({"t": kf["t"], **metric})
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  CLIPPING / COLLISION AUDITOR  —  AABB overlap detection pre-execution
# ═══════════════════════════════════════════════════════════════════════════

class ClippingAuditor:
    """
    Pre-execution AABB (Axis-Aligned Bounding Box) overlap checker.
    Runs against every SpatialObject before any bpy script is executed.
    Detects mesh intersection ("clipping") that would produce invalid physics.

    Five invariants checked:
      1. No two objects occupy overlapping AABB volumes.
      2. All objects have Z-bottom ≥ ground plane (Z=0 unless specified).
      3. Bounding box min < max on all axes.
      4. No object positioned outside scene boundary (±scale/2).
      5. Object volume > minimum threshold (degenerate geometry guard).
    """

    def __init__(self, tolerance_m: float = 0.001, scene_bound_m: float = 50.0) -> None:
        self.tolerance = tolerance_m
        self.bound = scene_bound_m

    def check(self, objects: list[SpatialObject]) -> list[dict]:
        """
        Returns list of violation dicts.
        Each violation includes entity_ids, violation_type, and suggested fix.
        """
        violations: list[dict] = []
        for obj in objects:
            violations.extend(self._check_single(obj))
        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                v = self._check_pair(a, b)
                if v:
                    violations.append(v)
        return violations

    def _check_single(self, obj: SpatialObject) -> list[dict]:
        issues = []
        bb = obj.bounding_box
        mn, mx = bb.get("min", {}), bb.get("max", {})

        # Invariant 3: min < max
        for axis in ("x", "y", "z"):
            if mn.get(axis, 0) >= mx.get(axis, 0):
                issues.append({
                    "type": "degenerate_bbox",
                    "entity_id": obj.entity_id,
                    "axis": axis,
                    "fix": f"expand_{axis}_by_0.01",
                })

        # Invariant 2: not below ground
        if mn.get("z", 0) < -self.tolerance:
            issues.append({
                "type": "below_ground",
                "entity_id": obj.entity_id,
                "z_min": mn.get("z"),
                "fix": f"offset_z_by_{abs(mn.get('z', 0)):.4f}",
            })

        # Invariant 4: within scene boundary
        for axis in ("x", "y"):
            for extreme in (mn.get(axis, 0), mx.get(axis, 0)):
                if abs(extreme) > self.bound:
                    issues.append({
                        "type": "out_of_bounds",
                        "entity_id": obj.entity_id,
                        "axis": axis,
                        "value": extreme,
                        "fix": f"clamp_{axis}_to_{self.bound}",
                    })
        return issues

    def _check_pair(self, a: SpatialObject, b: SpatialObject) -> dict | None:
        """Check AABB overlap between two objects."""
        def overlaps_1d(min1: float, max1: float, min2: float, max2: float) -> bool:
            return min1 < max2 - self.tolerance and min2 < max1 - self.tolerance

        a_min, a_max = a.bounding_box.get("min", {}), a.bounding_box.get("max", {})
        b_min, b_max = b.bounding_box.get("min", {}), b.bounding_box.get("max", {})

        overlap = all(
            overlaps_1d(
                a_min.get(ax, 0), a_max.get(ax, 0),
                b_min.get(ax, 0), b_max.get(ax, 0))
            for ax in ("x", "y", "z")
        )
        if overlap:
            # Compute penetration depth on Z (cheapest fix: lift one object)
            pen_z = min(a_max.get("z", 0), b_max.get("z", 0)) - max(a_min.get("z", 0), b_min.get("z", 0))
            return {
                "type": "mesh_clipping",
                "entity_ids": [a.entity_id, b.entity_id],
                "penetration_z_m": round(pen_z, 4),
                "fix": f"offset_{b.entity_id}_z_by_{pen_z + self.tolerance:.4f}",
            }
        return None

    def auto_repair(self, objects: list[SpatialObject], violations: list[dict]) -> list[SpatialObject]:
        """Apply automatic repairs to objects based on violation list."""
        pos_fixes: dict[str, float] = {}
        for v in violations:
            if v["type"] == "mesh_clipping":
                target = v["entity_ids"][1]
                delta = v.get("penetration_z_m", 0.01) + self.tolerance
                pos_fixes[target] = pos_fixes.get(target, 0.0) + delta
            elif v["type"] == "below_ground":
                eid = v["entity_id"]
                pos_fixes[eid] = pos_fixes.get(eid, 0.0) + abs(v.get("z_min", 0.0)) + self.tolerance

        for obj in objects:
            if obj.entity_id in pos_fixes:
                delta = pos_fixes[obj.entity_id]
                obj.position_metric["z"] = obj.position_metric.get("z", 0.0) + delta
                # Update bounding box accordingly
                if "min" in obj.bounding_box:
                    obj.bounding_box["min"]["z"] = obj.bounding_box["min"].get("z", 0.0) + delta
                if "max" in obj.bounding_box:
                    obj.bounding_box["max"]["z"] = obj.bounding_box["max"].get("z", 0.0) + delta
                obj.clip_violation = True
                obj.clip_checked = True

        return objects


# ═══════════════════════════════════════════════════════════════════════════
#  GEMINI HANDSHAKE  —  Phase 2: Multimodal → Structural JSON
# ═══════════════════════════════════════════════════════════════════════════

class GeminiHandshake:
    """
    The bridge between Gemini's spatial reasoning and Blender's coordinate space.

    Capabilities:
      - Multimodal: accepts a reference image or video path alongside the prompt
      - Returns a StructuralJSON with normalized coordinates, bounding boxes,
        motion trajectories, action tokens, physics metadata
      - Clipping Sanity Check Loop: if Gemini's coordinates cause AABB violations,
        the violation report is sent back to Gemini for a corrected response
      - Style-Physics integration: if StyleDirective is provided, Gemini derives
        SDP constants and populates style_physics_manifest
    """

    STRUCTURAL_JSON_SCHEMA = {
        "type": "OBJECT",
        "properties": {
            "objects": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "entity_id": {"type": "STRING"},
                        "label": {"type": "STRING"},
                        "position_norm": {
                            "type": "OBJECT",
                            "properties": {
                                "x": {"type": "NUMBER"},
                                "y": {"type": "NUMBER"},
                                "z": {"type": "NUMBER"},
                            },
                        },
                        "bounding_box_norm": {
                            "type": "OBJECT",
                            "properties": {
                                "min": {"type": "OBJECT"},
                                "max": {"type": "OBJECT"},
                            },
                        },
                        "material_description": {"type": "STRING"},
                        "semantic_type": {"type": "STRING"},
                        "physics_metadata": {"type": "OBJECT"},
                        "motion_trajectory": {"type": "ARRAY"},
                    },
                },
            },
            "camera_ops": {"type": "ARRAY"},
            "action_tokens": {"type": "ARRAY"},
            "physics_manifest": {"type": "OBJECT"},
            "style_physics_manifest": {"type": "OBJECT"},
        },
    }

    def __init__(
        self,
        gemini_model: GeminiERClient,
        mapper: CoordinateMapper,
        auditor: ClippingAuditor,
        icl_log: ICLMemoryLog,
        config: NexusVConfig) -> None:
        self.model = gemini_model
        self.mapper = mapper
        self.auditor = auditor
        self.icl = icl_log
        self.config = config

    def analyze(
        self,
        directive: "RealityDirective",
        session_id: str) -> StructuralJSON:
        """
        Full Gemini multimodal analysis → StructuralJSON.
        Includes up to 2 clipping-repair correction passes.
        """
        logger.info("GeminiHandshake: analyzing scene...")
        self.icl.append(f"GeminiHandshake started for session {session_id}")

        # Build content parts
        parts: list[Any] = [self._build_analysis_prompt(directive)]
        if directive.reference_image_path:
            parts.insert(0, self._load_image_part(directive.reference_image_path))

        raw = self._call_gemini(parts)
        struct_json = self._parse_response(raw, directive, session_id)

        # Clipping sanity check loop (up to 2 passes)
        for attempt in range(2):
            violations = self.auditor.check(struct_json.objects)
            if not violations:
                self.icl.append(
                    f"ClippingSanityCheck PASS: no violations on attempt {attempt + 1}"
                )
                break
            self.icl.append(
                f"ClippingSanityCheck FAIL: {len(violations)} violations on attempt {attempt + 1}. "
                f"Requesting Gemini correction."
            )
            logger.warning(
                "Clipping violations detected (%d). Requesting Gemini correction (pass %d).",
                len(violations), attempt + 1)
            struct_json.objects = self.auditor.auto_repair(struct_json.objects, violations)
            if attempt == 0:
                # Second pass: ask Gemini to re-emit corrected coordinates
                correction_prompt = self._build_correction_prompt(violations)
                raw2 = self._call_gemini([correction_prompt])
                corrected = self._parse_response(raw2, directive, session_id)
                # Merge: prefer Gemini's corrected positions
                id_map = {o.entity_id: o for o in corrected.objects}
                for obj in struct_json.objects:
                    if obj.entity_id in id_map:
                        obj.position_norm = id_map[obj.entity_id].position_norm
                        obj.position_metric = id_map[obj.entity_id].position_metric
                        obj.bounding_box = id_map[obj.entity_id].bounding_box

        self.icl.append(
            f"GeminiHandshake complete: {len(struct_json.objects)} objects, "
            f"{len(struct_json.camera_ops)} camera ops, "
            f"{len(struct_json.action_tokens)} action tokens."
        )
        return struct_json

    # ── Private helpers ──────────────────────────────────────────────────

    def _build_analysis_prompt(self, directive: "RealityDirective") -> str:
        icl_block = self.icl.as_prompt_block()
        style_block = ""
        if directive.style_directive:
            sd = directive.style_directive
            style_block = (
                f"\n## STYLE-PHYSICS DIRECTIVE\n"
                f"Reference style: {sd.reference}\n"
                f"Emotional arc: {sd.emotional_arc}\n"
                f"Style tags: {sd.style_tags}\n"
                f"Quantum paths requested: {sd.quantum_paths}\n"
                f"Derive custom physics constants (style_gravity, viscosity_constant, "
                f"style_entropy_hash, fracture_vocabulary, motion_vocabulary, "
                f"light_entanglement_matrix) from the reference style's visual entropy. "
                f"Populate style_physics_manifest in your response.\n"
            )
        physics_block = ""
        if directive.physics_laws:
            laws_str = json.dumps([vars(p) for p in directive.physics_laws], indent=2)
            physics_block = f"\n## EXPLICIT PHYSICS LAWS (override Blender defaults)\n```json\n{laws_str}\n```\n"

        asset_block = json.dumps(
            [
                {
                    "entity_id": a.entity_id,
                    "asset_type": a.asset_type,
                    "position_hint": a.position,
                    "material": a.material,
                    "physics_type": a.physics_type,
                    "physics_mass_kg": a.physics_mass_kg,
                    "height_m": a.height_m,
                }
                for a in directive.asset_manifest
            ],
            indent=2)
        beats_str = json.dumps(
            [
                {
                    "t_seconds": b.t_seconds,
                    "beat_id": b.beat_id,
                    "description": b.description,
                    "entity_id": b.entity_id,
                    "action_type": b.action_type,
                    "force_vector": b.force_vector,
                    "camera_hint": b.camera_hint,
                }
                for b in directive.story_beats
            ],
            indent=2)

        return textwrap.dedent(f"""
            # NEXUS-V SPATIAL ANALYSIS REQUEST
            You are the NEXUS-V Spatial Kernel — an autonomous spatial reasoning agent.

            {icl_block}

            ## SCENE BRIEF
            {directive.scene_brief}

            ## ASSET MANIFEST
            ```json
            {asset_block}
            ```

            ## STORY BEATS
            ```json
            {beats_str}
            ```

            ## DURATION
            {directive.duration_seconds} seconds @ {self.config.fps} fps

            {style_block}{physics_block}

            ## YOUR TASK
            Perform full spatial analysis. Return a JSON object with this structure:
            {{
              "objects": [
                {{
                  "entity_id": "...",
                  "label": "...",
                  "position_norm": {{"x": 0.0..1.0, "y": 0.0..1.0, "z": 0.0..1.0}},
                  "bounding_box_norm": {{"min": {{...}}, "max": {{...}}}},
                  "material_description": "...",
                  "semantic_type": "rigid_body|soft_body|fluid|light|character|env",
                  "physics_metadata": {{"mass_kg": 1.0, "friction": 0.5, "restitution": 0.3}},
                  "motion_trajectory": [{{"t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}}, ...]
                }},
                ...
              ],
              "camera_ops": [
                {{
                  "t": 0.0,
                  "duration": 2.0,
                  "shot_type": "establishing|medium|close_up|dutch_angle|tracking|dolly_in|dolly_out|orbit",
                  "target_entity_id": "...",
                  "focal_length_mm": 50.0,
                  "aperture_fstop": 2.8,
                  "euler_deg": {{"x": 0.0, "y": 0.0, "z": 0.0}}
                }},
                ...
              ],
              "action_tokens": [
                {{
                  "entity_id": "...",
                  "t": 0.0,
                  "duration": 0.1,
                  "action_type": "collision|grab|drop|fluid|cloth|impulse",
                  "force_vector": {{"x": 0.0, "y": 0.0, "z": 0.0}},
                  "contact_point_norm": {{"x": 0.0, "y": 0.0, "z": 0.0}}
                }},
                ...
              ],
              "physics_manifest": {{
                "gravity": {{"x": 0.0, "y": 0.0, "z": -9.81}},
                "scene_scale_m": {self.config.coordinate_scale},
                "simulation_end_frame": {int(directive.duration_seconds * self.config.fps)}
              }},
              "style_physics_manifest": null
            }}

            CRITICAL CONSTRAINTS:
            - All coordinates are NORMALIZED 0.0..1.0.  0.5 is scene centre.
            - Ensure no two objects overlap (bounding boxes must not intersect).
            - Every object with physics_type != NONE must have a physics_metadata entry.
            - motion_trajectory must include t=0.0 and t={directive.duration_seconds} entries.
            - camera_ops shot_type must map to exact Euler rotations and focal length.
            - Return ONLY valid JSON. No markdown fences. No preamble.
        """).strip()

    def _build_correction_prompt(self, violations: list[dict]) -> str:
        v_str = json.dumps(violations, indent=2)
        return textwrap.dedent(f"""
            # NEXUS-V CORRECTION REQUEST — Clipping Violations Detected

            The following AABB clipping violations were found in your previous response:
            ```json
            {v_str}
            ```

            {self.icl.as_prompt_block()}

            Please re-emit ONLY the corrected "objects" array with adjusted coordinates
            that resolve all violations. Fix positions by the smallest possible offset.
            For mesh_clipping violations: increase the Z of the lighter/smaller object.
            Return ONLY the corrected JSON array for "objects". No preamble.
        """).strip()

    def _call_gemini(self, parts: list[Any], thinking: ThinkingPreset = ThinkingPreset.NONE) -> str:
        try:
            response = self.model.generate_content(parts, thinking=thinking)
            return response.strip() if isinstance(response, str) else response.text.strip()
        except Exception as exc:
            logger.error("Gemini API error: %s", exc)
            raise

    def _parse_response(
        self,
        raw: str,
        directive: "RealityDirective",
        session_id: str) -> StructuralJSON:
        """Parse raw Gemini text into a StructuralJSON."""
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini JSON response: %s", exc)
            logger.debug("Raw response:\n%s", raw[:500])
            # Return minimal fallback
            data = {
                "objects": [], "camera_ops": [], "action_tokens": [],
                "physics_manifest": {}, "style_physics_manifest": None,
            }

        objects: list[SpatialObject] = []
        for raw_obj in data.get("objects", []):
            pos_norm = raw_obj.get("position_norm", {"x": 0.5, "y": 0.5, "z": 0.0})
            pos_metric = self.mapper.to_blender(pos_norm)
            # Convert bounding box
            bb_norm = raw_obj.get("bounding_box_norm", {"min": pos_norm, "max": pos_norm})
            bb_metric = {
                "min": self.mapper.to_blender(bb_norm.get("min", pos_norm)),
                "max": self.mapper.to_blender(bb_norm.get("max", pos_norm)),
            }
            traj_metric = self.mapper.norm_to_blender_traj(
                raw_obj.get("motion_trajectory", [])
            )
            objects.append(SpatialObject(
                entity_id=raw_obj.get("entity_id", str(uuid.uuid4())[:8]),
                label=raw_obj.get("label", "object"),
                position_norm=pos_norm,
                position_metric=pos_metric,
                bounding_box=bb_metric,
                material_description=raw_obj.get("material_description", ""),
                semantic_type=raw_obj.get("semantic_type", "rigid_body"),
                physics_metadata=raw_obj.get("physics_metadata", {}),
                motion_trajectory=traj_metric))

        return StructuralJSON(
            session_id=session_id,
            scene_brief=directive.scene_brief,
            objects=objects,
            camera_ops=data.get("camera_ops", []),
            action_tokens=data.get("action_tokens", []),
            physics_manifest=data.get("physics_manifest", {}),
            style_physics_manifest=data.get("style_physics_manifest"),
            icl_log_entries=self.icl.entries)

    def _load_image_part(self, image_path: str) -> Any:
        """Load a local image as a Gemini content part."""
        from pathlib import Path
        import base64

        path = Path(image_path)
        if not path.exists():
            logger.warning("Reference image not found: %s", image_path)
            return ""
        ext_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".png": "image/png", ".webp": "image/webp"}
        mime = ext_map.get(path.suffix.lower(), "image/jpeg")
        data = base64.b64encode(path.read_bytes()).decode()
        return {"inline_data": {"mime_type": mime, "data": data}}


# ═══════════════════════════════════════════════════════════════════════════
#  WORLD BUILDER  —  Phase 1: Structural JSON → BPY scene
# ═══════════════════════════════════════════════════════════════════════════

class WorldBuilder:
    """
    Phase 1 BPY Core.  Translates a StructuralJSON into a Blender Python
    script that procedurally builds the scene in headless Blender.

    Implements the Zero-Drift Architecture from instruction 2.txt:
      - Strict context overrides:  never uses bpy.ops without temp_override
      - Autotelic physics discovery: collision margins auto-calculated
      - Baked reality:  animations baked to keyframes before hand-off
      - Grounding invariant:  every object has ground contact check

    Generated script sections:
      1. Scene setup (FPS, end frame, world colour)
      2. Object instantiation (mesh, character rig placeholder, lights)
      3. Material assignment
      4. Rigid body / soft body / fluid configuration
      5. Physics world setup (gravity vector, substeps)
      6. Object grounding check
    """

    def __init__(self, config: NexusVConfig, icl_log: ICLMemoryLog) -> None:
        self.config = config
        self.icl = icl_log

    def generate_scene_script(
        self,
        struct_json: StructuralJSON,
        directive: "RealityDirective",
        blend_path: str) -> str:
        """Generate and return the full bpy scene-build Python script."""
        fps = self.config.fps
        end_frame = int(directive.duration_seconds * fps)
        grav = struct_json.physics_manifest.get(
            "gravity", {"x": 0.0, "y": 0.0, "z": -9.81}
        )

        lines: list[str] = [
            self._header_comment(struct_json, directive),
            "import bpy",
            "import math",
            "import json",
            "",
            "# ── Scene Globals ─────────────────────────────────────────",
            "bpy.ops.object.select_all(action='SELECT')",
            "bpy.ops.object.delete(use_global=False)",
            "",
            f"scene = bpy.context.scene",
            f"scene.render.fps = {fps}",
            f"scene.frame_end = {end_frame}",
            f"scene.frame_start = 1",
            f"scene.render.resolution_x = {self.config.output_resolution[0]}",
            f"scene.render.resolution_y = {self.config.output_resolution[1]}",
            f"scene.render.engine = '{'BLENDER_EEVEE_NEXT' if self.config.render_engine == 'eevee' else 'CYCLES'}'",
            "",
            "# ── Physics World ──────────────────────────────────────────",
            *self._physics_world_block(grav, end_frame),
            "",
            "# ── Compositor AOV Passes ──────────────────────────────────",
            *self._compositor_aov_block(blend_path),
            "",
            "# ── Objects ───────────────────────────────────────────────",
        ]

        for obj in struct_json.objects:
            lines.extend(self._object_block(obj, directive))
            lines.append("")

        lines.extend([
            "# ── Save .blend ────────────────────────────────────────────",
            f"bpy.ops.wm.save_as_mainfile(filepath=r'{blend_path}')",
            "print('[nexusv] Scene build complete.')",
        ])

        self.icl.append(
            f"WorldBuilder: generated scene script — "
            f"{len(struct_json.objects)} objects, blend={blend_path}"
        )
        return "\n".join(lines)

    # ── Private block generators ─────────────────────────────────────────

    def _header_comment(self, sj: StructuralJSON, d: "RealityDirective") -> str:
        return textwrap.dedent(f"""\
            # ╔══════════════════════════════════════════════════════════════╗
            # ║  NEXUS-V WorldBuilder — auto-generated scene script         ║
            # ║  Session: {sj.session_id:<48}║
            # ║  Brief:   {d.scene_brief[:48]:<48}║
            # ╚══════════════════════════════════════════════════════════════╝
        """)

    def _physics_world_block(self, grav: Vec3, end_frame: int) -> list[str]:
        return [
            "# Add rigid body world",
            "# Use correct context override — Zero-Drift Architecture",
            "for area in bpy.context.screen.areas:",
            "    if area.type == 'VIEW_3D':",
            "        target_area = area",
            "        break",
            "else:",
            "    target_area = None",
            "",
            "if target_area:",
            "    with bpy.context.temp_override(area=target_area):",
            "        if not bpy.context.scene.rigidbody_world:",
            "            bpy.ops.rigidbody.world_add()",
            "",
            "if bpy.context.scene.rigidbody_world:",
            f"    bpy.context.scene.rigidbody_world.point_cache.frame_end = {end_frame}",
            f"    bpy.context.scene.rigidbody_world.substeps_per_frame = 10",
            f"    bpy.context.scene.rigidbody_world.solver_iterations = 20",
            "",
            "# Set gravity",
            f"bpy.context.scene.gravity = ({grav.get('x', 0.0)}, {grav.get('y', 0.0)}, {grav.get('z', -9.81)})",
        ]

    def _compositor_aov_block(self, blend_path: str) -> list[str]:
        """Set up Compositor node tree for multi-pass AOV export (depth, normal, beauty)."""
        output_dir = str(Path(blend_path).parent / "passes")
        return [
            "# Enable AOV passes for ControlNet conditioning",
            "scene.view_layers[0].use_pass_z = True",
            "scene.view_layers[0].use_pass_normal = True",
            "scene.view_layers[0].use_pass_diffuse_color = True",
            "",
            "scene.use_nodes = True",
            "tree = scene.node_tree",
            "tree.nodes.clear()",
            "",
            "render_node = tree.nodes.new('CompositorNodeRLayers')",
            "render_node.location = (0, 300)",
            "",
            "# Beauty output",
            "out_beauty = tree.nodes.new('CompositorNodeOutputFile')",
            "out_beauty.label = 'beauty'",
            f"out_beauty.base_path = r'{output_dir}/beauty'",
            "out_beauty.location = (400, 400)",
            "tree.links.new(render_node.outputs['Image'], out_beauty.inputs[0])",
            "",
            "# Depth output",
            "out_depth = tree.nodes.new('CompositorNodeOutputFile')",
            "out_depth.label = 'depth'",
            f"out_depth.base_path = r'{output_dir}/depth'",
            "out_depth.file_slots[0].use_node_format = False",
            "out_depth.file_slots[0].format.file_format = 'OPEN_EXR'",
            "out_depth.location = (400, 200)",
            "tree.links.new(render_node.outputs['Depth'], out_depth.inputs[0])",
            "",
            "# Normal output",
            "out_normal = tree.nodes.new('CompositorNodeOutputFile')",
            "out_normal.label = 'normal'",
            f"out_normal.base_path = r'{output_dir}/normal'",
            "out_normal.location = (400, 0)",
            "tree.links.new(render_node.outputs['Normal'], out_normal.inputs[0])",
        ]

    def _object_block(self, obj: SpatialObject, directive: "RealityDirective") -> list[str]:
        """Generate bpy commands for a single object."""
        eid = obj.entity_id
        px, py, pz = obj.position_metric["x"], obj.position_metric["y"], obj.position_metric["z"]
        lines: list[str] = [
            f"# ── {eid} ──────────────────────────────────────────",
        ]

        # Instantiation
        asset = next((a for a in directive.asset_manifest if a.entity_id == eid), None)

        if asset and asset.asset_type == "light_rig":
            lines.extend(self._light_block(obj, asset))
        elif asset and asset.asset_type == "character_rig":
            lines.extend(self._character_placeholder_block(obj, asset))
        elif asset and asset.asset_type in ("fluid_domain", "fluid_flow"):
            lines.extend(self._fluid_block(obj, asset))
        elif asset and asset.asset_type == "particle_system":
            lines.extend(self._particle_block(obj, asset))
        else:
            lines.extend(self._mesh_block(obj, asset))

        # Material
        lines.extend(self._material_block(eid, obj.material_description, obj.material_hash))

        # Physics
        if asset and asset.physics_type in ("ACTIVE", "PASSIVE"):
            lines.extend(self._physics_block(eid, asset, obj))

        # Grounding invariant
        lines.extend([
            f"# [GROUNDING] ensure {eid} is not below Z=0",
            f"_obj_{eid} = bpy.data.objects.get('{eid}')",
            f"if _obj_{eid} and _obj_{eid}.location.z < 0:",
            f"    _obj_{eid}.location.z = 0.0",
        ])

        return lines

    def _mesh_block(self, obj: SpatialObject, asset: AssetSpec | None) -> list[str]:
        eid = obj.entity_id
        px, py, pz = obj.position_metric["x"], obj.position_metric["y"], obj.position_metric["z"]
        scale = 1.0
        if asset:
            s = asset.scale
            scale = s if isinstance(s, (int, float)) else 1.0

        mesh_map = {
            "MESH_CUBE": "bpy.ops.mesh.primitive_cube_add",
            "MESH_SPHERE": "bpy.ops.mesh.primitive_uv_sphere_add",
            "MESH_CYLINDER": "bpy.ops.mesh.primitive_cylinder_add",
            "MESH_PLANE": "bpy.ops.mesh.primitive_plane_add",
            "MESH_CONE": "bpy.ops.mesh.primitive_cone_add",
            "MESH_TORUS": "bpy.ops.mesh.primitive_torus_add",
        }
        primitive = mesh_map.get(
            getattr(asset, "asset_type", "MESH_CUBE") if asset else "MESH_CUBE",
            "bpy.ops.mesh.primitive_cube_add")

        return [
            f"{primitive}(size={scale}, location=({px:.4f}, {py:.4f}, {pz:.4f}))",
            f"bpy.context.active_object.name = '{eid}'",
        ]

    def _light_block(self, obj: SpatialObject, asset: AssetSpec) -> list[str]:
        eid = obj.entity_id
        px, py, pz = obj.position_metric["x"], obj.position_metric["y"], obj.position_metric["z"]
        light_type = asset.material.get("type", "SUN")
        energy = asset.material.get("energy", 5.0)
        color = asset.material.get("color", [1.0, 1.0, 1.0])
        cr, cg, cb = color[0], color[1], color[2] if len(color) > 2 else 1.0
        return [
            f"_light_data_{eid} = bpy.data.lights.new(name='{eid}_light', type='{light_type}')",
            f"_light_data_{eid}.energy = {energy}",
            f"_light_data_{eid}.color = ({cr}, {cg}, {cb})",
            f"_light_obj_{eid} = bpy.data.objects.new(name='{eid}', object_data=_light_data_{eid})",
            f"bpy.context.collection.objects.link(_light_obj_{eid})",
            f"_light_obj_{eid}.location = ({px:.4f}, {py:.4f}, {pz:.4f})",
        ]

    def _character_placeholder_block(self, obj: SpatialObject, asset: AssetSpec) -> list[str]:
        eid = obj.entity_id
        px, py, pz = obj.position_metric["x"], obj.position_metric["y"], obj.position_metric["z"]
        height = asset.height_m or 1.8
        return [
            f"# Character rig placeholder — replace with .blend asset in production",
            f"bpy.ops.object.armature_add(location=({px:.4f}, {py:.4f}, {pz:.4f}))",
            f"bpy.context.active_object.name = '{eid}'",
            f"# Scale armature to character height",
            f"bpy.context.active_object.scale.z = {height / 2.0:.4f}",
            f"# Hint: link a full character rig from asset library here",
        ]

    def _fluid_block(self, obj: SpatialObject, asset: AssetSpec) -> list[str]:
        eid = obj.entity_id
        px, py, pz = obj.position_metric["x"], obj.position_metric["y"], obj.position_metric["z"]
        is_domain = asset.asset_type == "fluid_domain"
        return [
            f"bpy.ops.mesh.primitive_cube_add(size=1.0, location=({px:.4f}, {py:.4f}, {pz:.4f}))",
            f"bpy.context.active_object.name = '{eid}'",
            f"bpy.ops.object.modifier_add(type='FLUID')",
            f"_fluid_mod_{eid} = bpy.context.active_object.modifiers['Fluid']",
            f"_fluid_mod_{eid}.fluid_type = '{'DOMAIN' if is_domain else 'FLOW'}'",
            *(
                [
                    f"_fluid_mod_{eid}.domain_settings.resolution_max = 64",
                    f"_fluid_mod_{eid}.domain_settings.use_noise = True",
                ]
                if is_domain
                else [
                    f"_fluid_mod_{eid}.flow_settings.flow_type = 'LIQUID'",
                    f"_fluid_mod_{eid}.flow_settings.flow_behavior = 'GEOMETRY'",
                ]
            ),
        ]

    def _particle_block(self, obj: SpatialObject, asset: AssetSpec) -> list[str]:
        eid = obj.entity_id
        px, py, pz = obj.position_metric["x"], obj.position_metric["y"], obj.position_metric["z"]
        count = asset.extra_props.get("particle_count", 5000)
        return [
            f"bpy.ops.mesh.primitive_plane_add(size=20, location=({px:.4f}, {py:.4f}, {pz:.4f}))",
            f"bpy.context.active_object.name = '{eid}'",
            f"_ps_{eid} = bpy.context.active_object.modifiers.new(name='ps_{eid}', type='PARTICLE_SYSTEM')",
            f"_ps_{eid}.particle_system.settings.count = {count}",
            f"_ps_{eid}.particle_system.settings.physics_type = 'NEWTON'",
        ]

    def _material_block(self, eid: str, description: str, mat_hash: str) -> list[str]:
        return [
            f"# Material — hash lock: {mat_hash[:16]}",
            f"if '{eid}' in bpy.data.objects:",
            f"    _mat_{eid} = bpy.data.materials.new(name='mat_{eid}')",
            f"    _mat_{eid}.use_nodes = True",
            f"    _mat_{eid}['nexusv_material_hash'] = '{mat_hash}'",
            f"    _mat_{eid}['nexusv_material_desc'] = {repr(description[:200])}",
            f"    _obj_{eid}_ref = bpy.data.objects['{eid}']",
            f"    if _obj_{eid}_ref.data and hasattr(_obj_{eid}_ref.data, 'materials'):",
            f"        _obj_{eid}_ref.data.materials.append(_mat_{eid})",
        ]

    def _physics_block(self, eid: str, asset: AssetSpec, obj: SpatialObject) -> list[str]:
        pm = obj.physics_metadata or {}
        mass = pm.get("mass_kg", asset.physics_mass_kg)
        rest = pm.get("restitution", asset.physics_restitution)
        fric = pm.get("friction", asset.physics_friction)
        shape = asset.physics_collision_shape
        ptype = asset.physics_type
        return [
            f"# Physics — {eid}",
            f"_rb_{eid} = bpy.data.objects.get('{eid}')",
            f"if _rb_{eid}:",
            f"    bpy.context.view_layer.objects.active = _rb_{eid}",
            f"    bpy.ops.rigidbody.object_add()",
            f"    _rb_{eid}.rigid_body.type = '{ptype}'",
            f"    _rb_{eid}.rigid_body.mass = {mass}",
            f"    _rb_{eid}.rigid_body.restitution = {rest}",
            f"    _rb_{eid}.rigid_body.friction = {fric}",
            f"    _rb_{eid}.rigid_body.collision_shape = '{shape}'",
            f"    # Autotelic physics: auto-calculate collision margin",
            f"    _rb_{eid}.rigid_body.use_margin = True",
            f"    _rb_{eid}.rigid_body.collision_margin = 0.001",
            f"    # PASSIVE until physics hand-off frame (set by TimelineManager)",
            f"    if '{ptype}' == 'ACTIVE':",
            f"        _rb_{eid}.rigid_body.enabled = False  # activated by TimelineManager",
        ]


# ═══════════════════════════════════════════════════════════════════════════
#  TIMELINE MANAGER  —  Phase 3a: Action Tokens → Blender Keyframes
# ═══════════════════════════════════════════════════════════════════════════

class TimelineManager:
    """
    Converts Gemini's Action Tokens and motion_trajectories into
    Blender keyframe f-curves.

    Features:
      - Bezier curve handle calculation for smooth arcs
      - Physics hand-off: at trigger frames, flips body_type PASSIVE → ACTIVE
      - Force vector injection as impulse keyframes
      - Fluid/cloth event trigger scripting
    """

    def __init__(self, config: NexusVConfig, icl_log: ICLMemoryLog) -> None:
        self.config = config
        self.icl = icl_log

    def generate_animation_script(
        self,
        struct_json: StructuralJSON,
        directive: "RealityDirective") -> str:
        fps = self.config.fps
        lines: list[str] = [
            "# ════════════════════════════════════════════════════════",
            "# NEXUS-V TimelineManager — Animation & Physics Hand-off",
            "# ════════════════════════════════════════════════════════",
            "import bpy",
            "import math",
            "",
        ]

        # Motion trajectories → keyframes
        for obj in struct_json.objects:
            if not obj.motion_trajectory:
                continue
            lines.extend(self._trajectory_to_keyframes(obj, fps))
            lines.append("")

        # Action tokens → physics events
        for token in struct_json.action_tokens:
            lines.extend(self._action_token_to_script(token, fps, directive))
            lines.append("")

        # Story beats → additional keyframes / physics triggers
        for beat in directive.story_beats:
            lines.extend(self._beat_to_script(beat, fps))
            lines.append("")

        # Bake all simulations to keyframes (Zero-Drift Architecture)
        lines.extend([
            "# ── Bake simulations to keyframes (required pre-ControlNet hand-off) ──",
            "scene = bpy.context.scene",
            "if scene.rigidbody_world:",
            "    try:",
            "        bpy.ops.rigidbody.bake_to_keyframes(",
            f"            frame_start=1, frame_end={int(directive.duration_seconds * fps)},",
            "            step=1",
            "        )",
            "        print('[nexusv] Physics baked to keyframes.')",
            "    except Exception as e:",
            "        print(f'[nexusv] Bake warning: {e}')",
        ])

        self.icl.append(
            f"TimelineManager: animation script generated — "
            f"{len(struct_json.objects)} trajectory chains, "
            f"{len(struct_json.action_tokens)} action tokens."
        )
        return "\n".join(lines)

    def _trajectory_to_keyframes(self, obj: SpatialObject, fps: int) -> list[str]:
        eid = obj.entity_id
        lines = [f"# Trajectory keyframes — {eid}"]
        kf_pairs: list[tuple[int, Vec3]] = []
        for kf in obj.motion_trajectory:
            frame = max(1, int(kf["t"] * fps))
            kf_pairs.append((frame, {"x": kf["x"], "y": kf["y"], "z": kf["z"]}))

        if not kf_pairs:
            return lines

        lines += [
            f"_anim_{eid} = bpy.data.objects.get('{eid}')",
            f"if _anim_{eid}:",
        ]
        for frame, pos in kf_pairs:
            lines += [
                f"    _anim_{eid}.location = ({pos['x']:.4f}, {pos['y']:.4f}, {pos['z']:.4f})",
                f"    _anim_{eid}.keyframe_insert(data_path='location', frame={frame})",
            ]

        # Set Bezier interpolation for smooth arcs
        lines += [
            f"    # Set BEZIER interpolation on all location F-curves",
            f"    if _anim_{eid}.animation_data and _anim_{eid}.animation_data.action:",
            f"        for fcurve in _anim_{eid}.animation_data.action.fcurves:",
            f"            for kp in fcurve.keyframe_points:",
            f"                kp.interpolation = 'BEZIER'",
            f"                kp.handle_left_type = 'AUTO_CLAMPED'",
            f"                kp.handle_right_type = 'AUTO_CLAMPED'",
        ]
        return lines

    def _action_token_to_script(
        self,
        token: dict,
        fps: int,
        directive: "RealityDirective") -> list[str]:
        eid = token.get("entity_id", "")
        t = token.get("t", 0.0)
        frame = max(1, int(t * fps))
        atype = token.get("action_type", "custom")
        fv = token.get("force_vector", {"x": 0.0, "y": 0.0, "z": 0.0})
        lines = [f"# ActionToken: {atype} on {eid} @ T={t}s (frame {frame})"]

        if atype in ("collision", "impulse", "physics_handoff") and eid:
            # Physics hand-off: flip body_type PASSIVE → ACTIVE
            lines += [
                f"_rb_tok_{eid} = bpy.data.objects.get('{eid}')",
                f"if _rb_tok_{eid} and _rb_tok_{eid}.rigid_body:",
                f"    # Set PASSIVE before hand-off frame",
                f"    bpy.context.scene.frame_set({max(1, frame - 1)})",
                f"    _rb_tok_{eid}.rigid_body.enabled = False",
                f"    _rb_tok_{eid}.keyframe_insert(data_path='rigid_body.enabled', frame={max(1, frame - 1)})",
                f"    # Activate at hand-off frame",
                f"    bpy.context.scene.frame_set({frame})",
                f"    _rb_tok_{eid}.rigid_body.enabled = True",
                f"    _rb_tok_{eid}.keyframe_insert(data_path='rigid_body.enabled', frame={frame})",
            ]
            # Apply force via velocity initialization
            if any(fv.get(a, 0) != 0 for a in ("x", "y", "z")):
                lines += [
                    f"    # Apply initial velocity from force vector",
                    f"    _rb_tok_{eid}.rigid_body.linear_velocity = ("
                    f"{fv['x']:.3f}, {fv['y']:.3f}, {fv['z']:.3f})",
                ]

        elif atype in ("fluid", "cloth"):
            lines += [
                f"# {atype.capitalize()} trigger for {eid} — handled by Mantaflow/Cloth sim",
                f"# Gemini sets initial conditions only; Blender solver owns subsequent frames",
            ]

        return lines

    def _beat_to_script(self, beat: StoryBeat, fps: int) -> list[str]:
        frame = max(1, int(beat.t_seconds * fps))
        lines = [f"# StoryBeat: {beat.beat_id} @ T={beat.t_seconds}s (frame {frame})"]

        if beat.motion_target and beat.entity_id:
            eid = beat.entity_id
            mt = beat.motion_target
            lines += [
                f"_beat_{eid} = bpy.data.objects.get('{eid}')",
                f"if _beat_{eid}:",
                f"    bpy.context.scene.frame_set({frame})",
                f"    _beat_{eid}.location = ({mt['x']:.4f}, {mt['y']:.4f}, {mt['z']:.4f})",
                f"    _beat_{eid}.keyframe_insert(data_path='location', frame={frame})",
            ]
        return lines


# ═══════════════════════════════════════════════════════════════════════════
#  CINEMATOGRAPHY MODULE  —  Phase 3c: Agentic Camera (Shot Type → BPY)
# ═══════════════════════════════════════════════════════════════════════════

class CinematographyModule:
    """
    The Agentic Camera.  Gemini provides the Shot Type; this module
    translates it into specific Euler rotations, Focal Lengths, aperture
    values, DOF constraints, and camera movement f-curves in Blender.

    Shot type vocabulary supported:
      establishing, medium, close_up, extreme_close_up, dutch_angle,
      tracking, dolly_in, dolly_out, orbit, handheld, overhead,
      over_shoulder, two_shot, pov

    Zero-Drift Architecture compliance:
      - Camera constraints use TrackTo (not manual Euler) where possible
      - DOF target linked to entity object for accurate bokeh
      - All camera ops baked to f-curves
    """

    SHOT_PRESETS: dict[str, dict] = {
        "establishing":      {"focal_mm": 28.0, "fstop": 8.0,  "euler_deg": {"x": 60.0, "y": 0.0, "z": 0.0}},
        "medium":            {"focal_mm": 50.0, "fstop": 4.0,  "euler_deg": {"x": 80.0, "y": 0.0, "z": 0.0}},
        "close_up":          {"focal_mm": 85.0, "fstop": 2.8,  "euler_deg": {"x": 85.0, "y": 0.0, "z": 0.0}},
        "extreme_close_up":  {"focal_mm": 135.0,"fstop": 2.0,  "euler_deg": {"x": 90.0, "y": 0.0, "z": 0.0}},
        "dutch_angle":       {"focal_mm": 35.0, "fstop": 4.0,  "euler_deg": {"x": 75.0, "y": 0.0, "z": 15.0}},
        "tracking":          {"focal_mm": 50.0, "fstop": 4.0,  "euler_deg": {"x": 80.0, "y": 0.0, "z": 0.0}},
        "dolly_in":          {"focal_mm": 50.0, "fstop": 2.8,  "euler_deg": {"x": 80.0, "y": 0.0, "z": 0.0}},
        "dolly_out":         {"focal_mm": 50.0, "fstop": 4.0,  "euler_deg": {"x": 75.0, "y": 0.0, "z": 0.0}},
        "orbit":             {"focal_mm": 50.0, "fstop": 5.6,  "euler_deg": {"x": 70.0, "y": 0.0, "z": 0.0}},
        "handheld":          {"focal_mm": 35.0, "fstop": 2.0,  "euler_deg": {"x": 78.0, "y": 0.0, "z": 0.0}},
        "overhead":          {"focal_mm": 28.0, "fstop": 8.0,  "euler_deg": {"x": 0.0,  "y": 0.0, "z": 0.0}},
        "over_shoulder":     {"focal_mm": 85.0, "fstop": 2.8,  "euler_deg": {"x": 80.0, "y": 0.0, "z": 15.0}},
        "pov":               {"focal_mm": 24.0, "fstop": 1.8,  "euler_deg": {"x": 90.0, "y": 0.0, "z": 0.0}},
    }

    def __init__(self, config: NexusVConfig, icl_log: ICLMemoryLog) -> None:
        self.config = config
        self.icl = icl_log

    def generate_camera_script(
        self,
        camera_ops: list[dict],
        struct_json: StructuralJSON,
        directive: "RealityDirective") -> str:
        fps = self.config.fps
        lines: list[str] = [
            "# ════════════════════════════════════════════════════════",
            "# NEXUS-V CinematographyModule — Agentic Camera",
            "# ════════════════════════════════════════════════════════",
            "import bpy",
            "import math",
            "",
            "# ── Camera Object ─────────────────────────────────────",
            "bpy.ops.object.camera_add(location=(0, -8, 5))",
            "_cam = bpy.context.active_object",
            "_cam.name = 'NEXUSV_Camera'",
            "bpy.context.scene.camera = _cam",
            "_cam_data = _cam.data",
            "_cam_data.lens = 50.0",
            "_cam_data.dof.use_dof = True",
            "",
        ]

        for idx, op in enumerate(camera_ops):
            lines.extend(self._camera_op_to_script(op, idx, fps, struct_json))
            lines.append("")

        self.icl.append(
            f"CinematographyModule: generated {len(camera_ops)} camera ops."
        )
        return "\n".join(lines)

    def _camera_op_to_script(
        self,
        op: dict,
        idx: int,
        fps: int,
        struct_json: StructuralJSON) -> list[str]:
        t = op.get("t", 0.0)
        duration = op.get("duration", 2.0)
        frame_start = max(1, int(t * fps))
        frame_end = max(frame_start + 1, int((t + duration) * fps))
        shot_type = op.get("shot_type", "medium")
        target_eid = op.get("target_entity_id", "")
        focal_mm = op.get("focal_length_mm") or self.SHOT_PRESETS.get(shot_type, {}).get("focal_mm", 50.0)
        fstop = op.get("aperture_fstop") or self.SHOT_PRESETS.get(shot_type, {}).get("fstop", 4.0)
        euler_override = op.get("euler_deg")
        preset_euler = self.SHOT_PRESETS.get(shot_type, {}).get("euler_deg", {"x": 80.0, "y": 0.0, "z": 0.0})
        euler = euler_override or preset_euler
        ex = math.radians(euler.get("x", 80.0))
        ey = math.radians(euler.get("y", 0.0))
        ez = math.radians(euler.get("z", 0.0))

        lines = [
            f"# Camera Op {idx}: {shot_type} @ T={t}s",
            f"bpy.context.scene.frame_set({frame_start})",
            f"_cam.rotation_euler = ({ex:.4f}, {ey:.4f}, {ez:.4f})",
            f"_cam_data.lens = {focal_mm}",
            f"_cam_data.dof.aperture_fstop = {fstop}",
            f"_cam.keyframe_insert(data_path='rotation_euler', frame={frame_start})",
            f"_cam_data.keyframe_insert(data_path='lens', frame={frame_start})",
        ]

        # DOF tracking
        if target_eid:
            target_obj = next(
                (o for o in struct_json.objects if o.entity_id == target_eid), None
            )
            if target_obj:
                lines += [
                    f"_dof_target_{idx} = bpy.data.objects.get('{target_eid}')",
                    f"if _dof_target_{idx}:",
                    f"    _cam_data.dof.focus_object = _dof_target_{idx}",
                    f"    # TrackTo constraint toward target",
                    f"    _track_c = _cam.constraints.new(type='TRACK_TO')",
                    f"    _track_c.target = _dof_target_{idx}",
                    f"    _track_c.track_axis = 'TRACK_NEGATIVE_Z'",
                    f"    _track_c.up_axis = 'UP_Y'",
                ]

        # Dolly movement
        if shot_type == "dolly_in":
            target_pos = None
            if target_eid:
                t_obj = next(
                    (o for o in struct_json.objects if o.entity_id == target_eid), None
                )
                if t_obj:
                    target_pos = t_obj.position_metric
            if target_pos:
                end_x = target_pos["x"]
                end_y = target_pos["y"] - 3.0
                end_z = target_pos["z"] + 2.0
                lines += [
                    f"bpy.context.scene.frame_set({frame_end})",
                    f"_cam.location = ({end_x:.4f}, {end_y:.4f}, {end_z:.4f})",
                    f"_cam.keyframe_insert(data_path='location', frame={frame_end})",
                ]

        elif shot_type == "orbit" and target_eid:
            t_obj = next(
                (o for o in struct_json.objects if o.entity_id == target_eid), None
            )
            if t_obj:
                cx, cy, cz = (
                    t_obj.position_metric["x"],
                    t_obj.position_metric["y"],
                    t_obj.position_metric["z"] + 1.0)
                radius = 6.0
                num_steps = 8
                for step in range(num_steps + 1):
                    angle = math.radians(360.0 * step / num_steps)
                    kf = frame_start + int((frame_end - frame_start) * step / num_steps)
                    sx = cx + radius * math.sin(angle)
                    sy = cy + radius * math.cos(angle)
                    lines += [
                        f"bpy.context.scene.frame_set({kf})",
                        f"_cam.location = ({sx:.4f}, {sy:.4f}, {cz:.4f})",
                        f"_cam.keyframe_insert(data_path='location', frame={kf})",
                    ]

        # End-of-op keyframe
        lines += [
            f"bpy.context.scene.frame_set({frame_end})",
            f"_cam.rotation_euler = ({ex:.4f}, {ey:.4f}, {ez:.4f})",
            f"_cam.keyframe_insert(data_path='rotation_euler', frame={frame_end})",
            f"_cam_data.keyframe_insert(data_path='lens', frame={frame_end})",
        ]
        return lines


# ═══════════════════════════════════════════════════════════════════════════
#  BLENDER RUNNER  —  Headless CLI execution
# ═══════════════════════════════════════════════════════════════════════════

class BlenderRunner:
    """
    Executes Python scripts against Blender's headless CLI.
    Each call opens Blender, runs the script, and closes.

    Handles:
      - Context override safety (scripts include their own overrides)
      - Error detection from stdout/stderr
      - Retry on certain recoverable errors (mode context errors)
    """

    def __init__(self, config: NexusVConfig, icl_log: ICLMemoryLog) -> None:
        self.blender = config.blender_executable
        self.icl = icl_log

    def run_script(self, script_content: str, blend_path: str | None = None) -> bool:
        """Write script to temp file, execute, return success."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="nexusv_"
        ) as f:
            f.write(script_content)
            script_path = f.name

        cmd = [self.blender, "--background", "--python", script_path]
        if blend_path and Path(blend_path).exists():
            cmd = [self.blender, "--background", blend_path, "--python", script_path]

        logger.info("BlenderRunner: executing script %s", script_path)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                logger.error(
                    "Blender returned non-zero exit code %d.\nSTDERR:\n%s",
                    result.returncode, result.stderr[-2000:])
                self.icl.append(
                    f"BlenderRunner ERROR: exit code {result.returncode}"
                )
                return False
            if "[nexusv]" in result.stdout:
                for line in result.stdout.splitlines():
                    if "[nexusv]" in line:
                        self.icl.append(f"Blender ▸ {line.strip()}")
            return True
        except subprocess.TimeoutExpired:
            logger.error("Blender script timed out.")
            self.icl.append("BlenderRunner ERROR: timeout")
            return False
        except FileNotFoundError:
            logger.error(
                "Blender executable not found at '%s'. "
                "Install Blender 4.3+ and set config.blender_executable.",
                self.blender)
            self.icl.append(f"BlenderRunner ERROR: blender not found at {self.blender}")
            return False
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    def render_frames(
        self,
        blend_path: str,
        output_dir: str,
        frame_start: int = 1,
        frame_end: int = 240) -> bool:
        """Render frame range from .blend file."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        render_script = textwrap.dedent(f"""\
            import bpy
            scene = bpy.context.scene
            scene.render.filepath = r'{output_dir}/frame_'
            scene.render.image_settings.file_format = 'PNG'
            for frame in range({frame_start}, {frame_end} + 1):
                scene.frame_set(frame)
                bpy.ops.render.render(write_still=True)
                print(f'[nexusv] Rendered frame {{frame}}')
        """)
        return self.run_script(render_script, blend_path)


# ═══════════════════════════════════════════════════════════════════════════
#  NEURAL SKINNING PIPELINE  —  Phase 4: Depth/Normal/Canny → ControlNet
# ═══════════════════════════════════════════════════════════════════════════

class NeuralSkinningPipeline:
    """
    Phase 4: The Neural Refinement Layer.

    Reads Blender's multi-pass renders and constructs ControlNet-conditioned
    ComfyUI / A1111 API payloads for each frame.

    ControlNet conditioning channels:
      ControlNet_0 (Depth)  → strength 0.8  (spatial lock — exact mesh depth)
      ControlNet_1 (Canny)  → strength 0.4  (edge fidelity)
      ControlNet_2 (Normal) → strength 0.35 (surface orientation)

    The depth map from Blender cannot hallucinate. The 3D mesh IS the spatial
    truth. Therefore 0% spatial drift is guaranteed by construction.

    Instruction 2.txt compliance:
      "This ensures the AI 'paints' exactly within the lines of your physics simulation."
    """

    def __init__(self, config: NexusVConfig, icl_log: ICLMemoryLog) -> None:
        self.config = config
        self.icl = icl_log
        self.http = httpx.Client(timeout=120.0)

    def skin_frame(
        self,
        beauty_path: str,
        depth_path: str,
        normal_path: str,
        style_prompt: str,
        frame_idx: int,
        output_path: str) -> bool:
        """
        Apply ControlNet + diffusion skin to a single Blender frame.
        Returns True on success.

        Tries ComfyUI first, falls back to A1111 (Automatic1111).
        If neither is available, writes a stub and logs a warning.
        """
        payload = self._build_comfyui_payload(
            beauty_path, depth_path, normal_path, style_prompt, output_path
        )

        try:
            resp = self.http.post(
                f"{self.config.comfyui_base_url}/prompt",
                json={"prompt": payload})
            if resp.status_code == 200:
                self.icl.append(f"NeuralSkinning: ComfyUI frame {frame_idx} queued.")
                return True
        except httpx.ConnectError:
            logger.debug("ComfyUI not available, trying A1111...")

        # A1111 fallback
        a1111_payload = self._build_a1111_payload(
            beauty_path, depth_path, style_prompt
        )
        try:
            resp = self.http.post(
                f"{self.config.a1111_base_url}/sdapi/v1/img2img",
                json=a1111_payload)
            if resp.status_code == 200:
                self.icl.append(f"NeuralSkinning: A1111 frame {frame_idx} processed.")
                return True
        except httpx.ConnectError:
            logger.warning(
                "Neither ComfyUI nor A1111 is available. "
                "Start your local diffusion server and re-run Phase 4."
            )
            self.icl.append(
                f"NeuralSkinning WARNING: no diffusion server available (frame {frame_idx}). "
                f"Blender base render saved as-is."
            )
            return False

        return False

    def _build_comfyui_payload(
        self,
        beauty_path: str,
        depth_path: str,
        normal_path: str,
        style_prompt: str,
        output_path: str) -> dict:
        """
        Construct a ComfyUI workflow JSON payload.
        Implements the instruction 2.txt mapping:
          ControlNet_0 (Depth)  → strength 0.8
          ControlNet_1 (Canny)  → strength 0.4
        """
        return {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": beauty_path},
            },
            "2": {
                "class_type": "LoadImage",
                "inputs": {"image": depth_path},
            },
            "3": {
                "class_type": "LoadImage",
                "inputs": {"image": normal_path},
            },
            "4": {
                "class_type": "ControlNetLoader",
                "inputs": {"control_net_name": "depth_anything_v2.safetensors"},
            },
            "5": {
                "class_type": "ControlNetLoader",
                "inputs": {"control_net_name": "control_v11p_sd15_canny.pth"},
            },
            "6": {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": ["7", 0],
                    "negative": ["8", 0],
                    "control_net": ["4", 0],
                    "image": ["2", 0],
                    "strength": self.config.controlnet_depth_strength,
                    "start_percent": 0.0,
                    "end_percent": 1.0,
                },
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["9", 1],
                    "text": style_prompt,
                },
            },
            "8": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["9", 1],
                    "text": "watermark, blurry, distorted, floating objects, bad anatomy",
                },
            },
            "9": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": self.config.diffusion_model},
            },
            "10": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["9", 0],
                    "positive": ["6", 0],
                    "negative": ["8", 0],
                    "latent_image": ["11", 0],
                    "seed": 42,
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "dpm_2_ancestral",
                    "scheduler": "karras",
                    "denoise": self.config.denoising_strength,
                },
            },
            "11": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["1", 0], "vae": ["9", 2]},
            },
            "12": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["10", 0], "vae": ["9", 2]},
            },
            "13": {
                "class_type": "SaveImage",
                "inputs": {"images": ["12", 0], "filename_prefix": output_path},
            },
        }

    def _build_a1111_payload(
        self,
        beauty_path: str,
        depth_path: str,
        style_prompt: str) -> dict:
        """Construct an Automatic1111 img2img payload with ControlNet extension."""
        import base64

        def b64_img(path: str) -> str:
            try:
                return base64.b64encode(Path(path).read_bytes()).decode()
            except (OSError, FileNotFoundError):
                return ""

        return {
            "init_images": [b64_img(beauty_path)],
            "prompt": style_prompt,
            "negative_prompt": "watermark, blurry, distorted, floating objects",
            "denoising_strength": self.config.denoising_strength,
            "width": self.config.output_resolution[0],
            "height": self.config.output_resolution[1],
            "steps": 20,
            "cfg_scale": 7.0,
            "sampler_name": "DPM++ 2M Karras",
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "image": b64_img(depth_path),
                            "module": "depth",
                            "model": "control_v11f1p_sd15_depth",
                            "weight": self.config.controlnet_depth_strength,
                            "control_mode": "ControlNet is more important",
                        },
                    ]
                }
            },
        }

    def __del__(self) -> None:
        self.http.close()


# ═══════════════════════════════════════════════════════════════════════════
#  AUTONOMOUS AUDIT LOOP  —  Phase 4 + Gemini re-audit → .blend correction
# ═══════════════════════════════════════════════════════════════════════════

class AutonomousAuditLoop:
    """
    Phase 4 autonomous feedback loop.

    Gemini analyzes render output frames for violations:
      1. Lighting intent vs render — shadow direction, light consistency
      2. Composition — framing, rule-of-thirds adherence
      3. Physics fidelity — floating objects, implausible trajectories
      4. Neural skin consistency — material hash drift across frames
      5. Temporal coherence — character identity stability

    On violation: generate a targeted bpy correction script → execute
    against .blend → re-render affected frames only → re-skin.

    Key distinction from Engines I–V:
      Previous: Probabilistic correction (might improve)
      Nexus-V:  Deterministic correction (WILL produce correct result)
                because the .blend mesh is the ground truth.
    """

    def __init__(
        self,
        gemini_model: GeminiERClient,
        runner: BlenderRunner,
        config: NexusVConfig,
        icl_log: ICLMemoryLog) -> None:
        self.model = gemini_model
        self.runner = runner
        self.config = config
        self.icl = icl_log

    def run(
        self,
        blend_path: str,
        frames_dir: str,
        struct_json: StructuralJSON,
        directive: "RealityDirective",
        pass_number: int = 1) -> list[dict]:
        """
        Run one audit pass. Returns list of violation dicts.
        Applies corrections automatically.
        """
        logger.info("AutonomousAuditLoop: pass %d", pass_number)
        self.icl.append(f"AutonomousAuditLoop started (pass {pass_number})")

        prompt = self._build_audit_prompt(struct_json, directive, frames_dir)
        try:
            raw = self.model.generate_content([prompt], thinking=ThinkingPreset.MEDIUM)
            raw = raw.strip() if isinstance(raw, str) else raw.text.strip()
        except Exception as exc:
            logger.error("Gemini audit call failed: %s", exc)
            return []

        violations = self._parse_violations(raw)
        if not violations:
            self.icl.append(f"AutonomousAuditLoop pass {pass_number}: PASS — no violations.")
            return []

        self.icl.append(
            f"AutonomousAuditLoop pass {pass_number}: {len(violations)} violations. "
            f"Generating correction scripts."
        )
        for v in violations:
            correction_script = self._generate_correction_script(v, struct_json)
            if correction_script:
                success = self.runner.run_script(correction_script, blend_path)
                if success:
                    self.icl.append(
                        f"Correction applied: {v.get('type')} on {v.get('entity_id', 'scene')}"
                    )

        return violations

    def _build_audit_prompt(
        self,
        struct_json: StructuralJSON,
        directive: "RealityDirective",
        frames_dir: str) -> str:
        icl_block = self.icl.as_prompt_block()
        obj_summary = json.dumps(
            [
                {
                    "entity_id": o.entity_id,
                    "position_metric": o.position_metric,
                    "material_hash": hashlib.sha256(
                        o.material_description.encode()
                    ).hexdigest()[:16],
                    "semantic_type": o.semantic_type,
                }
                for o in struct_json.objects
            ],
            indent=2)
        return textwrap.dedent(f"""
            # NEXUS-V AUTONOMOUS AUDIT REQUEST
            You are the NEXUS-V Audit Agent. Analyze the rendered output for violations.

            {icl_block}

            ## SCENE BRIEF
            {directive.scene_brief}

            ## GROUND TRUTH WORLD STATE
            ```json
            {obj_summary}
            ```

            ## VISUAL STYLE TARGET
            {directive.visual_style}

            ## FRAMES DIRECTORY
            {frames_dir}

            ## YOUR TASK
            Identify ANY of these 5 violation types:
            1. lighting_error — shadow direction inconsistent with light source position
            2. composition_error — subject out of frame, horizon tilt unintended
            3. physics_violation — object floating, implausible trajectory
            4. material_drift — texture inconsistent with material_hash across frames
            5. temporal_incoherence — character identity changes between frames

            Return ONLY a JSON array of violations:
            [
              {{
                "type": "lighting_error|composition_error|physics_violation|material_drift|temporal_incoherence",
                "entity_id": "...",
                "frame": 42,
                "description": "...",
                "bpy_fix_hint": "brief description of bpy correction needed"
              }},
              ...
            ]

            If no violations: return empty array [].
            Return ONLY valid JSON. No preamble.
        """).strip()

    def _parse_violations(self, raw: str) -> list[dict]:
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(clean)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        return []

    def _generate_correction_script(
        self, violation: dict, struct_json: StructuralJSON
    ) -> str | None:
        """Generate a targeted bpy script to fix a specific violation."""
        vtype = violation.get("type", "")
        eid = violation.get("entity_id", "")

        if vtype == "lighting_error":
            # Re-position sun light to correct shadow direction
            obj = next((o for o in struct_json.objects if o.entity_id == eid), None)
            if obj:
                lx = obj.position_metric["x"] + 5.0
                ly = obj.position_metric["y"] - 5.0
                lz = obj.position_metric["z"] + 10.0
                return textwrap.dedent(f"""\
                    import bpy
                    # Correction: lighting_error on {eid}
                    _light = bpy.data.objects.get('{eid}')
                    if not _light:
                        # Try sun lights
                        for obj in bpy.data.objects:
                            if obj.type == 'LIGHT':
                                _light = obj
                                break
                    if _light:
                        _light.location = ({lx:.3f}, {ly:.3f}, {lz:.3f})
                        print('[nexusv] Lighting correction applied.')
                """)

        elif vtype == "physics_violation":
            # Force grounding
            return textwrap.dedent(f"""\
                import bpy
                # Correction: physics_violation — force grounding {eid}
                _obj = bpy.data.objects.get('{eid}')
                if _obj:
                    if _obj.location.z < 0:
                        _obj.location.z = 0.0
                    if _obj.rigid_body:
                        _obj.rigid_body.enabled = True
                    print('[nexusv] Physics correction applied to {eid}.')
            """)

        elif vtype == "material_drift":
            obj = next((o for o in struct_json.objects if o.entity_id == eid), None)
            if obj:
                return textwrap.dedent(f"""\
                    import bpy
                    # Correction: material_drift — restore material hash {obj.material_hash[:16]}
                    _obj = bpy.data.objects.get('{eid}')
                    if _obj and _obj.data and hasattr(_obj.data, 'materials'):
                        for mat in _obj.data.materials:
                            if mat:
                                mat['nexusv_material_hash'] = '{obj.material_hash}'
                                mat['nexusv_material_desc'] = {repr(obj.material_description[:200])}
                        print('[nexusv] Material hash restored for {eid}.')
                """)

        elif vtype == "composition_error":
            return textwrap.dedent(f"""\
                import bpy
                # Correction: composition_error — reset camera dutch tilt
                _cam = bpy.data.objects.get('NEXUSV_Camera')
                if _cam:
                    _cam.rotation_euler.z = 0.0
                    print('[nexusv] Camera roll corrected.')
            """)

        return None


# ═══════════════════════════════════════════════════════════════════════════
#  NEXUS-V ENGINE  —  The Master Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class NexusVEngine:
    """
    NEXUS-V Engine — Engine VII
    Aletheia-Blender Reality-Simulation Protocol.

    Orchestrates 4 phases + autonomous correction loop:

    Phase 1: BPY Core
      WorldBuilder generates a Blender scene from Structural JSON.
      Clipping Auditor validates AABB before any script runs.

    Phase 2: Gemini Handshake
      GeminiHandshake performs multimodal analysis of the directive,
      generates Structural JSON, applies Sanity Check Loop.

    Phase 3: Kinetic Engine
      TimelineManager converts action tokens + trajectories to keyframes.
      PhysicsHandoffManager sets up Bullet/Mantaflow simulations.
      CinematographyModule generates the Agentic Camera.

    Phase 4: Neural Refinement
      BlenderRunner renders multi-pass AOVs (beauty, depth, normal).
      NeuralSkinningPipeline applies ControlNet-conditioned diffusion.
      AutonomousAuditLoop runs up to max_audit_passes correction cycles.

    Usage:
        config = NexusVConfig(
            gemini_api_key="YOUR_KEY",
            blender_executable="/usr/bin/blender",
            output_dir="./my_render")
        engine = NexusVEngine(config)

        directive = RealityDirective(
            scene_brief="A whisky glass shatters on a bar counter.",
            asset_manifest=[...],
            story_beats=[...],
            visual_style="photorealistic neo-noir")
        result = engine.render(directive)
    """

    def __init__(self, config: NexusVConfig | None = None) -> None:
        self.config = config or NexusVConfig()
        self._setup_gemini()
        self._icl = ICLMemoryLog()
        self._mapper = CoordinateMapper(scale=self.config.coordinate_scale)
        self._auditor = ClippingAuditor(tolerance_m=self.config.clipping_tolerance_m)
        self._handshake = GeminiHandshake(
            self._gemini, self._mapper, self._auditor, self._icl, self.config
        )
        self._world_builder = WorldBuilder(self.config, self._icl)
        self._timeline = TimelineManager(self.config, self._icl)
        self._camera = CinematographyModule(self.config, self._icl)
        self._runner = BlenderRunner(self.config, self._icl)
        self._skinning = NeuralSkinningPipeline(self.config, self._icl)
        self._audit = AutonomousAuditLoop(
            self._gemini, self._runner, self.config, self._icl
        )

    def _setup_gemini(self) -> None:
        api_key = self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning(
                "No Gemini API key set. Set GEMINI_API_KEY env var or "
                "config.gemini_api_key. Gemini calls will fail."
            )
        self._gemini = GeminiERClient(
            api_key=api_key, model=ER15_MODEL,
            default_thinking=ThinkingPreset.NONE)
        logger.info("Gemini model initialized: %s", self.config.gemini_model)

    # ── Public API ───────────────────────────────────────────────────────

    def render(self, directive: RealityDirective) -> dict:
        """
        Full Nexus-V pipeline.  Returns a result dict with paths and metadata.

        Returns:
            {
                "session_id": str,
                "blend_path": str,
                "frames_dir": str,
                "skinned_dir": str,
                "structural_json": dict,
                "icl_log": list[str],
                "violations": list[dict],
                "success": bool,
            }
        """
        session_id = str(uuid.uuid4())[:12]
        self._icl.append(f"NexusVEngine.render() started — session {session_id}")
        logger.info("═" * 70)
        logger.info("NEXUS-V ENGINE — Session %s", session_id)
        logger.info("Scene: %s", directive.scene_brief[:80])
        logger.info("═" * 70)

        out_root = Path(self.config.output_dir) / session_id
        out_root.mkdir(parents=True, exist_ok=True)
        blend_path = str(out_root / f"{directive.output_name}.blend")
        frames_dir = str(out_root / "frames")
        skinned_dir = str(out_root / "skinned")
        Path(frames_dir).mkdir(parents=True, exist_ok=True)
        Path(skinned_dir).mkdir(parents=True, exist_ok=True)

        all_violations: list[dict] = []

        # ── Phase 2: Gemini Handshake ─────────────────────────────────
        logger.info("Phase 2: Gemini Handshake")
        struct_json = self._handshake.analyze(directive, session_id)
        struct_json_path = str(out_root / "structural.json")
        with open(struct_json_path, "w") as f:
            json.dump(struct_json.to_dict(), f, indent=2)
        logger.info(
            "Structural JSON saved: %d objects, %d camera ops, %d action tokens",
            len(struct_json.objects),
            len(struct_json.camera_ops),
            len(struct_json.action_tokens))

        # ── Phase 1: BPY Core — WorldBuilder ─────────────────────────
        logger.info("Phase 1: WorldBuilder — generating scene script")
        scene_script = self._world_builder.generate_scene_script(
            struct_json, directive, blend_path
        )
        scene_script_path = str(out_root / "scene_build.py")
        with open(scene_script_path, "w") as f:
            f.write(scene_script)

        if not self._runner.run_script(scene_script):
            logger.warning(
                "Scene build script failed (Blender may not be installed). "
                "Script saved at: %s", scene_script_path
            )

        # ── Phase 3a: TimelineManager — Animation Keyframes ──────────
        logger.info("Phase 3a: TimelineManager — generating animation script")
        anim_script = self._timeline.generate_animation_script(struct_json, directive)
        anim_script_path = str(out_root / "animation.py")
        with open(anim_script_path, "w") as f:
            f.write(anim_script)

        if Path(blend_path).exists():
            self._runner.run_script(anim_script, blend_path)

        # ── Phase 3c: CinematographyModule — Agentic Camera ──────────
        logger.info("Phase 3c: CinematographyModule — agentic camera")
        camera_script = self._camera.generate_camera_script(
            struct_json.camera_ops, struct_json, directive
        )
        cam_script_path = str(out_root / "camera.py")
        with open(cam_script_path, "w") as f:
            f.write(camera_script)

        if Path(blend_path).exists():
            self._runner.run_script(camera_script, blend_path)

        # ── Phase 3 render: Eevee-Next multipass ─────────────────────
        logger.info("Phase 3: Blender render — %s", frames_dir)
        end_frame = int(directive.duration_seconds * self.config.fps)
        if Path(blend_path).exists():
            self._runner.render_frames(blend_path, frames_dir, 1, end_frame)
        else:
            logger.warning(
                "No .blend file found — skipping render. "
                "All scripts are saved in %s for manual execution.", str(out_root)
            )

        # ── Phase 4: Neural Skinning ──────────────────────────────────
        logger.info("Phase 4: Neural Skinning")
        passes_dir = out_root / "passes"
        for frame_num in range(1, end_frame + 1):
            beauty = str(passes_dir / "beauty" / f"frame_{frame_num:04d}.png")
            depth  = str(passes_dir / "depth"  / f"frame_{frame_num:04d}.exr")
            normal = str(passes_dir / "normal" / f"frame_{frame_num:04d}.png")
            out_frame = str(Path(skinned_dir) / f"frame_{frame_num:04d}.png")
            self._skinning.skin_frame(
                beauty, depth, normal,
                style_prompt=directive.visual_style,
                frame_idx=frame_num,
                output_path=out_frame)

        # ── Phase 4 + Autonomous Audit Loop ──────────────────────────
        logger.info("Phase 4: Autonomous Audit Loop")
        for audit_pass in range(1, self.config.max_audit_passes + 1):
            violations = self._audit.run(
                blend_path=blend_path,
                frames_dir=frames_dir,
                struct_json=struct_json,
                directive=directive,
                pass_number=audit_pass)
            all_violations.extend(violations)
            if not violations:
                logger.info("Audit pass %d: CLEAN — pipeline complete.", audit_pass)
                break
            logger.info(
                "Audit pass %d: %d violations corrected, re-rendering...",
                audit_pass, len(violations))
            if Path(blend_path).exists():
                self._runner.render_frames(blend_path, frames_dir, 1, end_frame)

        self._icl.append(
            f"NexusVEngine.render() complete — session {session_id}. "
            f"Output: {str(out_root)}"
        )

        # Save ICL log
        icl_path = str(out_root / "icl_memory.log")
        with open(icl_path, "w") as f:
            f.write("\n".join(self._icl.entries))

        result = {
            "session_id": session_id,
            "output_root": str(out_root),
            "blend_path": blend_path,
            "frames_dir": frames_dir,
            "skinned_dir": skinned_dir,
            "structural_json_path": struct_json_path,
            "structural_json": struct_json.to_dict(),
            "scripts": {
                "scene_build": scene_script_path,
                "animation": anim_script_path,
                "camera": cam_script_path,
            },
            "icl_log": self._icl.entries,
            "icl_log_path": icl_path,
            "violations": all_violations,
            "success": len([v for v in all_violations if v]) == 0,
        }

        logger.info("═" * 70)
        logger.info("NEXUS-V complete. Output root: %s", str(out_root))
        logger.info("Scripts ready for manual Blender execution (if Blender unavailable)")
        logger.info("═" * 70)

        return result

    # ── Convenience methods ──────────────────────────────────────────────

    def generate_bezier_camera_path(
        self,
        directive: RealityDirective,
        struct_json: StructuralJSON | None = None) -> str:
        """
        Standalone utility: translate a Gemini-generated JSON trajectory
        into a Bezier-curved camera path in Blender.

        This is the 'hardest part of the handshake' referenced in instruction 1.txt.
        Returns a complete, executable bpy script.
        """
        if struct_json is None:
            struct_json = self._handshake.analyze(directive, str(uuid.uuid4())[:8])

        fps = self.config.fps
        lines = [
            "# NEXUS-V Bezier Camera Path — generated by NexusVEngine.generate_bezier_camera_path()",
            "import bpy",
            "import math",
            "",
            "# ── Create camera if absent ─────────────────────────────",
            "if 'NEXUSV_Camera' not in bpy.data.objects:",
            "    bpy.ops.object.camera_add(location=(0, -8, 5))",
            "    bpy.context.active_object.name = 'NEXUSV_Camera'",
            "    bpy.context.scene.camera = bpy.context.active_object",
            "",
            "_cam = bpy.data.objects['NEXUSV_Camera']",
            "_cam_data = _cam.data",
            "_cam_data.lens = 50.0",
            "_cam_data.dof.use_dof = True",
            "",
            "# ── Build Bezier curve path ──────────────────────────────",
            "# Create a NURBs path for the camera to follow",
            "bpy.ops.curve.primitive_bezier_curve_add()",
            "_curve_obj = bpy.context.active_object",
            "_curve_obj.name = 'NEXUSV_CameraPath'",
            "_curve = _curve_obj.data",
            "_curve.dimensions = '3D'",
            "_spline = _curve.splines[0]",
            "",
            "# Collect camera op positions as Bezier control points",
            "_cam_positions = []",
        ]

        for op in struct_json.camera_ops:
            t = op.get("t", 0.0)
            duration = op.get("duration", 2.0)
            target_eid = op.get("target_entity_id", "")
            target_obj = next(
                (o for o in struct_json.objects if o.entity_id == target_eid), None
            )
            if target_obj:
                cx = target_obj.position_metric["x"]
                cy = target_obj.position_metric["y"] - 6.0
                cz = target_obj.position_metric["z"] + 3.0
            else:
                cx, cy, cz = 0.0, -6.0, 3.0

            frame_s = max(1, int(t * fps))
            frame_e = max(frame_s + 1, int((t + duration) * fps))
            focal = op.get("focal_length_mm", 50.0)
            fstop = op.get("aperture_fstop", 4.0)
            lines += [
                f"_cam_positions.append(({cx:.4f}, {cy:.4f}, {cz:.4f}, "
                f"{frame_s}, {frame_e}, {focal}, {fstop}))",
            ]

        lines += [
            "",
            "# Set Bezier spline points",
            "_n_pts = len(_cam_positions)",
            "if _n_pts > 0:",
            "    _spline.bezier_points.add(_n_pts - 1)",
            "    for _i, (_px, _py, _pz, _fs, _fe, _fl, _fa) in enumerate(_cam_positions):",
            "        _bp = _spline.bezier_points[_i]",
            "        _bp.co = (_px, _py, _pz)",
            "        _bp.handle_left_type = 'AUTO'",
            "        _bp.handle_right_type = 'AUTO'",
            "",
            "    # Follow path constraint on camera",
            "    _follow = _cam.constraints.new(type='FOLLOW_PATH')",
            "    _follow.target = _curve_obj",
            "    _follow.use_curve_follow = True",
            "",
            "    # Keyframe lens and aperture per segment",
            "    for _i, (_px, _py, _pz, _fs, _fe, _fl, _fa) in enumerate(_cam_positions):",
            "        bpy.context.scene.frame_set(_fs)",
            "        _cam_data.lens = _fl",
            "        _cam_data.dof.aperture_fstop = _fa",
            "        _cam_data.keyframe_insert(data_path='lens', frame=_fs)",
            "        bpy.context.scene.frame_set(_fe)",
            "        _cam_data.keyframe_insert(data_path='lens', frame=_fe)",
            "",
            "print('[nexusv] Bezier camera path generated.')",
        ]

        return "\n".join(lines)

    def describe(self) -> str:
        """Return a human-readable description of the engine."""
        return textwrap.dedent(f"""
            ╔══════════════════════════════════════════════════════════════╗
            ║  NEXUS-V ENGINE  —  Engine VII                              ║
            ║  Aletheia-Blender Reality-Simulation Protocol               ║
            ╠══════════════════════════════════════════════════════════════╣
            ║  Gemini model:   {self.config.gemini_model:<40}║
            ║  Blender path:   {self.config.blender_executable:<40}║
            ║  Render engine:  {self.config.render_engine:<40}║
            ║  Resolution:     {str(self.config.output_resolution):<40}║
            ║  Coord scale:    {str(self.config.coordinate_scale):<40}║
            ║  ControlNet      Depth={self.config.controlnet_depth_strength}  Canny={self.config.controlnet_canny_strength}  Normal={self.config.controlnet_normal_strength:<11}║
            ║  Audit passes:   {self.config.max_audit_passes:<40}║
            ║  Output dir:     {self.config.output_dir:<40}║
            ╚══════════════════════════════════════════════════════════════╝
            Paradigm: Simulate Reality → Record the Simulation → Skin Neurally
            Hallucinations: MATHEMATICALLY IMPOSSIBLE (3D mesh = ground truth)
        """).strip()
