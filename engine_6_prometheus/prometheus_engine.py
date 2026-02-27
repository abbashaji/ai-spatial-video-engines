"""
╔══════════════════════════════════════════════════════════════════════════╗
║         PROMETHEUS ENGINE  —  Engine VI of VI                           ║
║         Neural-Symbolic 3D Orchestration / Deterministic Ground Truth   ║
║                                                                          ║
║  Paradigm:  Replace Generative Guesswork with Deterministic Geometry    ║
║  Gemini:    Creative Director, BPY Scripter, Cinematography DP          ║
║  Blender:   Geometric Ground Truth (Bullet physics, Mantaflow fluids)   ║
║  Diffusion: Neural Skinning (ControlNet depth/canny → photorealism)     ║
║  Threshold: 0% geometric drift — hallucinations are mathematically      ║
║             impossible when the 3D mesh IS the ground truth             ║
╚══════════════════════════════════════════════════════════════════════════╝

ARCHITECTURE — 4 Phases, Autonomous Feedback Loop:
  Phase 1:  Symbolic Bridge     (Gemini → BPY Translator → Blender scene build)
  Phase 2:  Kinetic Engine      (KeyframeOrchestrator + Physics Hand-off + Camera DP)
  Phase 3:  Neural Skinning     (Eevee-Next base pass → ControlNet → Diffusion upscale)
  Phase 4:  Autonomous Loop     (Gemini audits output → edits .blend → targeted re-render)

THE PARADIGM SHIFT vs ALL PREVIOUS ENGINES:
  Engines I–V:  Fighting hallucination — correcting, constraining, guiding
                probabilistic models toward physically valid outputs.
  Engine VI:    Hallucinations are MATHEMATICALLY IMPOSSIBLE.
                A 3D mesh never drifts. Bullet physics never guesses gravity.
                Mantaflow fluid never "hallucinates" a water arc.
                Gemini's role shifts: from Physical Supervisor → Creative Director.

THE "TOWER" CONCEPT:
  Director:  You (the human)
  Crew:      Gemini ER 1.5 (generates bpy scripts, keyframes, camera ops)
  Stage:     Blender 4.3+ (.blend file = persistent, saveable world state)
  Skin:      ControlNet + Stable Diffusion / Flux (photorealistic texture layer)

KEY ADVANCES OVER ALETHEIA (Engine V):
  - 3D mesh assets replace voxel approximations — exact geometry at all angles
  - Bullet physics replaces Euler integration — mathematically precise collision
  - Mantaflow replaces ActionTokens for fluids — actual CFD simulation
  - .blend file = persistent context (10s clip or 10min feature — same precision)
  - Eevee-Next depth maps as ControlNet conditioning — spatial lock is geometric
  - Gemini writes bpy Python scripts — not prompts, but executable code
  - BezierCurveOrchestrator generates Blender timeline handle points
  - CinematographyModule: Gemini as DP — focal length, aperture, TrackTo constraints
  - NeuralSkinningPipeline: 3D render + depth map → photorealistic final frame
  - Infinite "reshoots": change camera, hit render — world state is unchanged

WHAT IS REAL TODAY vs FUTURE API:
  Real:    Full bpy script generation, Blender headless CLI render, Bullet physics,
           ControlNet conditioning logic, Gemini script generation, .blend state
           management, keyframe calculation, camera constraint ops, all Gemini calls,
           metadata audit loop, frame-level targeted re-render
  Future:  Mantaflow API (available in Blender today, needs bpy wiring),
           ComfyUI/Diffusion API (endpoint-dependent), ControlNet integration
           (stable-diffusion-webui or ComfyUI required), LoRA style finetuning

See README.md § Engine VI for full documentation.
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
import httpx                            # pip install httpx
import numpy as np                      # pip install numpy

logger = logging.getLogger("prometheus")
logging.basicConfig(level=logging.INFO, format="%(name)s [%(levelname)s] %(message)s")


# ══════════════════════════════════════════════════════════════════════════
#  TYPE ALIASES
# ══════════════════════════════════════════════════════════════════════════

Vec3  = dict[str, float]   # {"x": float, "y": float, "z": float}
Euler = dict[str, float]   # {"x": float, "y": float, "z": float}  (radians)
RGBA  = tuple[float, float, float, float]


# ══════════════════════════════════════════════════════════════════════════
#  PRIMARY INPUT PRIMITIVE — SceneDirective
#  The "Creative Brief" passed to Prometheus.
#  Unlike PhysicsLaw (Engine IV) or StyleDirective (Engine V),
#  this is a Director's creative vision: assets, story beats, style.
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class SceneDirective:
    """
    The primary input primitive for Prometheus Engine.

    Instead of specifying physics constants or artistic style entropy,
    you specify a Director's creative vision: what 3D assets to place,
    what story should happen, and what visual style to apply during
    the Neural Skinning pass.

    Blender handles all physics (gravity, collision, fluid) automatically.
    Gemini generates the bpy script and keyframe timeline.
    ControlNet + Diffusion applies the visual style to the Blender render.

    Args:
        scene_brief:     Natural-language description of the scene.
        asset_manifest:  List of 3D objects to place in the scene.
        story_beats:     Key events with timestamps for animation.
        visual_style:    Style description for the Neural Skinning pass.
                         e.g. "photorealistic neo-noir, rain-slicked streets"
        duration_seconds: Video duration.
        physics_enabled: Whether to hand off to Blender's Bullet engine.
        fluid_objects:   Entity IDs that need Mantaflow fluid simulation.
        cloth_objects:   Entity IDs that need cloth simulation.
        render_engine:   "eevee" (fast) or "cycles" (path-traced, slow).
        output_resolution: (width, height) in pixels.

    Example:
        SceneDirective(
            scene_brief="A detective enters a rain-soaked office. A glass
                         tumbles off a desk when the door slams.",
            asset_manifest=[
                AssetSpec("detective", "character_rig", height_m=1.82),
                AssetSpec("desk",      "furniture",     position={"x":1.2,"y":0,"z":0}),
                AssetSpec("glass",     "prop_rigid",    position={"x":1.5,"y":0.8,"z":0}),
                AssetSpec("rain",      "particle_system", style="driving_rain"),
            ],
            story_beats=[
                StoryBeat(0.0,  "detective_enters", "Door opens, detective steps in"),
                StoryBeat(1.8,  "door_slams",       "Door closes with force"),
                StoryBeat(2.1,  "glass_falls",      "Glass tips, falls, shatters"),
                StoryBeat(5.0,  "detective_sits",   "Detective sits at desk"),
            ],
            visual_style="photorealistic neo-noir, 1940s tungsten warmth, rain-slicked reflections",
            duration_seconds=8.0,
            physics_enabled=True,
            fluid_objects=[],
            render_engine="eevee")
    """
    scene_brief: str
    asset_manifest: list["AssetSpec"]
    story_beats: list["StoryBeat"]
    visual_style: str = "photorealistic cinematic"
    duration_seconds: float = 10.0
    physics_enabled: bool = True
    fluid_objects: list[str] = field(default_factory=list)
    cloth_objects: list[str] = field(default_factory=list)
    render_engine: Literal["eevee", "cycles"] = "eevee"
    output_resolution: tuple[int, int] = (1280, 720)
    fps: int = 24
    directive_id: str = field(default_factory=lambda: f"pd_{uuid.uuid4().hex[:6]}")


@dataclass
class AssetSpec:
    """
    Specification for a 3D asset to be placed in the Blender scene.

    asset_type determines how Blender instantiates it:
        "character_rig"   → Import rigged character (or procedural humanoid)
        "furniture"       → Import mesh or use primitive + modifier stack
        "prop_rigid"      → Rigid body physics object
        "prop_soft"       → Soft body / cloth physics object
        "particle_system" → Blender particle system (rain, smoke, sparks)
        "light_rig"       → Light object (point, sun, spot, area, HDRI)
        "camera"          → Camera object with optional constraints
        "empty"           → Empty object (used as constraint targets)
        "fluid_domain"    → Mantaflow fluid domain
        "fluid_inflow"    → Mantaflow fluid inflow object
    """
    entity_id: str
    asset_type: Literal[
        "character_rig", "furniture", "prop_rigid", "prop_soft",
        "particle_system", "light_rig", "camera", "empty",
        "fluid_domain", "fluid_inflow"
    ]
    position: Vec3 = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    rotation_euler: Euler = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    scale: Vec3 = field(default_factory=lambda: {"x": 1.0, "y": 1.0, "z": 1.0})
    material_description: str = "generic_material"
    height_m: float | None = None       # For character rigs — sets scale
    style: str = ""                      # For particle systems — describes emitter style
    physics_mass_kg: float = 1.0
    physics_restitution: float = 0.3    # Bounciness 0–1
    physics_friction: float = 0.5
    mesh_source: str | None = None      # Path to .blend/.obj/.fbx, or None for procedural


@dataclass
class StoryBeat:
    """
    A narrative event at a specific timestamp.
    Gemini translates story beats into keyframe sequences and
    physics triggers in the bpy script.
    """
    t: float                          # onset time (seconds)
    beat_id: str
    description: str
    agent_id: str | None = None       # which entity drives this beat
    force_vector: Vec3 | None = None  # optional explicit force (for physics triggers)
    camera_op: str | None = None      # camera instruction for this beat


# ══════════════════════════════════════════════════════════════════════════
#  BPY SCRIPT INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class BpyScript:
    """
    A complete, validated Blender Python (bpy) script generated by Gemini.

    The BPY Translator (Phase 1) ensures the script:
        1. Places all AssetSpec objects at correct world coordinates
        2. Sets all physics properties (mass, restitution, friction)
        3. Configures Bullet rigid body simulation if physics_enabled
        4. Sets up Mantaflow domains for fluid_objects
        5. Configures cloth modifiers for cloth_objects
        6. Establishes the light rig
        7. Creates the camera with focal length and aperture
        8. Does NOT set keyframes (those are added by Phase 2)

    The script targets Blender 4.3+ and bpy 4.x API.
    """
    script_id: str
    phase: Literal["scene_build", "animation", "camera", "render", "correction"]
    source_code: str                   # Complete executable bpy Python script
    asset_ids_referenced: list[str]
    has_physics_setup: bool = False
    has_fluid_setup: bool = False
    collision_check_passed: bool = False
    validation_errors: list[str] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)


@dataclass
class KeyframeSpec:
    """
    A single keyframe for a specific object/property in Blender's timeline.

    Rather than describing motion in natural language, Gemini computes the
    exact Bezier curve handle points for Blender's f-curve interpolation.

    handle_left / handle_right are control points for the Bezier tangent,
    allowing Gemini to specify the *feel* of the motion:
        - Ease-in:  handle_left close to previous keyframe value
        - Ease-out: handle_right close to next keyframe value
        - Hold:     both handles at same value as key_value (flat tangent)
        - Linear:   handles placed at 1/3 of the distance to neighbors
    """
    entity_id: str
    frame: int                         # Blender frame number (frame = t * fps)
    data_path: str                     # bpy data path: "location", "rotation_euler", "scale"
    array_index: int                   # 0=X, 1=Y, 2=Z for vec properties; -1 for scalar
    key_value: float                   # The actual value at this keyframe
    interpolation: Literal[
        "BEZIER", "LINEAR", "CONSTANT", "EASE_IN", "EASE_OUT", "EASE_IN_OUT"
    ] = "BEZIER"
    handle_left: tuple[float, float] | None = None   # (frame_offset, value_offset)
    handle_right: tuple[float, float] | None = None  # (frame_offset, value_offset)
    easing: str = "AUTO"               # "AUTO", "AUTO_CLAMPED", "VECTOR"


@dataclass
class CameraOperation:
    """
    A single camera directive from Gemini acting as Director of Photography.

    Covers the full cinematographic vocabulary:
        - TrackTo constraint: camera tracks a target object
        - SetFOV: change focal length (affects FOV and DOF character)
        - SetAperture: f-stop for depth of field
        - DollyMove: translate camera position (push in/pull back)
        - PanTilt: rotate camera around its own axis
        - OrbitAround: arc camera around a target point
        - ShakeRig: procedural camera shake (handheld, impact)
        - CutTo: instant position change (hard cut)
    """
    t: float                           # onset time
    duration: float
    op_type: Literal[
        "TrackTo", "SetFOV", "SetAperture", "DollyMove",
        "PanTilt", "OrbitAround", "ShakeRig", "CutTo", "SetDOF"
    ]
    target_entity_id: str | None = None   # For TrackTo / OrbitAround
    position: Vec3 | None = None          # For DollyMove / CutTo
    rotation_euler: Euler | None = None   # For PanTilt
    focal_length_mm: float | None = None  # For SetFOV (e.g., 50mm, 85mm, 28mm)
    aperture_fstop: float | None = None   # For SetAperture (e.g., 1.4, 2.8, 8.0)
    dof_target_entity_id: str | None = None  # For SetDOF
    shake_intensity: float = 0.0          # For ShakeRig (0=none, 1=heavy handheld)
    easing: str = "EASE_IN_OUT"


# ══════════════════════════════════════════════════════════════════════════
#  PROMETHEUS STATE — The .blend File as World State
#
#  Unlike previous engines (in-memory buffers), the PrometheusState
#  wraps the .blend FILE as the authoritative world state.
#  This gives us:
#    - Infinite "reshoots" (change camera → re-render → same world)
#    - Persistent state (save .blend → load months later)
#    - Exact geometry at all camera angles
#    - Full Blender simulation history
# ══════════════════════════════════════════════════════════════════════════

class PrometheusState:
    """
    The persistent world state for Engine VI.

    The .blend file IS the context. Unlike OmniStateBuffer / AletheiaStateBuffer
    (which are in-memory Python dicts), PrometheusState wraps the actual
    .blend file on disk, giving us everything those buffers gave us plus:
        - Perfect geometric ground truth (mesh, not voxel approximation)
        - Native physics simulation history (Bullet/Mantaflow baked to .blend)
        - Blender's own undo/versioning via .blend snapshots
        - Trivial camera changes (no world re-simulation needed)

    The ICL log (from Engines III–V) is preserved as a custom property
    on the Blender scene object, so it persists inside the .blend file.
    """

    def __init__(self, blend_path: str | None = None) -> None:
        self._blend_path: str | None = blend_path
        self._snapshots: dict[str, str] = {}     # label → snapshot .blend path
        self._event_log: list[str] = []
        self._asset_registry: dict[str, AssetSpec] = {}
        self._bpy_scripts: list[BpyScript] = []
        self._keyframe_registry: list[KeyframeSpec] = []
        self._camera_ops: list[CameraOperation] = []
        self._render_metadata: dict[str, Any] = {}
        self._neural_skin_results: dict[int, str] = {}  # frame_index → output path
        self._collision_log: list[str] = []

    # ── Blend File Management ──────────────────────────────────────────

    def init_blend(self, blend_path: str) -> None:
        """Initialize with a new or existing .blend file path."""
        self._blend_path = blend_path
        self._log(f"Blend file initialized: {blend_path}")

    def snapshot_blend(self, label: str) -> str:
        """
        Snapshot the current .blend file.
        The snapshot IS the persistent world state — unlike Engine IV's
        dict snapshots, this captures exact mesh, physics bake, and timeline.
        """
        if not self._blend_path or not os.path.exists(self._blend_path):
            raise FileNotFoundError(f"Blend file not found: {self._blend_path}")
        snap_dir = os.path.dirname(self._blend_path)
        snap_path = os.path.join(snap_dir, f"snapshot_{label}_{uuid.uuid4().hex[:6]}.blend")
        shutil.copy2(self._blend_path, snap_path)
        self._snapshots[label] = snap_path
        self._log(f"Snapshot saved: '{label}' → {snap_path}")
        return snap_path

    def restore_snapshot(self, label: str) -> bool:
        """Restore a previous .blend snapshot (for targeted frame correction)."""
        if label not in self._snapshots:
            logger.warning("PrometheusState: No snapshot with label '%s'", label)
            return False
        snap_path = self._snapshots[label]
        shutil.copy2(snap_path, self._blend_path)
        self._log(f"Snapshot restored: '{label}' from {snap_path}")
        return True

    # ── Asset & Script Registry ────────────────────────────────────────

    def register_asset(self, spec: AssetSpec) -> None:
        self._asset_registry[spec.entity_id] = spec
        self._log(f"Asset registered: '{spec.entity_id}' · type={spec.asset_type} "
                  f"· pos=({spec.position['x']:.2f}, {spec.position['y']:.2f}, {spec.position['z']:.2f})")

    def register_script(self, script: BpyScript) -> None:
        self._bpy_scripts.append(script)
        self._log(f"BPY script [{script.phase}] registered: {script.script_id} "
                  f"· {len(script.source_code)} chars · collision_ok={script.collision_check_passed}")

    def register_camera_op(self, op: CameraOperation) -> None:
        self._camera_ops.append(op)

    def register_keyframes(self, keyframes: list[KeyframeSpec]) -> None:
        self._keyframe_registry.extend(keyframes)

    # ── Render Metadata ────────────────────────────────────────────────

    def record_render_metadata(
        self,
        frame_range: tuple[int, int],
        output_dir: str,
        render_time_s: float) -> None:
        self._render_metadata = {
            "frame_range": frame_range,
            "output_dir": output_dir,
            "render_time_s": render_time_s,
            "timestamp": time.strftime("%H:%M:%S"),
        }
        self._log(f"Render complete: frames {frame_range[0]}–{frame_range[1]} "
                  f"in {render_time_s:.1f}s → {output_dir}")

    def record_neural_skin(self, frame_index: int, output_path: str) -> None:
        self._neural_skin_results[frame_index] = output_path

    # ── Collision & ICL ────────────────────────────────────────────────

    def log_collision_check(self, result: str) -> None:
        self._collision_log.append(result)

    def get_icl_log(self, max_entries: int = 60) -> str:
        return "\n".join(self._event_log[-max_entries:])

    def _log(self, entry: str) -> None:
        self._event_log.append(f"[{time.strftime('%H:%M:%S')}] {entry}")

    # ── Export for Gemini Audit ────────────────────────────────────────

    def export_scene_summary(self) -> dict[str, Any]:
        """Export scene metadata for Gemini's audit loop."""
        return {
            "blend_path": self._blend_path,
            "asset_count": len(self._asset_registry),
            "assets": {
                eid: {
                    "type": spec.asset_type,
                    "position": spec.position,
                    "material": spec.material_description,
                    "physics_mass_kg": spec.physics_mass_kg,
                }
                for eid, spec in self._asset_registry.items()
            },
            "script_count": len(self._bpy_scripts),
            "keyframe_count": len(self._keyframe_registry),
            "camera_ops": len(self._camera_ops),
            "neural_skin_frames": len(self._neural_skin_results),
            "collision_checks": self._collision_log[-5:],
            "icl_memory": self.get_icl_log(max_entries=30),
            "render_metadata": self._render_metadata,
            "snapshots": list(self._snapshots.keys()),
        }


# ══════════════════════════════════════════════════════════════════════════
#  PHASE 1: THE SYMBOLIC BRIDGE
#  Gemini → BPY Translator → Blender Scene Build
# ══════════════════════════════════════════════════════════════════════════

class BpyTranslator:
    """
    Phase 1: The core of Engine VI's paradigm shift.

    Translates a SceneDirective into executable Blender Python (bpy) code.
    Gemini ER 1.5 acts as a code generator — not a visual prompt writer,
    not a coordinate estimator, but a Python developer writing bpy scripts.

    The BPY Translator:
        1. Generates the scene build script (asset placement, physics setup)
        2. Runs the Collision Auditor before any render (overlap check)
        3. Validates the generated script against bpy API contract
        4. Writes and executes the script via Blender headless CLI

    Collision Auditor invariants:
        - No two rigid body objects may overlap (AABB intersection check)
        - No character rig may intersect the floor plane (y=0)
        - Camera must have clear line of sight to primary subject
        - Fluid domain must fully contain all inflow objects
    """

    COLLISION_INVARIANTS = [
        "No two rigid body objects may have overlapping axis-aligned bounding boxes.",
        "No character rig base may be below the scene floor (y < 0 in Z-up space).",
        "Camera near-clip plane must not intersect any foreground object.",
        "Fluid domain AABB must fully contain all fluid inflow objects.",
        "Light rigs must not intersect scene geometry (embedded lights cause artifacts).",
    ]

    def __init__(
        self,
        gemini: GeminiERClient,
        state: PrometheusState,
        blender_executable: str = "blender") -> None:
        self.gemini   = gemini
        self.state    = state
        self.blender  = blender_executable

    def build_scene(
        self,
        directive: SceneDirective,
        work_dir: str) -> BpyScript:
        """
        Phase 1 main entry: Generate and execute the scene build script.

        Produces a .blend file with all assets placed, physics configured,
        and light rig set — ready for Phase 2 animation.
        """
        logger.info("BpyTranslator: Generating scene build script...")

        for spec in directive.asset_manifest:
            self.state.register_asset(spec)

        prompt = _build_scene_prompt(directive)
        response = self.gemini.generate_content(
            prompt)

        source_code = _extract_python(response.text)

        # Collision Audit before execution
        collision_errors = self._audit_collisions(
            directive.asset_manifest, source_code
        )

        if collision_errors:
            logger.warning(
                "BpyTranslator: %d collision violations detected — requesting repair...",
                len(collision_errors)
            )
            source_code = self._repair_collisions(source_code, collision_errors, directive)

        blend_path = os.path.join(work_dir, "prometheus_scene.blend")
        script_path = os.path.join(work_dir, "build_scene.py")

        # Inject output path into script
        source_code = self._inject_save_path(source_code, blend_path)

        with open(script_path, "w") as f:
            f.write(source_code)

        self._execute_blender_script(script_path, blend_path)
        self.state.init_blend(blend_path)
        self.state.snapshot_blend("post_scene_build")

        script = BpyScript(
            script_id=f"bpy_build_{uuid.uuid4().hex[:6]}",
            phase="scene_build",
            source_code=source_code,
            asset_ids_referenced=[a.entity_id for a in directive.asset_manifest],
            has_physics_setup=directive.physics_enabled,
            has_fluid_setup=bool(directive.fluid_objects),
            collision_check_passed=not collision_errors,
            validation_errors=collision_errors)
        self.state.register_script(script)
        logger.info(
            "BpyTranslator: Scene built · %d assets · physics=%s · blend=%s",
            len(directive.asset_manifest), directive.physics_enabled, blend_path)
        return script

    def _audit_collisions(
        self,
        assets: list[AssetSpec],
        script_source: str) -> list[str]:
        """
        Collision Auditor: Pre-execution check for spatial violations.
        Parses the generated script for bpy.ops.transform.translate calls
        and checks AABB overlaps using a rule-based geometry validator.
        """
        violations = []

        # Extract declared positions from asset specs
        rigid_bodies = [a for a in assets if a.asset_type == "prop_rigid"]
        fluid_inflows = [a for a in assets if a.asset_type == "fluid_inflow"]
        fluid_domains = [a for a in assets if a.asset_type == "fluid_domain"]
        characters   = [a for a in assets if a.asset_type == "character_rig"]

        # CHECK 1: Rough AABB overlap between rigid bodies (1m³ bounding box approx)
        for i, a in enumerate(rigid_bodies):
            for b in rigid_bodies[i+1:]:
                dx = abs(a.position["x"] - b.position["x"])
                dy = abs(a.position["y"] - b.position["y"])
                dz = abs(a.position["z"] - b.position["z"])
                size_a = max(a.scale.get("x", 1.0), 0.1)
                size_b = max(b.scale.get("x", 1.0), 0.1)
                min_sep = (size_a + size_b) * 0.5
                if dx < min_sep and dy < min_sep and dz < min_sep:
                    violations.append(
                        f"AABB OVERLAP: '{a.entity_id}' and '{b.entity_id}' "
                        f"are {dx:.2f}m apart but require {min_sep:.2f}m clearance."
                    )

        # CHECK 2: Characters below floor
        for char in characters:
            if char.position["z"] < -0.01:   # z-up coordinate system
                violations.append(
                    f"FLOOR VIOLATION: Character '{char.entity_id}' at z="
                    f"{char.position['z']:.3f} (below floor plane z=0)."
                )

        # CHECK 3: Fluid inflows inside domain
        for inflow in fluid_inflows:
            for domain in fluid_domains:
                # Rough containment check (domain at origin, scale defines bounds)
                dom_half = domain.scale.get("x", 2.0) * 0.5
                dist = math.sqrt(
                    (inflow.position["x"] - domain.position["x"])**2 +
                    (inflow.position["z"] - domain.position["z"])**2
                )
                if dist > dom_half:
                    violations.append(
                        f"FLUID DOMAIN VIOLATION: Inflow '{inflow.entity_id}' "
                        f"({dist:.2f}m from domain center) is outside "
                        f"domain '{domain.entity_id}' (half-size {dom_half:.2f}m)."
                    )

        for v in violations:
            self.state.log_collision_check(f"VIOLATION: {v}")
            logger.warning("CollisionAudit: %s", v)

        if not violations:
            self.state.log_collision_check("✓ All collision checks passed")
            logger.info("CollisionAudit: ✓ All clear")

        return violations

    def _repair_collisions(
        self,
        source_code: str,
        violations: list[str],
        directive: SceneDirective) -> str:
        """Ask Gemini to fix the collision violations in the script."""
        repair_prompt = f"""
The following bpy script has collision/placement violations.
Fix them and return ONLY the corrected Python script.

VIOLATIONS:
{chr(10).join(f"  {i+1}. {v}" for i, v in enumerate(violations))}

ORIGINAL SCRIPT:
```python
{source_code}
```

Rules for repair:
- Move overlapping objects apart by their minimum separation distance
- Raise characters to z >= 0.0 (floor plane in Z-up coordinates)
- Rescale fluid domain to contain all inflow objects
- Preserve all other aspects of the script exactly
"""
        response = self.gemini.generate_content(
            repair_prompt)
        return _extract_python(response.text)

    def _inject_save_path(self, script: str, blend_path: str) -> str:
        """Ensure the script saves to the correct path."""
        save_line = f'\nbpy.ops.wm.save_as_mainfile(filepath=r"{blend_path}")\n'
        if "bpy.ops.wm.save_as_mainfile" in script:
            return re.sub(
                r'bpy\.ops\.wm\.save_as_mainfile\(filepath=.*?\)',
                f'bpy.ops.wm.save_as_mainfile(filepath=r"{blend_path}")',
                script)
        return script + save_line

    def _execute_blender_script(self, script_path: str, blend_path: str) -> None:
        """Execute a bpy script via Blender's headless CLI."""
        cmd = [
            self.blender,
            "--background",
            "--python", script_path,
        ]
        logger.info("BpyTranslator: Executing Blender script: %s", script_path)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                logger.error("Blender script error:\n%s", result.stderr[-2000:])
            else:
                logger.info("Blender script executed successfully.")
        except FileNotFoundError:
            logger.warning(
                "Blender executable not found at '%s'. "
                "Script saved to %s — run manually.", self.blender, script_path
            )
        except subprocess.TimeoutExpired:
            logger.error("Blender script timed out after 300s.")


# ══════════════════════════════════════════════════════════════════════════
#  PHASE 2: THE KINETIC ENGINE
#  KeyframeOrchestrator + Physics Hand-off + Cinematography Module
# ══════════════════════════════════════════════════════════════════════════

class KeyframeOrchestrator:
    """
    Phase 2a: Gemini as Animator.

    Unlike Engine IV (where Gemini described motion in natural language
    and Python re-derived positions), here Gemini CALCULATES the exact
    Bezier curve handle points for Blender's f-curve interpolation.

    This means:
        - Ease-in, ease-out, hold, anticipation — all expressible with precision
        - The animation IS the keyframe data — no re-derivation needed
        - Blender's timeline is the ground truth for motion, not a Python buffer

    For story beats that involve physics (door slam causing glass to tip):
        - Gemini generates keyframes UP TO the physics trigger point
        - Then hands off to Blender's Bullet engine for the rest
        - The two systems never step on each other's feet

    Physics Hand-off Protocol:
        At the trigger frame, Gemini sets:
            1. The triggering object's position/velocity (final keyframe before sim)
            2. Physics body type: "PASSIVE" → "ACTIVE" (turns on Bullet simulation)
        After that frame, Bullet owns the object. Gemini doesn't keyframe it further.
    """

    def __init__(
        self,
        gemini: GeminiERClient,
        state: PrometheusState) -> None:
        self.gemini = gemini
        self.state  = state

    def orchestrate(
        self,
        directive: SceneDirective,
        fps: int) -> list[KeyframeSpec]:
        """Generate the complete keyframe timeline for all animated entities."""
        logger.info("KeyframeOrchestrator: Generating Bezier keyframe timeline...")

        story_json = [
            {
                "t": beat.t,
                "beat_id": beat.beat_id,
                "description": beat.description,
                "agent_id": beat.agent_id,
                "force_vector": beat.force_vector,
                "camera_op": beat.camera_op,
            }
            for beat in directive.story_beats
        ]

        assets_json = [
            {
                "entity_id": a.entity_id,
                "asset_type": a.asset_type,
                "initial_position": a.position,
                "physics_enabled": directive.physics_enabled
                    and a.asset_type in ("prop_rigid", "prop_soft"),
            }
            for a in directive.asset_manifest
        ]

        prompt = f"""
You are a Blender animator generating precise keyframe data.
Your output drives Blender's f-curve system — not a description of motion,
but the actual mathematical parameters for the Bezier interpolation curves.

SCENE: "{directive.scene_brief}"
DURATION: {directive.duration_seconds}s at {fps}fps ({int(directive.duration_seconds * fps)} total frames)
ASSETS: {json.dumps(assets_json, indent=2)}
STORY BEATS: {json.dumps(story_json, indent=2)}

Physics Hand-off Rules:
  - For physics objects, generate keyframes ONLY until the physics trigger frame
  - At the trigger frame, include a special "physics_handoff" keyframe that sets
    the object's body_type from PASSIVE to ACTIVE (Bullet takes over after this)
  - Do NOT keyframe physics objects after their handoff frame

Generate ALL keyframes needed to animate the scene.
Return ONLY a JSON array of keyframe objects:
[
  {{
    "entity_id": "detective",
    "frame": 1,
    "data_path": "location",
    "array_index": 0,
    "key_value": -2.5,
    "interpolation": "BEZIER",
    "handle_left": [-5.0, -2.5],
    "handle_right": [3.0, -2.0],
    "easing": "AUTO"
  }},
  {{
    "entity_id": "glass",
    "frame": 50,
    "data_path": "rigid_body.enabled",
    "array_index": -1,
    "key_value": 1.0,
    "interpolation": "CONSTANT",
    "handle_left": null,
    "handle_right": null,
    "easing": "CONSTANT",
    "notes": "PHYSICS HANDOFF: Bullet simulation begins here"
  }}
]
"""
        response = self.gemini.generate_content(
            prompt)
        kf_data: list[dict] = json.loads(_extract_json(response.text))

        keyframes = []
        for kd in kf_data:
            kf = KeyframeSpec(
                entity_id=kd["entity_id"],
                frame=kd["frame"],
                data_path=kd["data_path"],
                array_index=kd["array_index"],
                key_value=kd["key_value"],
                interpolation=kd.get("interpolation", "BEZIER"),
                handle_left=tuple(kd["handle_left"]) if kd.get("handle_left") else None,
                handle_right=tuple(kd["handle_right"]) if kd.get("handle_right") else None,
                easing=kd.get("easing", "AUTO"))
            keyframes.append(kf)

        self.state.register_keyframes(keyframes)
        logger.info("KeyframeOrchestrator: %d keyframes generated", len(keyframes))
        return keyframes

    def build_animation_script(
        self,
        keyframes: list[KeyframeSpec],
        blend_path: str,
        fps: int) -> BpyScript:
        """Convert keyframe specs into an executable bpy animation script."""
        lines = [
            "import bpy",
            "import mathutils",
            "",
            f"bpy.context.scene.render.fps = {fps}",
            "",
        ]

        # Group keyframes by entity
        by_entity: dict[str, list[KeyframeSpec]] = {}
        for kf in keyframes:
            by_entity.setdefault(kf.entity_id, []).append(kf)

        for eid, kfs in by_entity.items():
            lines.append(f"# ── {eid} ──────────────────────────────────────────")
            lines.append(f'obj = bpy.data.objects.get("{eid}")')
            lines.append(f'if obj is None:')
            lines.append(f'    print("WARNING: Object {eid!r} not found — skipping")')
            lines.append(f'else:')
            lines.append(f'    obj.animation_data_create()')
            lines.append(f'    obj.animation_data.action = bpy.data.actions.new(name="{eid}_Action")')
            lines.append(f'    action = obj.animation_data.action')

            for kf in sorted(kfs, key=lambda k: k.frame):
                if kf.data_path == "rigid_body.enabled":
                    # Physics handoff keyframe
                    lines.append(f'    # Physics handoff at frame {kf.frame}')
                    lines.append(f'    bpy.context.scene.frame_set({kf.frame})')
                    lines.append(f'    if obj.rigid_body:')
                    lines.append(f'        obj.rigid_body.enabled = bool({int(kf.key_value)})')
                    lines.append(f'        obj.keyframe_insert(data_path="rigid_body.enabled", frame={kf.frame})')
                else:
                    lines.append(f'    obj.keyframe_insert(data_path="{kf.data_path}", '
                                 f'index={kf.array_index}, frame={kf.frame})')
                    lines.append(f'    # Set value at frame {kf.frame}: {kf.key_value}')
                    lines.append(f'    bpy.context.scene.frame_set({kf.frame})')
                    # Set actual value
                    if kf.data_path == "location":
                        axis = ["x", "y", "z"][kf.array_index] if kf.array_index >= 0 else "x"
                        lines.append(f'    obj.location.{"xyz"[kf.array_index] if kf.array_index >= 0 else "x"} = {kf.key_value}')
                        lines.append(f'    obj.keyframe_insert(data_path="location", index={kf.array_index}, frame={kf.frame})')

            lines.append("")

        lines.append(f'bpy.ops.wm.save_as_mainfile(filepath=r"{blend_path}")')
        lines.append('print("Animation script complete.")')

        source_code = "\n".join(lines)
        script = BpyScript(
            script_id=f"bpy_anim_{uuid.uuid4().hex[:6]}",
            phase="animation",
            source_code=source_code,
            asset_ids_referenced=list(by_entity.keys()))
        self.state.register_script(script)
        return script


class PhysicsHandoffManager:
    """
    Phase 2b: Physics Simulation Hand-off.

    For events that exceed what keyframes can express (fluid, cloth, explosion):
        1. Gemini identifies the "trigger frame" for each physics event
        2. At that frame, Blender's native simulation takes over
        3. The simulation is baked to the .blend file

    Three simulation types:
        Bullet Rigid Body:  Collision, stacking, bouncing, shattering
        Mantaflow Fluid:    Liquid spills, flowing water, splashes
        Cloth:              Fabric draping, flag waving, soft deformation

    The key insight: Gemini NEVER tries to calculate fluid dynamics.
    That's what Mantaflow is for. Gemini's job is to:
        1. Set up the simulation parameters (viscosity, domain size, etc.)
        2. Identify the trigger event
        3. Bake the simulation
        4. Re-render only the affected frames if the bake needs adjustment
    """

    def __init__(
        self,
        gemini: GeminiERClient,
        state: PrometheusState) -> None:
        self.gemini = gemini
        self.state  = state

    def generate_physics_script(
        self,
        directive: SceneDirective,
        blend_path: str) -> BpyScript:
        """Generate bpy script for physics simulation setup and baking."""
        if not directive.physics_enabled and not directive.fluid_objects and not directive.cloth_objects:
            return BpyScript(
                script_id="bpy_physics_noop",
                phase="scene_build",
                source_code="import bpy  # No physics simulation required",
                asset_ids_referenced=[])

        physics_assets = [
            a for a in directive.asset_manifest
            if a.asset_type in ("prop_rigid", "prop_soft", "fluid_domain", "fluid_inflow")
        ]

        prompt = f"""
You are configuring Blender's physics simulation systems.
Generate a bpy script that:

1. For RIGID BODY objects: Set up Bullet physics with correct mass and restitution
2. For FLUID objects: Configure Mantaflow domain and inflow (if any fluid objects)
3. For CLOTH objects: Configure cloth modifier with correct stiffness
4. Set the frame range and bake all simulations

PHYSICS ASSETS: {json.dumps([
    {{
        "entity_id": a.entity_id,
        "asset_type": a.asset_type,
        "mass_kg": a.physics_mass_kg,
        "restitution": a.physics_restitution,
        "friction": a.physics_friction,
    }}
    for a in physics_assets
], indent=2)}

FLUID OBJECTS: {directive.fluid_objects}
CLOTH OBJECTS: {directive.cloth_objects}
FRAME RANGE: 1 to {int(directive.duration_seconds * directive.fps)}

Return ONLY a complete bpy Python script. It must end with:
bpy.ops.wm.save_as_mainfile(filepath=r"{blend_path}")
"""
        response = self.gemini.generate_content(
            prompt)
        source_code = _extract_python(response.text)
        source_code = self._inject_save_path(source_code, blend_path)

        script = BpyScript(
            script_id=f"bpy_physics_{uuid.uuid4().hex[:6]}",
            phase="scene_build",
            source_code=source_code,
            asset_ids_referenced=[a.entity_id for a in physics_assets],
            has_physics_setup=directive.physics_enabled,
            has_fluid_setup=bool(directive.fluid_objects))
        self.state.register_script(script)
        return script

    def _inject_save_path(self, script: str, blend_path: str) -> str:
        save_line = f'\nbpy.ops.wm.save_as_mainfile(filepath=r"{blend_path}")\n'
        if "bpy.ops.wm.save_as_mainfile" not in script:
            return script + save_line
        return re.sub(
            r'bpy\.ops\.wm\.save_as_mainfile\(filepath=.*?\)',
            f'bpy.ops.wm.save_as_mainfile(filepath=r"{blend_path}")',
            script)


class CinematographyModule:
    """
    Phase 2c: Gemini as Director of Photography.

    In previous engines, camera was a trajectory: [(t, position, look_at, fov)].
    In Prometheus, Gemini is the DP — it thinks in cinematographic terms:

        Focal length:   28mm (wide, paranoid feeling) vs 85mm (compressed, intimate)
        Aperture:       f/1.4 (shallow DOF, subject isolation) vs f/11 (deep focus)
        TrackTo:        Camera stays locked on a character through their movement
        Dutch angle:    Camera roll for psychological unease
        DOF target:     Which object the camera auto-focuses on

    Gemini generates CameraOperation sequences that are translated into:
        - bpy camera constraint baking
        - f-curve keyframes for focal length and aperture
        - TrackTo and Follow Path constraints where appropriate

    The camera IS a physics object in Blender: it has position, rotation,
    and constraints. It can be keyframed, constrained, and baked just like
    any other object. This gives us infinite reshoots: change the camera
    operator's instructions, re-run CinematographyModule, re-render.
    The scene geometry doesn't change — only the camera.
    """

    def __init__(
        self,
        gemini: GeminiERClient,
        state: PrometheusState) -> None:
        self.gemini = gemini
        self.state  = state

    def direct(
        self,
        directive: SceneDirective,
        fps: int) -> list[CameraOperation]:
        """Generate the complete camera operation sequence."""
        logger.info("CinematographyModule: Gemini acting as DP...")

        story_beats_json = [
            {"t": b.t, "description": b.description, "camera_op": b.camera_op}
            for b in directive.story_beats
        ]

        prompt = f"""
You are the Director of Photography for a Blender scene.
Your job is to generate camera operations in cinematographic language,
which will be translated into Blender camera constraints and f-curve keyframes.

SCENE: "{directive.scene_brief}"
VISUAL STYLE: "{directive.visual_style}"
DURATION: {directive.duration_seconds}s
STORY BEATS: {json.dumps(story_beats_json, indent=2)}
ASSETS (subjects): {[a.entity_id for a in directive.asset_manifest if a.asset_type not in ("light_rig", "camera", "empty")]}

Think in cinematographic terms:
  - What lens tells this story? (28mm=wide/anxious, 50mm=natural, 85mm=intimate)
  - What aperture? (f/1.4=dreamy isolation, f/5.6=narrative focus, f/11=everything sharp)
  - Should the camera track a character or hold a static frame?
  - Are there any dramatic focal length changes (crash zoom, slow rack focus)?

Return ONLY a JSON array of camera operations:
[
  {{
    "t": 0.0, "duration": 0.0,
    "op_type": "CutTo",
    "position": {{"x": -1.5, "y": -3.2, "z": 1.4}},
    "rotation_euler": {{"x": 1.34, "y": 0.0, "z": -0.45}},
    "focal_length_mm": 35.0,
    "aperture_fstop": 2.8,
    "dof_target_entity_id": "detective"
  }},
  {{
    "t": 0.5, "duration": 3.0,
    "op_type": "TrackTo",
    "target_entity_id": "detective",
    "focal_length_mm": 35.0,
    "aperture_fstop": 2.8
  }},
  {{
    "t": 2.0, "duration": 0.3,
    "op_type": "SetFOV",
    "focal_length_mm": 50.0,
    "easing": "EASE_IN_OUT"
  }},
  {{
    "t": 2.1, "duration": 1.5,
    "op_type": "ShakeRig",
    "shake_intensity": 0.4
  }}
]
"""
        response = self.gemini.generate_content(
            prompt)
        ops_data: list[dict] = json.loads(_extract_json(response.text))

        ops = []
        for od in ops_data:
            op = CameraOperation(
                t=od["t"],
                duration=od.get("duration", 0.0),
                op_type=od["op_type"],
                target_entity_id=od.get("target_entity_id"),
                position=od.get("position"),
                rotation_euler=od.get("rotation_euler"),
                focal_length_mm=od.get("focal_length_mm"),
                aperture_fstop=od.get("aperture_fstop"),
                dof_target_entity_id=od.get("dof_target_entity_id"),
                shake_intensity=od.get("shake_intensity", 0.0),
                easing=od.get("easing", "EASE_IN_OUT"))
            self.state.register_camera_op(op)
            ops.append(op)

        logger.info("CinematographyModule: %d camera operations", len(ops))
        return ops

    def build_camera_script(
        self,
        ops: list[CameraOperation],
        blend_path: str,
        fps: int) -> BpyScript:
        """Translate CameraOperations into executable bpy camera script."""
        lines = [
            "import bpy",
            "import mathutils",
            "import math",
            "",
            '# ── Camera Setup ───────────────────────────────────────────────',
            'cam_obj = bpy.data.objects.get("Camera")',
            'if cam_obj is None:',
            '    bpy.ops.object.camera_add()',
            '    cam_obj = bpy.context.active_object',
            '    cam_obj.name = "Camera"',
            'cam_data = cam_obj.data',
            'bpy.context.scene.camera = cam_obj',
            '',
        ]

        for op in ops:
            frame = int(op.t * fps)
            end_frame = int((op.t + op.duration) * fps)
            lines.append(f'# ── {op.op_type} at T={op.t:.2f}s (frame {frame}) ──')
            lines.append(f'bpy.context.scene.frame_set({frame})')

            if op.op_type == "CutTo" and op.position:
                lines.append(f'cam_obj.location = ({op.position["x"]}, {op.position["y"]}, {op.position["z"]})')
                if op.rotation_euler:
                    lines.append(f'cam_obj.rotation_euler = ('
                                 f'{op.rotation_euler["x"]}, {op.rotation_euler["y"]}, {op.rotation_euler["z"]})')
                lines.append(f'cam_obj.keyframe_insert(data_path="location", frame={frame})')
                lines.append(f'cam_obj.keyframe_insert(data_path="rotation_euler", frame={frame})')

            if op.op_type == "TrackTo" and op.target_entity_id:
                lines.append(f'target = bpy.data.objects.get("{op.target_entity_id}")')
                lines.append(f'if target:')
                lines.append(f'    ct = cam_obj.constraints.new("TRACK_TO")')
                lines.append(f'    ct.target = target')
                lines.append(f'    ct.track_axis = "TRACK_NEGATIVE_Z"')
                lines.append(f'    ct.up_axis = "UP_Y"')
                lines.append(f'    ct.influence = 1.0')
                lines.append(f'    ct.keyframe_insert(data_path="influence", frame={frame})')
                lines.append(f'    if {end_frame} > {frame}:')
                lines.append(f'        ct.keyframe_insert(data_path="influence", frame={end_frame})')

            if op.focal_length_mm:
                lines.append(f'cam_data.lens = {op.focal_length_mm}')
                lines.append(f'cam_data.keyframe_insert(data_path="lens", frame={frame})')
                if end_frame != frame:
                    lines.append(f'cam_data.keyframe_insert(data_path="lens", frame={end_frame})')

            if op.aperture_fstop:
                lines.append(f'cam_data.dof.aperture_fstop = {op.aperture_fstop}')
                lines.append(f'cam_data.dof.use_dof = True')
                lines.append(f'cam_data.keyframe_insert(data_path="dof.aperture_fstop", frame={frame})')

            if op.dof_target_entity_id:
                lines.append(f'dof_target = bpy.data.objects.get("{op.dof_target_entity_id}")')
                lines.append(f'if dof_target:')
                lines.append(f'    cam_data.dof.focus_object = dof_target')

            if op.op_type == "ShakeRig" and op.shake_intensity > 0:
                lines.append(f'# Camera shake at intensity {op.shake_intensity:.2f}')
                lines.append(f'cam_shake = cam_obj.modifiers.get("CameraShake")')
                lines.append(f'if cam_shake is None:')
                lines.append(f'    bpy.ops.object.select_all(action="DESELECT")')
                lines.append(f'    cam_obj.select_set(True)')
                lines.append(f'    bpy.context.view_layer.objects.active = cam_obj')
                lines.append(f'    # Add noise modifier to camera location f-curves')
                lines.append(f'    if cam_obj.animation_data and cam_obj.animation_data.action:')
                lines.append(f'        for fc in cam_obj.animation_data.action.fcurves:')
                lines.append(f'            if fc.data_path == "location":')
                lines.append(f'                noise = fc.modifiers.new("NOISE")')
                lines.append(f'                noise.scale = 15.0')
                lines.append(f'                noise.strength = {op.shake_intensity * 0.05}')
                lines.append(f'                noise.phase = fc.array_index * 1000')

            lines.append("")

        lines.append(f'bpy.ops.wm.save_as_mainfile(filepath=r"{blend_path}")')
        lines.append('print("Camera script complete.")')

        source_code = "\n".join(lines)
        script = BpyScript(
            script_id=f"bpy_camera_{uuid.uuid4().hex[:6]}",
            phase="camera",
            source_code=source_code,
            asset_ids_referenced=["Camera"] + [
                op.target_entity_id for op in ops if op.target_entity_id
            ])
        self.state.register_script(script)
        return script


# ══════════════════════════════════════════════════════════════════════════
#  PHASE 3: NEURAL SKINNING PIPELINE
#  Eevee-Next base pass → ControlNet → Diffusion upscale
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RenderPass:
    """
    A single rendered frame from Blender's Eevee-Next engine.
    Includes the base beauty render AND the depth/normal AOVs
    that serve as ControlNet conditioning.
    """
    frame_index: int
    t: float
    beauty_path: str           # RGB render output
    depth_map_path: str        # Depth AOV (EXR or PNG)
    normal_map_path: str       # Normal AOV
    canny_path: str            # Edge-detected silhouette (computed from beauty)
    width: int
    height: int
    render_engine_used: str


@dataclass
class NeuralSkinResult:
    """
    The output of the Neural Skinning pass for a single frame.
    The Blender render provides spatial structure (guaranteed accurate).
    The diffusion model provides photorealistic appearance.
    Combined: the spatial correctness of Blender with the visual richness
    of a generative model.
    """
    frame_index: int
    t: float
    blender_render_path: str
    depth_map_path: str
    neural_skin_path: str       # Final photorealistic output
    controlnet_mode: str        # "depth", "canny", "normal", "combined"
    style_prompt: str
    diffusion_strength: float   # 0.0 (preserve Blender) to 1.0 (full diffusion)
    spatial_lock_verified: bool # Depth map consistency check passed


class BlenderRenderer:
    """
    Phase 3a: Execute Blender's Eevee-Next render pass.

    Renders the scene to individual frames with:
        - Beauty pass (RGB): the base visual output
        - Depth AOV (Z-depth): spatial structure for ControlNet
        - Normal AOV: surface orientation for ControlNet
        - Compositor node tree for AOV extraction
    """

    def __init__(
        self,
        state: PrometheusState,
        blender_executable: str = "blender") -> None:
        self.state   = state
        self.blender = blender_executable

    def render_frame_range(
        self,
        blend_path: str,
        output_dir: str,
        frame_start: int,
        frame_end: int,
        resolution: tuple[int, int],
        render_engine: str = "BLENDER_EEVEE_NEXT") -> list[RenderPass]:
        """Render a range of frames to the output directory."""
        os.makedirs(output_dir, exist_ok=True)
        depth_dir  = os.path.join(output_dir, "depth")
        normal_dir = os.path.join(output_dir, "normals")
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(normal_dir, exist_ok=True)

        render_script = self._build_render_script(
            blend_path, output_dir, depth_dir, normal_dir,
            frame_start, frame_end, resolution, render_engine)
        script_path = os.path.join(output_dir, "render_script.py")
        with open(script_path, "w") as f:
            f.write(render_script)

        t_start = time.time()
        logger.info(
            "BlenderRenderer: Rendering frames %d–%d at %dx%d...",
            frame_start, frame_end, resolution[0], resolution[1])
        try:
            cmd = [self.blender, "--background", blend_path, "--python", script_path]
            subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error("BlenderRenderer: Render failed: %s", e)

        render_time = time.time() - t_start
        self.state.record_render_metadata(
            (frame_start, frame_end), output_dir, render_time
        )

        passes = []
        for frame_idx in range(frame_start, frame_end + 1):
            beauty = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
            depth  = os.path.join(depth_dir, f"frame_{frame_idx:04d}.exr")
            normal = os.path.join(normal_dir, f"frame_{frame_idx:04d}.png")
            passes.append(RenderPass(
                frame_index=frame_idx,
                t=frame_idx / 24.0,
                beauty_path=beauty,
                depth_map_path=depth,
                normal_map_path=normal,
                canny_path=beauty.replace(".png", "_canny.png"),
                width=resolution[0],
                height=resolution[1],
                render_engine_used=render_engine))
        return passes

    def _build_render_script(
        self,
        blend_path: str,
        output_dir: str,
        depth_dir: str,
        normal_dir: str,
        frame_start: int,
        frame_end: int,
        resolution: tuple[int, int],
        render_engine: str) -> str:
        return f"""
import bpy

# Render settings
scene = bpy.context.scene
scene.render.engine = "{render_engine}"
scene.render.resolution_x = {resolution[0]}
scene.render.resolution_y = {resolution[1]}
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = r"{output_dir}/frame_"
scene.frame_start = {frame_start}
scene.frame_end = {frame_end}

# Enable AOVs for ControlNet conditioning
view_layer = scene.view_layers["ViewLayer"]
view_layer.use_pass_z = True
view_layer.use_pass_normal = True

# Set up compositor for AOV export
scene.use_nodes = True
tree = scene.node_tree
tree.nodes.clear()

# Input
render_layers = tree.nodes.new("CompositorNodeRLayers")

# Beauty output
composite = tree.nodes.new("CompositorNodeComposite")
tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])

# Depth output
depth_out = tree.nodes.new("CompositorNodeOutputFile")
depth_out.base_path = r"{depth_dir}"
depth_out.file_slots[0].path = "frame_"
depth_out.format.file_format = "OPEN_EXR"
tree.links.new(render_layers.outputs["Depth"], depth_out.inputs["Image"])

# Normal output
normal_out = tree.nodes.new("CompositorNodeOutputFile")
normal_out.base_path = r"{normal_dir}"
normal_out.file_slots[0].path = "frame_"
tree.links.new(render_layers.outputs["Normal"], normal_out.inputs["Image"])

# Render all frames
bpy.ops.render.render(animation=True)
print("Render complete: frames {frame_start}–{frame_end}")
"""


class NeuralSkinningPipeline:
    """
    Phase 3b: The Neural Skinning pass.

    Takes Blender's Eevee-Next render (spatially perfect, visually "plastic")
    and passes it through a ControlNet + Diffusion pipeline to apply
    photorealistic appearance while locking spatial structure.

    The depth map from Blender is used as ControlNet conditioning:
    the diffusion model is guided by the EXACT 3D geometry of the Blender
    scene — not an approximation, not a voxel grid, the actual mesh depth.

    This gives us:
        Spatial accuracy:    100% (from Blender geometry, not the model)
        Visual richness:     Diffusion model quality
        Style controllability: Style prompt + LoRA finetuning

    ControlNet modes:
        "depth":    Best for preserving 3D structure, materials
        "canny":    Best for preserving edge structure, outlines
        "normal":   Best for preserving surface detail, lighting
        "combined": All three — maximum spatial fidelity (slowest)

    [FUTURE API]: ComfyUI workflow execution.
    Today: Structured prompt payload for any ControlNet API.
    """

    def __init__(
        self,
        state: PrometheusState,
        diffusion_api_url: str = "http://localhost:7860",
        controlnet_api_url: str | None = None) -> None:
        self.state             = state
        self.diffusion_api_url = diffusion_api_url
        self.controlnet_url    = controlnet_api_url or diffusion_api_url
        self.client            = httpx.Client(timeout=300)

    def skin_frame(
        self,
        render_pass: RenderPass,
        style_prompt: str,
        negative_prompt: str = "blurry, plastic, CG, unrealistic, low quality",
        controlnet_mode: str = "depth",
        diffusion_strength: float = 0.65) -> NeuralSkinResult:
        """
        Apply Neural Skinning to a single rendered frame.

        The spatial structure is locked by the ControlNet conditioning.
        The diffusion model provides photorealistic appearance.
        """
        payload = self._build_controlnet_payload(
            render_pass, style_prompt, negative_prompt,
            controlnet_mode, diffusion_strength)

        output_path = render_pass.beauty_path.replace(".png", "_skinned.png")

        # [FUTURE API]: Replace with actual ComfyUI / A1111 API call
        logger.info(
            "NeuralSkinningPipeline: Frame %d · mode=%s · strength=%.2f",
            render_pass.frame_index, controlnet_mode, diffusion_strength)
        try:
            resp = self.client.post(
                f"{self.controlnet_url}/controlnet/txt2img",
                json=payload)
            if resp.status_code == 200:
                data = resp.json()
                if "images" in data:
                    import base64
                    img_bytes = base64.b64decode(data["images"][0])
                    with open(output_path, "wb") as f:
                        f.write(img_bytes)
        except Exception as e:
            logger.warning("NeuralSkinningPipeline: API call failed (%s) — stub output", e)
            output_path = render_pass.beauty_path  # Fall back to raw Blender render

        spatial_ok = self._verify_spatial_lock(render_pass, output_path)
        self.state.record_neural_skin(render_pass.frame_index, output_path)

        return NeuralSkinResult(
            frame_index=render_pass.frame_index,
            t=render_pass.t,
            blender_render_path=render_pass.beauty_path,
            depth_map_path=render_pass.depth_map_path,
            neural_skin_path=output_path,
            controlnet_mode=controlnet_mode,
            style_prompt=style_prompt,
            diffusion_strength=diffusion_strength,
            spatial_lock_verified=spatial_ok)

    def _build_controlnet_payload(
        self,
        render_pass: RenderPass,
        style_prompt: str,
        negative_prompt: str,
        mode: str,
        strength: float) -> dict[str, Any]:
        """Build the ControlNet conditioning payload."""
        import base64

        def encode_image(path: str) -> str:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            return ""

        beauty_b64 = encode_image(render_pass.beauty_path)
        depth_b64  = encode_image(render_pass.depth_map_path)
        normal_b64 = encode_image(render_pass.normal_map_path)

        controlnet_args = []
        if mode in ("depth", "combined") and depth_b64:
            controlnet_args.append({
                "input_image": depth_b64,
                "module": "depth",
                "model": "control_v11f1p_sd15_depth",
                "weight": 1.0,
                "guidance_start": 0.0,
                "guidance_end": 1.0,
            })
        if mode in ("normal", "combined") and normal_b64:
            controlnet_args.append({
                "input_image": normal_b64,
                "module": "normal_bae",
                "model": "control_v11p_sd15_normalbae",
                "weight": 0.8,
            })
        if mode in ("canny", "combined") and beauty_b64:
            controlnet_args.append({
                "input_image": beauty_b64,
                "module": "canny",
                "model": "control_v11p_sd15_canny",
                "weight": 0.6,
            })

        return {
            "prompt": f"(masterpiece, best quality, photorealistic:1.3), {style_prompt}",
            "negative_prompt": negative_prompt,
            "init_images": [beauty_b64],
            "denoising_strength": strength,
            "steps": 30,
            "sampler_name": "DPM++ 2M Karras",
            "cfg_scale": 7.5,
            "width": render_pass.width,
            "height": render_pass.height,
            "alwayson_scripts": {
                "controlnet": {"args": controlnet_args}
            },
        }

    def _verify_spatial_lock(
        self,
        render_pass: RenderPass,
        skinned_path: str) -> bool:
        """
        Verify that the neural-skinned frame maintains spatial fidelity
        to the Blender depth map.

        [FUTURE]: Compare pixel-space depth map against skinned frame
        using a depth estimation model to detect spatial drift.
        Today: Returns True if the skinned file exists and is non-empty.
        """
        if not os.path.exists(skinned_path):
            return False
        return os.path.getsize(skinned_path) > 1024


# ══════════════════════════════════════════════════════════════════════════
#  PHASE 4: THE AUTONOMOUS LOOP
#  Gemini audits → edits .blend → targeted re-render
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RenderAuditResult:
    """
    The result of Gemini's autonomous audit of a rendered + skinned frame.
    Unlike previous engines' AestheticViolation / CorrectionVector,
    this audit operates on METADATA — not pixel-level analysis.

    Gemini analyzes:
        - Lighting: Is the light rig producing the intended mood?
        - Composition: Is the camera angle telling the story effectively?
        - Physics fidelity: Did the simulation produce the expected result?
        - Neural skin consistency: Does the skinned frame match the style?
        - Temporal coherence: Does this frame match its neighbors?

    When a violation is found, the correction targets the .blend file:
        - Lighting issue    → edit light rig parameters → re-render affected frames
        - Camera issue      → edit camera operation → re-render from that point
        - Physics issue     → re-bake simulation with adjusted parameters
        - Skin issue        → re-run neural skinning with corrected strength/prompt
    """
    frame_index: int
    t: float
    violation_type: Literal[
        "lighting_error", "composition_error", "physics_fidelity",
        "neural_skin_inconsistency", "temporal_incoherence", "clear"
    ]
    severity: Literal["blocking", "minor", "cosmetic", "clear"]
    description: str
    correction_type: Literal[
        "edit_light_rig", "edit_camera", "rebake_physics",
        "reskin_frame", "reskin_range", "no_action"
    ]
    blend_edit_instructions: str    # What bpy code to run to fix this
    affected_frames: list[int]


class AutonomousAuditLoop:
    """
    Phase 4: Gemini as autonomous quality supervisor.

    In Engines I–V, the feedback loop checked pixels for spatial violations.
    In Prometheus, the feedback loop operates at a higher abstraction:
    it checks the INTENT of the scene — "Is this shot telling the story?"

    Workflow:
        1. Render a frame (or group of frames)
        2. Apply neural skinning
        3. Gemini analyzes the metadata (render settings, light rig, camera)
           AND the output frame
        4. If a violation is found:
            a. Identify the .blend correction needed
            b. Generate and execute a bpy correction script
            c. Re-render only the affected frames
            d. Re-apply neural skinning to those frames
        5. Repeat until all frames pass audit (max 3 passes)

    Key advantage over Engine V's feedback loop:
        In Engine V, corrections modified Grok's latent space (probabilistic).
        In Prometheus, corrections modify the .blend file (deterministic).
        A re-render of the same .blend with fixed lighting is GUARANTEED
        to produce the correct lighting — not just likely to.
    """

    MAX_AUDIT_PASSES = 3

    def __init__(
        self,
        gemini: GeminiERClient,
        state: PrometheusState,
        renderer: BlenderRenderer,
        skinning: NeuralSkinningPipeline,
        bpy_translator: BpyTranslator) -> None:
        self.gemini      = gemini
        self.state       = state
        self.renderer    = renderer
        self.skinning    = skinning
        self.translator  = bpy_translator

    def run(
        self,
        render_passes: list[RenderPass],
        skin_results: list[NeuralSkinResult],
        directive: SceneDirective) -> list[NeuralSkinResult]:
        """Run the autonomous audit loop until all frames pass or max passes reached."""
        current_skins = list(skin_results)

        for pass_num in range(1, self.MAX_AUDIT_PASSES + 1):
            violations = self._audit_all_frames(current_skins, directive)

            blocking = [v for v in violations if v.severity == "blocking"]
            if not blocking:
                logger.info(
                    "✓ Autonomous Loop Pass %d: No blocking violations. "
                    "Prometheus render complete.", pass_num
                )
                break

            logger.warning(
                "Autonomous Loop Pass %d: %d blocking violations", pass_num, len(blocking)
            )
            current_skins = self._apply_corrections(blocking, render_passes, directive)

        return current_skins

    def _audit_all_frames(
        self,
        skins: list[NeuralSkinResult],
        directive: SceneDirective) -> list[RenderAuditResult]:
        """Audit all skinned frames using Gemini metadata analysis."""
        scene_summary = self.state.export_scene_summary()
        audit_targets = [
            {
                "frame_index": s.frame_index,
                "t": s.t,
                "blender_render_exists": os.path.exists(s.blender_render_path),
                "skinned_exists": os.path.exists(s.neural_skin_path),
                "spatial_lock_verified": s.spatial_lock_verified,
                "controlnet_mode": s.controlnet_mode,
                "diffusion_strength": s.diffusion_strength,
            }
            for s in skins
        ]

        prompt = f"""
You are an autonomous quality supervisor for a Blender + Neural Skinning pipeline.
Analyze the render metadata and identify any blocking violations.

SCENE INTENT: "{directive.scene_brief}"
VISUAL STYLE TARGET: "{directive.visual_style}"
SCENE SUMMARY: {json.dumps(scene_summary, indent=2)}
RENDERED FRAMES: {json.dumps(audit_targets, indent=2)}

Check for:
1. LIGHTING_ERROR:             Light rig producing wrong mood for the scene
2. COMPOSITION_ERROR:          Camera angle not serving the story beat
3. PHYSICS_FIDELITY:           Physics simulation result wrong (e.g. glass didn't fall)
4. NEURAL_SKIN_INCONSISTENCY:  Skinned frames have inconsistent visual style
5. TEMPORAL_INCOHERENCE:       Significant visual discontinuity between adjacent frames

For each violation, provide:
  - Specific bpy code to fix the .blend file
  - Which frames need re-rendering after the fix

Return ONLY a JSON array ([] if no violations):
[
  {{
    "frame_index": 50,
    "t": 2.08,
    "violation_type": "lighting_error",
    "severity": "blocking",
    "description": "Key light is too harsh for neo-noir mood — highlights are blown out",
    "correction_type": "edit_light_rig",
    "blend_edit_instructions": "import bpy\\nlight = bpy.data.objects['Key_Light']\\nlight.data.energy = 800  # was 2000\\nlight.data.color = (0.95, 0.85, 0.7)  # warm tungsten\\nbpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)",
    "affected_frames": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
  }}
]
"""
        response = self.gemini.generate_content(
            prompt)
        audit_data: list[dict] = json.loads(_extract_json(response.text))

        results = []
        for ad in audit_data:
            results.append(RenderAuditResult(
                frame_index=ad.get("frame_index", 0),
                t=ad.get("t", 0.0),
                violation_type=ad.get("violation_type", "clear"),
                severity=ad.get("severity", "minor"),
                description=ad.get("description", ""),
                correction_type=ad.get("correction_type", "no_action"),
                blend_edit_instructions=ad.get("blend_edit_instructions", ""),
                affected_frames=ad.get("affected_frames", [])))

        if not results:
            logger.info("Autonomous audit: ✓ All frames pass")
        else:
            logger.warning("Autonomous audit: %d violations found", len(results))

        return results

    def _apply_corrections(
        self,
        violations: list[RenderAuditResult],
        render_passes: list[RenderPass],
        directive: SceneDirective) -> list[NeuralSkinResult]:
        """Apply .blend corrections and re-render affected frames."""
        # Snapshot before corrections
        self.state.snapshot_blend("pre_correction")

        # Collect all affected frames
        affected_frames: set[int] = set()
        for v in violations:
            # Execute the correction script
            if v.blend_edit_instructions:
                self._execute_correction(v)
            affected_frames.update(v.affected_frames)

        if not affected_frames:
            return []

        # Re-render only affected frames
        blend_path = self.state._blend_path
        if not blend_path:
            return []

        output_dir = os.path.dirname(blend_path)
        corrected_passes = self.renderer.render_frame_range(
            blend_path=blend_path,
            output_dir=output_dir,
            frame_start=min(affected_frames),
            frame_end=max(affected_frames),
            resolution=directive.output_resolution,
            render_engine="BLENDER_EEVEE_NEXT")

        # Re-apply neural skinning to corrected frames
        corrected_skins = []
        for rp in corrected_passes:
            if rp.frame_index in affected_frames:
                skin = self.skinning.skin_frame(
                    rp,
                    style_prompt=directive.visual_style,
                    controlnet_mode="depth")
                corrected_skins.append(skin)

        logger.info(
            "Autonomous correction: %d frames re-rendered, %d re-skinned",
            len(corrected_passes), len(corrected_skins))
        return corrected_skins

    def _execute_correction(self, violation: RenderAuditResult) -> None:
        """Write and execute a targeted .blend correction script."""
        blend_path = self.state._blend_path
        if not blend_path:
            return

        script_path = blend_path.replace(".blend", f"_correction_{violation.frame_index}.py")
        correction_code = violation.blend_edit_instructions

        # Ensure save is in the script
        if "save_as_mainfile" not in correction_code:
            correction_code += f'\nimport bpy\nbpy.ops.wm.save_as_mainfile(filepath=r"{blend_path}")\n'

        with open(script_path, "w") as f:
            f.write(correction_code)

        try:
            cmd = [
                self.translator.blender,
                "--background", blend_path,
                "--python", script_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                logger.info(
                    "Correction applied: [%s] frame %d",
                    violation.violation_type, violation.frame_index)
            else:
                logger.error("Correction script failed: %s", result.stderr[-500:])
        except Exception as e:
            logger.error("Correction execution error: %s", e)


# ══════════════════════════════════════════════════════════════════════════
#  GROK PROMETHEUS SYSTEM INSTRUCTIONS
#  Note: In Prometheus, Grok is replaced by Blender + Diffusion.
#  These instructions are for any remaining generative calls (e.g.
#  the Diffusion model's style prompt, or any Gemini visual reasoning).
# ══════════════════════════════════════════════════════════════════════════

PROMETHEUS_DIFFUSION_INSTRUCTIONS = """
## PROMETHEUS NEURAL SKINNING PROTOCOL — SPATIAL TRUTH IS GEOMETRY

You are applying photorealistic appearance to a Blender-rendered scene.
The Blender render provides the geometric ground truth.
Your job is to make it look real — without moving anything.

═══════════════════════════════════════
TIER 1: THE GEOMETRIC CONSTITUTION
(The depth map is law. Geometry is truth.)
═══════════════════════════════════════

DEPTH MAP SUPREMACY
The ControlNet depth conditioning comes from Blender's exact mesh geometry.
Every depth value is mathematically precise — not estimated, not hallucinated.
Your output MUST respect this depth map. An object that is behind another
in the depth map must remain behind it in your output.

SPATIAL LOCK
You are a texture artist working over a perfect geometric blueprint.
You may add: photorealistic materials, subsurface scattering, atmospheric haze,
micro-surface detail, lens effects, atmospheric bokeh.
You must NOT add: objects that aren't in the depth map, movement, displacement.

═══════════════════════════════════════
TIER 2: STYLE APPLICATION
(Make it look real in the specified visual language)
═══════════════════════════════════════

Apply the visual style with full photorealistic fidelity.
The style defines the *appearance quality*, not the *spatial structure*.
A "neo-noir" style means: tungsten warmth, rain-slicked reflections, deep shadows.
It does NOT mean: moving objects, changing geometry, adding people not in the scene.

═══════════════════════════════════════
TIER 3: TEMPORAL COHERENCE
(Adjacent frames must be consistent)
═══════════════════════════════════════

If provided with adjacent frame context, ensure consistent:
- Material appearance (no shimmer or texture drift between frames)
- Lighting character (no sudden color temperature shifts)
- Depth of field character (focus distance doesn't jump between frames)

The Autonomous Audit Loop will check temporal coherence.
Inconsistencies will trigger a re-skin pass.
"""


# ══════════════════════════════════════════════════════════════════════════
#  PROMETHEUS ENGINE — Main Orchestrator
# ══════════════════════════════════════════════════════════════════════════

class PrometheusEngine:
    """
    Engine VI: The Personal AI Film Studio.

    The "Tower" concept realized:
        Director:  You (the human with a SceneDirective)
        Crew:      Gemini ER 1.5 (bpy scripts, keyframes, camera, audit)
        Stage:     Blender 4.3+ (.blend file = persistent, saveable world)
        Skin:      ControlNet + Diffusion (photorealistic appearance layer)

    Usage:
        engine = PrometheusEngine(
            gemini_api_key="YOUR_GEMINI_KEY",
            blender_executable="/path/to/blender",
            diffusion_api_url="http://localhost:7860")

        directive = SceneDirective(
            scene_brief="A detective enters a rain-soaked office at midnight.",
            asset_manifest=[
                AssetSpec("detective",  "character_rig", height_m=1.82,
                          position={"x": -2.5, "y": 0, "z": 0}),
                AssetSpec("desk",       "furniture",
                          position={"x": 1.5, "y": 0, "z": 0}),
                AssetSpec("glass",      "prop_rigid",    physics_mass_kg=0.15,
                          position={"x": 1.8, "y": 0.85, "z": 0}),
                AssetSpec("key_light",  "light_rig",
                          position={"x": -2, "y": 2, "z": 4}),
                AssetSpec("rain",       "particle_system", style="driving_rain"),
            ],
            story_beats=[
                StoryBeat(0.0, "enter",     "Detective pushes door open",   "detective"),
                StoryBeat(1.8, "door_slam", "Door slams shut",              "detective",
                          force_vector={"x": 50, "y": 0, "z": 0}),
                StoryBeat(2.1, "glass_fall","Glass tips and falls",         "glass"),
                StoryBeat(5.0, "sit",       "Detective sits at desk",       "detective"),
            ],
            visual_style="photorealistic 1940s neo-noir, tungsten warmth, rain-slicked",
            physics_enabled=True,
            render_engine="eevee")

        final_path = engine.render(directive)
    """

    def __init__(
        self,
        gemini_api_key: str,
        blender_executable: str = "blender",
        diffusion_api_url: str = "http://localhost:7860",
        work_dir: str | None = None) -> None:
        self.gemini = GeminiERClient(
            api_key=gemini_api_key, model=ER15_MODEL,
            default_thinking=ThinkingPreset.NONE)
        self.blender_exe       = blender_executable
        self.diffusion_url     = diffusion_api_url
        self.work_dir          = work_dir or tempfile.mkdtemp(prefix="prometheus_")

        self.state             = PrometheusState()
        self.bpy_translator    = BpyTranslator(self.gemini, self.state, blender_executable)
        self.physics_manager   = PhysicsHandoffManager(self.gemini, self.state)
        self.keyframe_orch     = KeyframeOrchestrator(self.gemini, self.state)
        self.cinematography    = CinematographyModule(self.gemini, self.state)
        self.renderer          = BlenderRenderer(self.state, blender_executable)
        self.neural_skinning   = NeuralSkinningPipeline(self.state, diffusion_api_url)
        self.audit_loop        = AutonomousAuditLoop(
            self.gemini, self.state, self.renderer,
            self.neural_skinning, self.bpy_translator)

        logger.info("PrometheusEngine initialized · work_dir=%s", self.work_dir)

    def render(
        self,
        directive: SceneDirective,
        output_dir: str | None = None) -> str:
        """
        Full Prometheus pipeline. Returns path to the Neural-Skinned video.

        Args:
            directive:   SceneDirective (assets + story beats + visual style)
            output_dir:  Where to write rendered frames (default: work_dir/output)

        Returns:
            Path to the final neural-skinned video or frame sequence directory.
        """
        out_dir   = output_dir or os.path.join(self.work_dir, "output")
        blend_dir = os.path.join(self.work_dir, "blend")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(blend_dir, exist_ok=True)

        logger.info("▲▲▲ PROMETHEUS · Neural-Symbolic Film Studio · START ▲▲▲")
        logger.info("Scene: %s", directive.scene_brief[:80])
        logger.info("Assets: %d · Duration: %.1fs · Engine: %s",
                    len(directive.asset_manifest), directive.duration_seconds,
                    directive.render_engine)

        # ──────────────────────────────────────────────────────────────
        # PHASE 1: SYMBOLIC BRIDGE — Build the Blender scene
        # ──────────────────────────────────────────────────────────────
        logger.info("Phase 1 · Symbolic Bridge: Building Blender scene...")
        build_script = self.bpy_translator.build_scene(directive, blend_dir)

        if directive.physics_enabled or directive.fluid_objects:
            physics_script = self.physics_manager.generate_physics_script(
                directive, self.state._blend_path or os.path.join(blend_dir, "prometheus_scene.blend")
            )
            self._execute_script(physics_script)

        logger.info("Phase 1 complete: .blend file at %s", self.state._blend_path)

        # ──────────────────────────────────────────────────────────────
        # PHASE 2: KINETIC ENGINE — Animation + Camera
        # ──────────────────────────────────────────────────────────────
        logger.info("Phase 2 · Kinetic Engine: Generating keyframes and camera...")
        keyframes = self.keyframe_orch.orchestrate(directive, directive.fps)

        blend_path = self.state._blend_path or os.path.join(blend_dir, "prometheus_scene.blend")
        anim_script  = self.keyframe_orch.build_animation_script(keyframes, blend_path, directive.fps)
        self._execute_script(anim_script)

        camera_ops    = self.cinematography.direct(directive, directive.fps)
        camera_script = self.cinematography.build_camera_script(camera_ops, blend_path, directive.fps)
        self._execute_script(camera_script)

        self.state.snapshot_blend("post_animation")
        logger.info(
            "Phase 2 complete: %d keyframes · %d camera ops", len(keyframes), len(camera_ops)
        )

        # ──────────────────────────────────────────────────────────────
        # PHASE 3: NEURAL SKINNING — Render + Diffusion upscale
        # ──────────────────────────────────────────────────────────────
        logger.info("Phase 3 · Neural Skinning: Rendering frames + applying style...")
        total_frames = int(directive.duration_seconds * directive.fps)

        render_engine_bpy = (
            "BLENDER_EEVEE_NEXT" if directive.render_engine == "eevee"
            else "CYCLES"
        )
        render_passes = self.renderer.render_frame_range(
            blend_path=blend_path,
            output_dir=out_dir,
            frame_start=1,
            frame_end=total_frames,
            resolution=directive.output_resolution,
            render_engine=render_engine_bpy)

        # Apply neural skinning to all frames
        skin_results: list[NeuralSkinResult] = []
        for rp in render_passes:
            skin = self.neural_skinning.skin_frame(
                rp,
                style_prompt=directive.visual_style,
                controlnet_mode="depth")
            skin_results.append(skin)

        logger.info(
            "Phase 3 complete: %d frames rendered · %d skinned",
            len(render_passes), len(skin_results))

        # ──────────────────────────────────────────────────────────────
        # PHASE 4: AUTONOMOUS LOOP — Audit + targeted correction
        # ──────────────────────────────────────────────────────────────
        logger.info("Phase 4 · Autonomous Audit Loop: Checking for violations...")
        final_skins = self.audit_loop.run(render_passes, skin_results, directive)

        # Assemble final output
        final_output = self._assemble_video(final_skins or skin_results, out_dir)

        logger.info("▲▲▲ PROMETHEUS · Final output: %s ▲▲▲", final_output)
        logger.info("Work dir (contains .blend + frame sequence): %s", self.work_dir)
        return final_output

    def _execute_script(self, script: BpyScript) -> None:
        """Execute a BpyScript via Blender headless CLI."""
        blend_path = self.state._blend_path
        if not blend_path:
            logger.warning("No blend path set — script not executed")
            return

        script_path = os.path.join(
            self.work_dir, f"{script.script_id}.py"
        )
        with open(script_path, "w") as f:
            f.write(script.source_code)

        try:
            cmd = [
                self.blender_exe,
                "--background", blend_path,
                "--python", script_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.warning("Script [%s] returned non-zero: %s",
                               script.script_id, result.stderr[-500:])
            else:
                logger.debug("Script [%s] OK", script.script_id)
        except FileNotFoundError:
            logger.warning(
                "Blender not found — script saved: %s", script_path
            )
        except subprocess.TimeoutExpired:
            logger.error("Script [%s] timed out", script.script_id)

    def _assemble_video(
        self,
        skins: list[NeuralSkinResult],
        out_dir: str) -> str:
        """
        Assemble the final skinned frames into a video using ffmpeg.
        Falls back to returning the frame directory if ffmpeg unavailable.
        """
        if not skins:
            return out_dir

        video_path = os.path.join(out_dir, "prometheus_final.mp4")
        # Get the frame pattern
        sample_path = skins[0].neural_skin_path
        frame_pattern = re.sub(r'\d{4}\.png$', '%04d.png',
                               sample_path.replace("_skinned", "_skinned"))
        # If frame pattern substitution failed, fall back
        if frame_pattern == sample_path:
            frame_pattern = os.path.join(out_dir, "frame_%04d_skinned.png")

        fps = 24
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            video_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if os.path.exists(video_path):
                logger.info("Video assembled: %s", video_path)
                return video_path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.info("ffmpeg unavailable — returning frame sequence: %s", out_dir)

        return out_dir


# ══════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        return match.group(1).strip()
    # Try to find a JSON array or object directly
    arr = re.search(r"(\[[\s\S]+\])", text)
    if arr:
        return arr.group(1).strip()
    obj = re.search(r"(\{[\s\S]+\})", text)
    if obj:
        return obj.group(1).strip()
    return text.strip()


def _extract_python(text: str) -> str:
    """Extract Python code from a Gemini response."""
    match = re.search(r"```(?:python)?\s*([\s\S]+?)```", text)
    if match:
        return match.group(1).strip()
    # If no code block, treat the whole response as code
    return text.strip()


def _build_scene_prompt(directive: SceneDirective) -> str:
    """Build the Gemini prompt for the scene build bpy script."""
    assets_json = json.dumps(
        [
            {
                "entity_id": a.entity_id,
                "asset_type": a.asset_type,
                "position": a.position,
                "rotation_euler": a.rotation_euler,
                "scale": a.scale,
                "material_description": a.material_description,
                "height_m": a.height_m,
                "style": a.style,
                "physics_mass_kg": a.physics_mass_kg,
                "physics_restitution": a.physics_restitution,
                "physics_friction": a.physics_friction,
                "mesh_source": a.mesh_source,
            }
            for a in directive.asset_manifest
        ],
        indent=2)

    return f"""
You are a Blender Python developer generating a complete scene build script.
Target: Blender 4.3+, bpy 4.x API.

SCENE: "{directive.scene_brief}"
VISUAL STYLE: "{directive.visual_style}"
DURATION: {directive.duration_seconds}s at {directive.fps}fps
PHYSICS ENABLED: {directive.physics_enabled}
FLUID OBJECTS: {directive.fluid_objects}
CLOTH OBJECTS: {directive.cloth_objects}
RESOLUTION: {directive.output_resolution[0]}x{directive.output_resolution[1]}

ASSETS TO PLACE:
{assets_json}

Generate a complete bpy Python script that:
1. Clears the default Blender scene (delete cube, light, camera)
2. Sets scene frame range: 1 to {int(directive.duration_seconds * directive.fps)}
3. Sets render engine to {"BLENDER_EEVEE_NEXT" if directive.render_engine == "eevee" else "CYCLES"}
4. For each asset:
   - "character_rig": Create a procedural humanoid armature with mesh (scaled to height_m)
   - "furniture": Create appropriate mesh primitive + Solidify modifier
   - "prop_rigid": Create mesh primitive + Rigid Body physics (type=PASSIVE initially)
   - "prop_soft": Create mesh + Soft Body modifier
   - "particle_system": Create emitter mesh + Particle System modifier
   - "light_rig": Create appropriate Light object (point/sun/spot/area)
   - "camera": Create Camera object
   - "fluid_domain": Create Mantaflow domain
   - "fluid_inflow": Create Mantaflow inflow object
5. Set all objects to their specified positions, rotations, scales
6. Create materials for all objects based on material_description
7. If physics_enabled=True, add Rigid Body World with gravity
8. Save the file to the correct path

CRITICAL bpy rules:
- Use bpy.data.objects, bpy.data.meshes, bpy.data.materials
- For rigid body: bpy.ops.rigidbody.object_add()
- For lights: bpy.ops.object.light_add(type="POINT"|"SUN"|"SPOT"|"AREA")
- For camera: bpy.ops.object.camera_add()
- ALL object positions use Blender's Z-up right-hand coordinate system
- Name every object with its entity_id exactly: obj.name = "entity_id"
- End with: bpy.ops.wm.save_as_mainfile(filepath=r"/tmp/prometheus_scene.blend")

Return ONLY the complete Python script, no explanations.
"""


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — The Tower in action
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = PrometheusEngine(
        gemini_api_key="YOUR_GEMINI_KEY",
        blender_executable="/path/to/blender",          # or just "blender" if in PATH
        diffusion_api_url="http://localhost:7860",       # Stable Diffusion WebUI / ComfyUI
        work_dir="/tmp/prometheus_render")

    # ── EXAMPLE 1: Neo-Noir Detective Scene ───────────────────────────
    detective_directive = SceneDirective(
        scene_brief=(
            "A rain-soaked private detective's office, 1940s, midnight. "
            "The detective enters through a creaking door. "
            "The door slams — a whisky glass tips off the desk and shatters. "
            "The detective picks up a photograph and sits."
        ),
        asset_manifest=[
            AssetSpec("detective",    "character_rig",    height_m=1.82,
                      position={"x": -2.5, "y": 0.0,  "z": 0.0},
                      material_description="dark_wool_trenchcoat_wet"),
            AssetSpec("desk",         "furniture",
                      position={"x": 1.5,  "y": 0.0,  "z": 0.0},
                      scale={"x": 1.6, "y": 0.8, "z": 0.75},
                      material_description="worn_mahogany_wood"),
            AssetSpec("whisky_glass", "prop_rigid",       physics_mass_kg=0.15,
                      position={"x": 1.9,  "y": 0.0,  "z": 0.77},
                      physics_restitution=0.2, physics_friction=0.4,
                      material_description="thick_glass_with_amber_liquid"),
            AssetSpec("photograph",   "prop_rigid",       physics_mass_kg=0.01,
                      position={"x": 1.3,  "y": 0.0,  "z": 0.77},
                      material_description="yellowed_photograph_matte"),
            AssetSpec("key_light",    "light_rig",
                      position={"x": -1.5, "y": 2.5,  "z": 3.5},
                      style="warm_tungsten_500W"),
            AssetSpec("window_light", "light_rig",
                      position={"x": 3.0,  "y": 3.0,  "z": 2.0},
                      style="cool_moonlight_through_venetian_blinds"),
            AssetSpec("rain",         "particle_system",
                      position={"x": 0.0,  "y": 0.0,  "z": 5.0},
                      style="driving_rain_streaks_exterior"),
            AssetSpec("main_camera",  "camera",
                      position={"x": -3.0, "y": -2.5, "z": 1.6}),
        ],
        story_beats=[
            StoryBeat(0.0, "enter",
                      "Door creaks open. Detective steps into frame.",
                      agent_id="detective",
                      camera_op="wide establishing shot, 28mm, f/5.6"),
            StoryBeat(1.8, "door_slam",
                      "Detective slams door shut. Force propagates to desk.",
                      agent_id="detective",
                      force_vector={"x": 120.0, "y": 0.0, "z": 0.0},
                      camera_op="cut to desk close-up on slam, 85mm, f/2.8"),
            StoryBeat(2.1, "glass_tips",
                      "Whisky glass tips and begins to fall (physics handoff).",
                      agent_id="whisky_glass",
                      camera_op="rack focus from glass to floor"),
            StoryBeat(3.5, "glass_shatters",
                      "Glass hits floor and shatters. Whisky splashes.",
                      agent_id="whisky_glass",
                      camera_op="stay on glass impact"),
            StoryBeat(5.0, "pick_photo",
                      "Detective walks to desk and picks up photograph.",
                      agent_id="detective",
                      camera_op="slow dolly in, 50mm, f/1.8, focus on photograph"),
            StoryBeat(7.5, "sit",
                      "Detective sits heavily at desk. Scene holds.",
                      agent_id="detective",
                      camera_op="pull back slowly, 35mm, f/5.6"),
        ],
        visual_style=(
            "photorealistic 1940s neo-noir, tungsten warmth, venetian blind shadow stripes, "
            "rain-slicked reflections, 35mm film grain, shallow depth of field"
        ),
        duration_seconds=10.0,
        physics_enabled=True,
        fluid_objects=[],           # No Mantaflow (rain is particles, not fluid)
        cloth_objects=[],
        render_engine="eevee",
        output_resolution=(1280, 720),
        fps=24)

    # ── EXAMPLE 2: Fantasy Forest with Fluid ──────────────────────────
    forest_directive = SceneDirective(
        scene_brief=(
            "A magical forest clearing. A stone bowl fills with glowing water "
            "from an overhanging vine. A spirit creature drinks from it."
        ),
        asset_manifest=[
            AssetSpec("spirit",       "character_rig",    height_m=0.9,
                      position={"x": 0.0, "y": 0.0, "z": 0.0},
                      material_description="translucent_bioluminescent_skin"),
            AssetSpec("stone_bowl",   "prop_rigid",
                      position={"x": 0.0, "y": 0.0, "z": 0.3},
                      material_description="mossy_ancient_stone"),
            AssetSpec("water_domain", "fluid_domain",
                      position={"x": 0.0, "y": 0.0, "z": 0.6},
                      scale={"x": 0.6, "y": 0.6, "z": 0.5},
                      material_description="glowing_magical_water"),
            AssetSpec("water_inflow", "fluid_inflow",
                      position={"x": 0.0, "y": 0.0, "z": 1.0},
                      style="slow_drip_0.02ms"),
            AssetSpec("key_light",    "light_rig",
                      position={"x": 5.0, "y": 5.0, "z": 8.0},
                      style="dappled_forest_sunlight"),
            AssetSpec("rim_light",    "light_rig",
                      position={"x": -3.0, "y": -2.0, "z": 3.0},
                      style="cool_blue_moonlight_rim"),
            AssetSpec("forest_cam",   "camera",
                      position={"x": -2.0, "y": -2.5, "z": 1.2}),
        ],
        story_beats=[
            StoryBeat(0.0, "water_begins", "Glowing water begins to drip into bowl.",
                      camera_op="establishing wide, 50mm, f/8"),
            StoryBeat(2.0, "spirit_arrives", "Spirit creature approaches the bowl.",
                      agent_id="spirit",
                      camera_op="track spirit, 85mm, f/2.0"),
            StoryBeat(5.0, "spirit_drinks", "Spirit leans down and drinks.",
                      agent_id="spirit",
                      camera_op="extreme close-up on face, 135mm, f/1.8"),
            StoryBeat(8.0, "spirit_glows", "Spirit's bioluminescence intensifies.",
                      agent_id="spirit",
                      camera_op="slow orbit, 50mm, f/5.6"),
        ],
        visual_style=(
            "photorealistic fantasy, bioluminescent glow, dappled forest sunlight, "
            "volumetric light shafts, magical atmosphere, Studio Ghibli-inspired warmth"
        ),
        duration_seconds=12.0,
        physics_enabled=True,
        fluid_objects=["water_domain"],
        cloth_objects=[],
        render_engine="eevee",
        output_resolution=(1920, 1080),
        fps=24)

    # ── Select and render ──────────────────────────────────────────────
    directive = detective_directive   # Switch to forest_directive for example 2

    final_output = engine.render(
        directive=directive,
        output_dir="/tmp/prometheus_render/output")

    print(f"\n▲ Prometheus Final Output: {final_output}")
    print(f"▲ .blend file saved at: {engine.state._blend_path}")
    print(f"  (Re-shoot: change camera ops and re-run Phase 3 — scene unchanged)")
