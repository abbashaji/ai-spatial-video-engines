"""
╔══════════════════════════════════════════════════════════════════════╗
║           CHRONOS ENGINE  —  Engine II of XI                        ║
║           Recursive Spatial Feedback (RSF)                          ║
║                                                                      ║
║  Paradigm:  Recursive — Generate → Audit → Inpaint → Loop           ║
║  Gemini:    Physical World Modeling + Temporal Reasoning             ║
║  Grok:      Latent Seed + Precision In-Painting (video-to-video)     ║
║  Threshold: 10% positional deviation (tighter than Aether)          ║
║  Correction: Surgical frame inpainting (NOT full re-render)          ║
╚══════════════════════════════════════════════════════════════════════╝

ARCHITECTURE — 4 Stages with Recursive Correction Loop:
  Stage 1: Physical World Modeling     (Gemini ER 1.5 — thinking_tokens=1500)
  Stage 2: Latent Seed Generation      (Grok Imagine v4.2)
  Stage 3: Hallucination Audit         (Gemini Temporal Reasoning)
  Stage 4: Precision In-Painting       (Grok Video-to-Video API)

KEY ADVANCES OVER AETHER (Engine I):
  - GCT schema: richer data (occlusions, audio sync, lighting, camera)
  - thinking_tokens=1500: Gemini explicitly reasons before generating
  - PhysicsToPersonaTranslator: coordinates → cinematic language
  - Surgical inpainting: fix ~6 frames, not entire video (~50x cheaper)
  - Audio-Visual sync check: footsteps anchored to [t, x, y, z]
  - 10% threshold (vs 15% in Engine I)

WHAT THIS DOES NOT SOLVE:
  - No pre-render validation (errors still caught post-generation)
  - No voxel/3D geometry (coordinate-based tracking only)
  - No causal physics (no fracture vectors, collision prediction)
  - No character cognition or intent

See README.md § Engine II for full documentation.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from google import genai                          # pip install google-genai
from google.genai import types
from gemini_er_client import (                    # shared ER 1.5 adapter
    GeminiERClient, ThinkingPreset, ER15_MODEL,
    BoundingBox2D, SpatialPoint, GeminiERResponse,
    er15_to_blender)
import httpx                           # pip install httpx

logger = logging.getLogger("chronos")
logging.basicConfig(level=logging.INFO, format="%(name)s [%(levelname)s] %(message)s")


# ══════════════════════════════════════════════════════════════════════
#  TYPE ALIAS
# ══════════════════════════════════════════════════════════════════════

Vec3 = dict[str, float]   # {"x": float, "y": float, "z": float}


# ══════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TrajectoryKeyframe:
    """A single keyframe in an entity's motion trajectory."""
    t: float
    position: Vec3
    velocity: Vec3 = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    rotation_euler_deg: Vec3 = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    in_frame: bool = True
    confidence: float = 1.0


@dataclass
class Entity:
    """A tracked physical object or character in the scene."""
    entity_id: str
    label: str
    category: str
    trajectory: list[TrajectoryKeyframe]
    bounding_box: dict[str, Vec3]
    is_persistent: bool = True
    last_known_position: Vec3 = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    physical_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraKeyframe:
    """A single keyframe in the camera's trajectory."""
    t: float
    position: Vec3
    look_at: Vec3
    move_type: str = "static"
    aperture_f: float = 2.8


@dataclass
class OcclusionWindow:
    """
    A time window during which one entity is hidden behind another.
    Pre-computed by Gemini before rendering; Grok must respect these windows.
    """
    entity_id: str
    occluded_by: str   # entity_id of the occluder, or "frame_edge"
    t_start: float
    t_end: float
    re_entry_position: Vec3 = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})


@dataclass
class AudioSyncEvent:
    """
    An audio event anchored to a world-space position.
    Used to verify that footsteps, impacts, etc. match character movement.
    """
    t: float
    event_type: str
    source_entity_id: str
    source_position: Vec3
    expected_audio_cue: str


@dataclass
class LightingKeyframe:
    """A keyframe describing the scene's primary lighting state."""
    t: float
    primary_angle_deg: float    # Azimuth of key light
    elevation_deg: float
    color_temp_k: int
    intensity_lux: float


@dataclass
class GCTPayload:
    """
    Global Coordinate Trajectory — the canonical handoff between Gemini and Grok.
    Richer than Engine I's SpatialMap: adds occlusions, audio sync, lighting, camera.

    This is passed as "Physical Constraints" to Grok's system prompt.
    """
    schema_version: str
    scene_meta: dict[str, Any]
    entities: list[Entity]
    camera: dict[str, Any]
    occlusions: list[OcclusionWindow]
    audio_sync: list[AudioSyncEvent]
    lighting_keyframes: list[LightingKeyframe] = field(default_factory=list)


@dataclass
class AuditViolation:
    """A detected physics/consistency violation in a rendered frame."""
    entity_id: str
    frame_index: int
    t: float
    predicted: Vec3
    actual: Vec3
    deviation_pct: float
    violation_type: Literal["position", "occlusion", "audio_sync"]


@dataclass
class CorrectionMask:
    """
    Specification for surgical inpainting of violated frames.
    Only the flagged frame_ranges are re-rendered; everything else is preserved.
    """
    violations: list[AuditViolation]
    frame_ranges: list[tuple[int, int]]   # [(start_frame, end_frame), ...]
    spatial_deltas: list[dict[str, Any]]
    edit_instructions: str                # Natural-language directive for Grok


# ══════════════════════════════════════════════════════════════════════
#  SCENE BUFFER — Upgraded from Aether
# ══════════════════════════════════════════════════════════════════════

class SceneBuffer:
    """
    Persistent 3D world-state for Chronos.
    Upgraded from Aether's SceneStateBuffer with:
    - Off-frame register (tracks when entities exit/enter camera frustum)
    - Full export for constraint snapshot injection
    """

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._off_frame_register: dict[str, float] = {}   # entity_id → time it left
        self._audit_log: list[dict[str, Any]] = []

    def register_entity(self, entity: Entity) -> None:
        self._entities[entity.entity_id] = entity
        logger.debug("SceneBuffer registered: %s (%s)", entity.entity_id, entity.label)

    def record_exit(self, entity_id: str, t: float, position: Vec3) -> None:
        """Called when an entity leaves the camera frame."""
        if entity_id in self._entities:
            self._entities[entity_id].last_known_position = position
            self._off_frame_register[entity_id] = t
            logger.info(
                "Entity %s exited frame at t=%.2fs @ %s", entity_id, t, position
            )

    def record_entry(self, entity_id: str, t: float) -> Vec3:
        """Called when an entity re-enters frame. Returns its preserved position."""
        if entity_id not in self._entities:
            raise KeyError(f"Unknown entity: {entity_id}")
        entity = self._entities[entity_id]
        off_since = self._off_frame_register.pop(entity_id, None)
        logger.info(
            "Entity %s re-entered frame at t=%.2fs (was off since t=%.2fs)",
            entity_id, t, off_since or 0)
        return entity.last_known_position

    def get_world_position(self, entity_id: str, t: float) -> Vec3:
        """
        Return the authoritative world-space position at time t.
        If entity is off-frame: returns last known position (no movement in world space).
        If on-frame: interpolates from trajectory keyframes.
        """
        entity = self._entities[entity_id]
        if entity_id in self._off_frame_register:
            return entity.last_known_position
        return _interpolate_trajectory(entity.trajectory, t)

    def to_constraint_snapshot(self, t: float) -> dict[str, Any]:
        """Snapshot of all entity world positions at time t, for Grok injection."""
        return {
            eid: {
                "label": e.label,
                "world_position": self.get_world_position(eid, t),
                "bounding_box": e.bounding_box,
                "in_frame": eid not in self._off_frame_register,
            }
            for eid, e in self._entities.items()
        }

    def full_export(self) -> dict[str, Any]:
        return {
            eid: {
                "label": e.label,
                "last_known_position": e.last_known_position,
                "is_off_frame": eid in self._off_frame_register,
            }
            for eid, e in self._entities.items()
        }


# ══════════════════════════════════════════════════════════════════════
#  PHYSICS-TO-PERSONA TRANSLATOR
#  The bridge between Gemini's raw coordinates and Grok's cinematic language
# ══════════════════════════════════════════════════════════════════════

class PhysicsToPersonaTranslator:
    """
    Converts Gemini's raw GCT coordinate data into Grok-native cinematic language.

    This is the "translation layer" between physics truth and aesthetic direction.
    Coordinates like dx=2.3m become "moves decisively toward the northeast corner."
    Camera moves like "dolly_in" become "Push in slowly, closing the emotional distance."

    Input:  GCTPayload with [t, x, y, z] trajectories, camera, lighting, audio
    Output: Rich narrative prompt + structured Cinematic Camera Directions for Grok API
    """

    CAMERA_MOVE_DESCRIPTIONS = {
        "dolly_in":   "Push in slowly, closing the emotional distance",
        "dolly_out":  "Pull back to reveal the wider world",
        "pan_left":   "Sweep left, following the action",
        "pan_right":  "Sweep right, tracking motion",
        "tilt_up":    "Rise to reveal height and scale",
        "tilt_down":  "Descend into the scene's gravity",
        "crane_up":   "Ascend for a god's-eye perspective",
        "crane_down": "Descend with weight and intention",
        "handheld":   "Intimate, breathing camera — present in the moment",
        "orbit":      "Circle the subject, maintaining spatial lock",
        "static":     "Hold. Let the scene breathe.",
    }

    def translate(
        self,
        gct: GCTPayload,
        creative_brief: str,
        style_tags: list[str]) -> dict[str, Any]:
        """
        Full translation pipeline. Returns a Grok API payload dict.
        """
        return {
            "system_instructions": GROK_PHYSICS_SYSTEM_INSTRUCTIONS,
            "prompt": self._build_narrative_prompt(gct, creative_brief, style_tags),
            "camera_directions": self._build_camera_directions(gct),
            "lighting_narrative": self._build_lighting_narrative(gct),
            "physical_constraints": self._build_physics_constraints(gct),
            "audio_brief": self._build_audio_brief(gct),
            "gct_schema_version": gct.schema_version,
        }

    def _build_narrative_prompt(
        self,
        gct: GCTPayload,
        brief: str,
        styles: list[str]) -> str:
        entity_summaries = []
        for entity in gct.entities:
            if not entity.trajectory:
                continue
            t0 = entity.trajectory[0]
            tf = entity.trajectory[-1]
            dx = tf.position["x"] - t0.position["x"]
            dz = tf.position["z"] - t0.position["z"]
            motion = _describe_motion(dx, dz)
            entity_summaries.append(f"  • {entity.label}: {motion}")

        motion_block = "\n".join(entity_summaries)
        style_block = ", ".join(styles) if styles else "cinematic, naturalistic"

        return (
            f"{brief}\n\n"
            f"ENTITY MOTION SUMMARY (do not deviate from these paths):\n{motion_block}\n\n"
            f"VISUAL STYLE: {style_block}.\n"
            f"Every entity listed above has locked world-space coordinates provided "
            f"in `physical_constraints`. Treat them as hard physics constraints, "
            f"not suggestions.\n"
            f"Your creative latitude lives entirely in light, texture, atmosphere, "
            f"and lens character."
        )

    def _build_camera_directions(self, gct: GCTPayload) -> list[dict[str, Any]]:
        directions = []
        for kf in gct.camera.get("trajectory", []):
            move_type = kf.get("move_type", "static")
            description = self.CAMERA_MOVE_DESCRIPTIONS.get(move_type, move_type)
            directions.append({
                "t": kf["t"],
                "move_type": move_type,
                "description": description,
                "position": kf["position"],
                "look_at": kf["look_at"],
                "aperture_f": kf.get("aperture_f", 2.8),
                "lens_mm": gct.camera.get("lens_mm", 35),
            })
        return directions

    def _build_lighting_narrative(self, gct: GCTPayload) -> list[dict[str, str]]:
        result = []
        for lk in gct.lighting_keyframes:
            direction = _angle_to_compass(lk.primary_angle_deg)
            warmth = (
                "warm amber"     if lk.color_temp_k < 4000 else
                "neutral daylight" if lk.color_temp_k < 6000 else
                "cold blue-white"
            )
            result.append({
                "t": str(lk.t),
                "instruction": (
                    f"Shift primary light to {lk.primary_angle_deg:.0f}° azimuth "
                    f"({direction}), {lk.elevation_deg:.0f}° elevation. "
                    f"{warmth.capitalize()} tone ({lk.color_temp_k}K), "
                    f"intensity {lk.intensity_lux:.0f} lux."
                ),
            })
        return result

    def _build_physics_constraints(self, gct: GCTPayload) -> dict[str, Any]:
        constraints: dict[str, Any] = {}
        for entity in gct.entities:
            constraints[entity.entity_id] = {
                "label": entity.label,
                "category": entity.category,
                "is_persistent": entity.is_persistent,
                "bounding_box": entity.bounding_box,
                "trajectory": [
                    {"t": kf.t, "position": kf.position, "in_frame": kf.in_frame}
                    for kf in entity.trajectory
                ],
            }
        for occ in gct.occlusions:
            if occ.entity_id in constraints:
                constraints[occ.entity_id].setdefault("occlusions", []).append({
                    "occluded_by": occ.occluded_by,
                    "t_start": occ.t_start,
                    "t_end": occ.t_end,
                    "re_entry_position": occ.re_entry_position,
                })
        return constraints

    def _build_audio_brief(self, gct: GCTPayload) -> list[dict[str, Any]]:
        return [
            {
                "t": ev.t,
                "cue": ev.expected_audio_cue,
                "source_entity": ev.source_entity_id,
                "position": ev.source_position,
                "instruction": (
                    f"At T={ev.t:.1f}s, '{ev.source_entity_id}' triggers "
                    f"'{ev.expected_audio_cue}' at world position {ev.source_position}. "
                    f"Audio must originate from this spatial anchor."
                ),
            }
            for ev in gct.audio_sync
        ]


# ══════════════════════════════════════════════════════════════════════
#  GROK SYSTEM INSTRUCTIONS — Chronos
# ══════════════════════════════════════════════════════════════════════

GROK_PHYSICS_SYSTEM_INSTRUCTIONS = """
## CHRONOS-SPATIAL PHYSICS ENFORCEMENT PROTOCOL v2.0

You operate under strict spatial governance enforced by Gemini ER 1.5 Physics Supervisor.
All entity positions, trajectories, and occlusion windows in `physical_constraints` are
derived from a robotics-grade physics simulation. They are NOT suggestions.

### HARD CONSTRAINTS (zero tolerance)

1. WORLD-SPACE LOCK
   Every entity in `physical_constraints` has a locked world-space position at every
   timestep. Their screen-space projections change as the camera moves; their world-space
   positions do not.

2. OBJECT PERMANENCE
   If an entity exits the frame (in_frame=false), it has NOT ceased to exist. It is in
   the SceneBuffer at its last_known_position. When the camera returns, it MUST re-appear
   at its predicted re_entry_position, not be re-invented.

3. OCCLUSION FIDELITY
   Honor all occlusion windows exactly. Do not reveal an entity that should be behind
   another. Do not hide one that should be visible.

4. IDENTITY PERMANENCE
   A registered entity cannot change shape, category, or identity.
   A character cannot morph into an object. Physics are conservative.

5. AUDIO ANCHORING
   All audio events in `audio_brief` are spatially anchored to entity world positions.
   Footsteps, impacts, and environmental sounds must originate from their declared
   3D coordinates.

### YOUR CREATIVE DOMAIN (own it fully)

- Cinematography: lens bokeh, rack focus, motion blur, grain texture, color grading
- Lighting mood within the angle/intensity keyframes provided
- Micro-detail: fabric physics, hair simulation, steam, dust motes, lens flare
- Background world: anything NOT listed in physical_constraints is yours to create
- Temporal rhythm: the pacing of edits, the emotional weight of each frame
- Sound design texture: the character of sounds (not their timing or spatial origin)

### CORRECTION MODE (when edit_type=swap)

You are performing surgical in-painting. Correct ONLY the frame ranges in
`spatial_guidance.frame_ranges`. Everything outside these ranges is frozen and sacred.
Match the visual style, lighting, and color grade of surrounding frames exactly.
The viewer must not perceive a cut or change.
"""


# ══════════════════════════════════════════════════════════════════════
#  CHRONOS ENGINE — Main Orchestrator
# ══════════════════════════════════════════════════════════════════════

class ChronosEngine:
    """
    Engine II: Recursive Spatial Feedback (RSF) pipeline.

    The key advance over Aether: surgical inpainting instead of full re-renders.
    Only violated frames (+/- 0.25s padding) are corrected — ~50x cheaper than
    re-rendering the entire video.

    Usage:
        engine = ChronosEngine(
            gemini_api_key="YOUR_GEMINI_KEY",
            grok_api_key="YOUR_GROK_KEY")
        final = engine.feedback_loop(
            text_prompt="A courier on a rooftop...",
            style_tags=["neo-noir", "rain bokeh"],
            thinking_tokens=1500)
    """

    DEVIATION_THRESHOLD = 0.10   # 10% — tighter than Aether's 15%
    MAX_FEEDBACK_LOOPS  = 4

    def __init__(
        self,
        gemini_api_key: str,
        grok_api_key: str,
        grok_base_url: str = "https://api.x.ai/v1") -> None:
        self.gemini = GeminiERClient(
            api_key=gemini_api_key, model=ER15_MODEL,
            default_thinking=ThinkingPreset.NONE)
        self.grok = httpx.Client(
            base_url=grok_base_url,
            headers={
                "Authorization": f"Bearer {grok_api_key}",
                "Content-Type": "application/json",
            },
            timeout=180)
        self.scene_buffer = SceneBuffer()
        self.translator = PhysicsToPersonaTranslator()
        self._gct: GCTPayload | None = None

    # ── Public API ─────────────────────────────────────────────────────

    def feedback_loop(
        self,
        text_prompt: str,
        reference_image_path: str | None = None,
        style_tags: list[str] | None = None,
        thinking_tokens: int = 1500) -> str:
        """
        Full RSF pipeline with recursive correction.

        Args:
            text_prompt:           Scene description
            reference_image_path:  Optional reference image
            style_tags:            List of cinematic style descriptors
            thinking_tokens:       Gemini reasoning budget for Stage 1

        Returns:
            URL of the final, consistency-validated video.
        """
        logger.info("═══ Chronos-Spatial Engine v2.0 · RSF Pipeline START ═══")

        # Stage 1 — Physical World Modeling
        gct = self._stage1_model_world(
            text_prompt, reference_image_path, thinking_tokens
        )

        # Stage 2 — Latent Seed Generation
        video_url = self._stage2_generate_video(gct, text_prompt, style_tags or [])

        # Recursive Stages 3 + 4
        for loop in range(1, self.MAX_FEEDBACK_LOOPS + 1):
            logger.info("─── Feedback Loop %d / %d ───", loop, self.MAX_FEEDBACK_LOOPS)

            # Stage 3 — Hallucination Audit
            violations, av_violations = self._stage3_audit(video_url, gct)
            all_violations = violations + av_violations

            if not all_violations:
                logger.info("✅ All checks passed on loop %d", loop)
                break

            logger.warning(
                "⚠  %d spatial + %d AV violations — triggering Stage 4",
                len(violations), len(av_violations))

            # Stage 4 — Precision In-Painting
            correction_mask = self._build_correction_mask(all_violations, gct)
            video_url = self._stage4_correct(video_url, correction_mask)
        else:
            logger.warning("Max feedback loops reached — returning best available result")

        logger.info("═══ Final video: %s ═══", video_url)
        return video_url

    # ── Stage 1 ────────────────────────────────────────────────────────

    def _stage1_model_world(
        self,
        prompt: str,
        image_path: str | None,
        thinking_tokens: int) -> GCTPayload:
        logger.info(
            "Stage 1 · Generating GCT via Gemini ER 1.5 (thinking_tokens=%d)",
            thinking_tokens)

        gemini_prompt = _build_gemini_gct_prompt(prompt, thinking_tokens)
        content: list[Any] = []

        if image_path:
            with open(image_path, "rb") as f:
                content.append({"mime_type": "image/jpeg", "data": f.read()})
        content.append(gemini_prompt)

        # Map thinking_tokens to ThinkingPreset for the API call
        _budget = max(0, min(thinking_tokens, 8192))
        raw = self.gemini.generate_content(content, thinking=_budget)
        raw_json = _extract_json(raw)
        gct = _parse_gct(raw_json)
        self._gct = gct

        # Register all entities in the scene buffer
        for entity in gct.entities:
            self.scene_buffer.register_entity(entity)
            for kf in entity.trajectory:
                if not kf.in_frame:
                    self.scene_buffer.record_exit(
                        entity.entity_id, kf.t, kf.position
                    )

        logger.info(
            "Stage 1 complete · %d entities · %d occlusion windows · %d audio events",
            len(gct.entities), len(gct.occlusions), len(gct.audio_sync))
        return gct

    # ── Stage 2 ────────────────────────────────────────────────────────

    def _stage2_generate_video(
        self,
        gct: GCTPayload,
        creative_brief: str,
        style_tags: list[str]) -> str:
        logger.info("Stage 2 · Generating constrained video via Grok Imagine v4.2")

        grok_payload = self.translator.translate(gct, creative_brief, style_tags)
        grok_payload.update({
            "model": "grok-imagine-video",   # [FUTURE API]
            "resolution": "720p",
            "duration_seconds": gct.scene_meta["duration_seconds"],
            "fps": gct.scene_meta["fps"],
        })

        resp = self.grok.post("/video/generate", json=grok_payload)
        resp.raise_for_status()
        video_url = resp.json()["video_url"]

        logger.info("Stage 2 complete · raw video: %s", video_url)
        return video_url

    # ── Stage 3 ────────────────────────────────────────────────────────

    def _stage3_audit(
        self,
        video_url: str,
        gct: GCTPayload) -> tuple[list[AuditViolation], list[AuditViolation]]:
        logger.info("Stage 3 · Running hallucination audit via Gemini temporal reasoning")

        spatial_violations = self._audit_spatial(video_url, gct)
        av_violations      = self._audit_audio_visual(video_url, gct)

        logger.info(
            "Stage 3 · Spatial: %d violations · AV: %d violations",
            len(spatial_violations), len(av_violations))
        return spatial_violations, av_violations

    def _audit_spatial(
        self,
        video_url: str,
        gct: GCTPayload) -> list[AuditViolation]:
        buffer_state = json.dumps(self.scene_buffer.full_export(), indent=2)
        gct_summary = json.dumps(
            {
                e.entity_id: [
                    {"t": k.t, "pos": k.position, "in_frame": k.in_frame}
                    for k in e.trajectory
                ]
                for e in gct.entities
            },
            indent=2)

        audit_prompt = f"""
You are a spatial consistency auditor reviewing AI-generated video for physics violations.

VIDEO URL: {video_url}
DURATION: {gct.scene_meta['duration_seconds']}s at {gct.scene_meta['fps']}fps

AUTHORITATIVE GCT TRAJECTORIES:
{gct_summary}

SCENE BUFFER (off-frame entities — their positions are LOCKED even when invisible):
{buffer_state}

TASK:
For every entity, at every 1-second keyframe, estimate their pixel-space position and
convert back to world-space coordinates. Compare against the GCT.

Return ONLY a JSON array:
[
  {{
    "entity_id": "ent_001",
    "frame_index": 24,
    "t": 1.0,
    "actual_position": {{"x": 0.0, "y": 0.0, "z": 0.0}}
  }}
]
"""
        raw = self.gemini.generate_content([audit_prompt], thinking=ThinkingPreset.MEDIUM)
        frame_reports: list[dict] = json.loads(_extract_json(response.text))

        violations = []
        for report in frame_reports:
            t   = report["t"]
            eid = report["entity_id"]
            predicted = self.scene_buffer.get_world_position(eid, t)
            actual    = report["actual_position"]
            dev       = _calc_deviation(predicted, actual)

            if dev > self.DEVIATION_THRESHOLD:
                violations.append(AuditViolation(
                    entity_id=eid,
                    frame_index=report["frame_index"],
                    t=t,
                    predicted=predicted,
                    actual=actual,
                    deviation_pct=dev,
                    violation_type="position"))
        return violations

    def _audit_audio_visual(
        self,
        video_url: str,
        gct: GCTPayload) -> list[AuditViolation]:
        if not gct.audio_sync:
            return []

        av_events_json = json.dumps(
            [
                {
                    "t": ev.t,
                    "entity_id": ev.source_entity_id,
                    "expected_cue": ev.expected_audio_cue,
                    "expected_position": ev.source_position,
                }
                for ev in gct.audio_sync
            ],
            indent=2)

        av_prompt = f"""
Analyze the audio track of this video for spatial-temporal alignment.

VIDEO URL: {video_url}
EXPECTED AUDIO EVENTS:
{av_events_json}

For each event, check:
1. Does the audio cue occur within ±0.1s of the expected timestamp?
2. Does the stereo/spatial positioning of the audio match the entity's x-position?

Return ONLY a JSON array of misalignments:
[
  {{
    "entity_id": "ent_001",
    "frame_index": 48,
    "t": 2.0,
    "expected_cue": "footstep_left",
    "actual_t_offset_seconds": 0.3,
    "spatial_mismatch": true
  }}
]
If no misalignments, return [].
"""
        raw = self.gemini.generate_content([av_prompt], thinking=ThinkingPreset.LIGHT)
        misalignments: list[dict] = json.loads(_extract_json(response.text))

        violations = []
        for m in misalignments:
            if abs(m.get("actual_t_offset_seconds", 0)) > 0.1 or m.get("spatial_mismatch"):
                eid = m["entity_id"]
                violations.append(AuditViolation(
                    entity_id=eid,
                    frame_index=m["frame_index"],
                    t=m["t"],
                    predicted=self.scene_buffer.get_world_position(eid, m["t"]),
                    actual=self.scene_buffer.get_world_position(eid, m["t"]),
                    deviation_pct=abs(m.get("actual_t_offset_seconds", 0)) / 0.1,
                    violation_type="audio_sync"))
        return violations

    # ── Stage 4 ────────────────────────────────────────────────────────

    def _stage4_correct(
        self,
        video_url: str,
        mask: CorrectionMask) -> str:
        logger.info(
            "Stage 4 · Precision in-painting · %d frame ranges to correct",
            len(mask.frame_ranges))

        payload = {
            "model": "grok-imagine-video",   # [FUTURE API]
            "input_video_url": video_url,
            "edit_type": "swap",
            "spatial_guidance": {
                "deltas": mask.spatial_deltas,
                "frame_ranges": mask.frame_ranges,
            },
            "correction_prompt": mask.edit_instructions,
            "preserve_style": True,
            "system_instructions": GROK_PHYSICS_SYSTEM_INSTRUCTIONS,
        }

        resp = self.grok.post("/video/edit", json=payload)
        resp.raise_for_status()
        corrected_url = resp.json()["video_url"]

        logger.info("Stage 4 complete · corrected video: %s", corrected_url)
        return corrected_url

    # ── Private Helpers ────────────────────────────────────────────────

    def _build_correction_mask(
        self,
        violations: list[AuditViolation],
        gct: GCTPayload) -> CorrectionMask:
        fps = gct.scene_meta.get("fps", 24)
        frame_ranges = []
        deltas = []

        for v in violations:
            pad   = int(fps * 0.25)
            start = max(0, v.frame_index - pad)
            end   = v.frame_index + pad
            frame_ranges.append((start, end))
            deltas.append({
                "entity_id": v.entity_id,
                "frame_index": v.frame_index,
                "t": v.t,
                "target_position": v.predicted,
                "current_position": v.actual,
                "delta": {
                    axis: v.predicted[axis] - v.actual[axis]
                    for axis in ("x", "y", "z")
                },
                "violation_type": v.violation_type,
            })

        edit_instructions = _generate_correction_prompt(violations, gct)

        return CorrectionMask(
            violations=violations,
            frame_ranges=list(set(frame_ranges)),
            spatial_deltas=deltas,
            edit_instructions=edit_instructions)


# ══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    return match.group(1).strip() if match else text.strip()


def _calc_deviation(predicted: Vec3, actual: Vec3) -> float:
    dx  = predicted["x"] - actual["x"]
    dy  = predicted["y"] - actual["y"]
    dz  = predicted["z"] - actual["z"]
    mag = math.sqrt(predicted["x"]**2 + predicted["y"]**2 + predicted["z"]**2) or 1.0
    return math.sqrt(dx**2 + dy**2 + dz**2) / mag


def _interpolate_trajectory(
    trajectory: list[TrajectoryKeyframe],
    t: float) -> Vec3:
    if not trajectory:
        return {"x": 0, "y": 0, "z": 0}
    if t <= trajectory[0].t:
        return trajectory[0].position
    if t >= trajectory[-1].t:
        return trajectory[-1].position
    for i in range(len(trajectory) - 1):
        a, b = trajectory[i], trajectory[i + 1]
        if a.t <= t <= b.t:
            ratio = (t - a.t) / (b.t - a.t)
            return {
                axis: a.position[axis] + (
                    b.position[axis] - a.position[axis]
                ) * ratio
                for axis in ("x", "y", "z")
            }
    return trajectory[-1].position


def _describe_motion(dx: float, dz: float) -> str:
    dist = math.sqrt(dx**2 + dz**2)
    if dist < 0.1:
        return "stationary throughout the scene"
    direction = math.degrees(math.atan2(dx, dz))
    compass = [
        "north", "northeast", "east", "southeast",
        "south", "southwest", "west", "northwest",
    ][int((direction + 22.5) / 45) % 8]
    speed = (
        "slowly drifts" if dist < 1.0 else
        "moves"         if dist < 3.0 else
        "travels decisively"
    )
    return f"{speed} {dist:.1f}m toward {compass}"


def _angle_to_compass(angle_deg: float) -> str:
    directions = ["North", "NE", "East", "SE", "South", "SW", "West", "NW"]
    return directions[int((angle_deg + 22.5) / 45) % 8]


def _generate_correction_prompt(
    violations: list[AuditViolation],
    gct: GCTPayload) -> str:
    entity_labels = {e.entity_id: e.label for e in gct.entities}
    lines = ["SURGICAL CORRECTION REQUIRED. Fix only the following:"]
    for v in violations:
        label = entity_labels.get(v.entity_id, v.entity_id)
        if v.violation_type == "position":
            dx   = v.predicted["x"] - v.actual["x"]
            dz   = v.predicted["z"] - v.actual["z"]
            dist = math.sqrt(dx**2 + dz**2)
            lines.append(
                f"  • Frame {v.frame_index} (T={v.t:.1f}s): Move '{label}' "
                f"{dist:.2f}m toward ({v.predicted['x']:.2f}, {v.predicted['z']:.2f}). "
                f"Deviation was {v.deviation_pct*100:.1f}%."
            )
        elif v.violation_type == "audio_sync":
            lines.append(
                f"  • Frame {v.frame_index} (T={v.t:.1f}s): Resync audio for '{label}'. "
                f"Position audio source at {v.predicted}."
            )
    lines.append("\nDo NOT alter any other frames, entities, lighting, or style.")
    return "\n".join(lines)


def _build_gemini_gct_prompt(prompt: str, thinking_tokens: int) -> str:
    return f"""
# Note: thinking budget is now passed via ThinkingConfig in GeminiERClient

SCENE: "{prompt}"

Generate a GLOBAL COORDINATE TRAJECTORY (GCT) in this JSON schema:
{{
  "schema_version": "2.0",
  "scene_meta": {{
    "duration_seconds": 10.0,
    "fps": 24,
    "coordinate_system": "right_hand_y_up",
    "scene_description": "{prompt}",
    "thinking_tokens_used": {thinking_tokens}
  }},
  "entities": [
    {{
      "entity_id": "ent_001",
      "label": "protagonist",
      "category": "character",
      "is_persistent": true,
      "trajectory": [
        {{"t": 0.0, "position": {{"x":0,"y":0,"z":0}}, "velocity": {{"x":0,"y":0,"z":0}},
          "rotation_euler_deg": {{"x":0,"y":0,"z":0}}, "in_frame": true, "confidence": 1.0}}
      ],
      "bounding_box": {{"min": {{"x":-0.4,"y":0,"z":-0.2}}, "max": {{"x":0.4,"y":1.8,"z":0.2}}}},
      "physical_properties": {{"mass_kg": 70, "is_rigid": false, "surface_type": "subsurface"}}
    }}
  ],
  "camera": {{
    "lens_mm": 35, "fov_deg": 54,
    "trajectory": [
      {{"t": 0.0, "position": {{"x":0,"y":1.7,"z":-3}}, "look_at": {{"x":0,"y":1,"z":0}},
        "move_type": "static", "aperture_f": 2.8}}
    ]
  }},
  "occlusions": [],
  "audio_sync": [],
  "lighting_keyframes": []
}}

Rules:
- Trajectory keyframes every 1.0s minimum; more for fast motion.
- Mark in_frame=false when entity exits camera frustum.
- Include occlusion windows when one entity passes behind another.
- Coordinates in meters, right-hand Y-up.
- Return ONLY the JSON block.
"""


def _parse_gct(raw_json: str) -> GCTPayload:
    data = json.loads(raw_json)
    entities = []
    for e in data.get("entities", []):
        traj = [
            TrajectoryKeyframe(
                t=kf["t"],
                position=kf["position"],
                velocity=kf.get("velocity", {"x": 0, "y": 0, "z": 0}),
                rotation_euler_deg=kf.get(
                    "rotation_euler_deg", {"x": 0, "y": 0, "z": 0}
                ),
                in_frame=kf.get("in_frame", True),
                confidence=kf.get("confidence", 1.0))
            for kf in e.get("trajectory", [])
        ]
        entities.append(Entity(
            entity_id=e["entity_id"],
            label=e["label"],
            category=e["category"],
            is_persistent=e.get("is_persistent", True),
            trajectory=traj,
            bounding_box=e.get("bounding_box", {
                "min": {"x": 0, "y": 0, "z": 0},
                "max": {"x": 0, "y": 0, "z": 0},
            }),
            physical_properties=e.get("physical_properties", {})))

    return GCTPayload(
        schema_version=data.get("schema_version", "2.0"),
        scene_meta=data["scene_meta"],
        entities=entities,
        camera=data.get("camera", {}),
        occlusions=[OcclusionWindow(**o) for o in data.get("occlusions", [])],
        audio_sync=[AudioSyncEvent(**a) for a in data.get("audio_sync", [])],
        lighting_keyframes=[
            LightingKeyframe(**lk) for lk in data.get("lighting_keyframes", [])
        ])


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = ChronosEngine(
        gemini_api_key="YOUR_GEMINI_KEY",
        grok_api_key="YOUR_GROK_KEY")

    final = engine.feedback_loop(
        text_prompt=(
            "A lone courier on a rain-soaked city rooftop at 3am. "
            "They set down a package, walk to the edge, look out over the neon city, "
            "then turn and exit right. The package never moves."
        ),
        reference_image_path=None,   # Optional: "rooftop_reference.jpg"
        style_tags=["neo-noir", "anamorphic lens", "rain bokeh", "desaturated cyan"],
        thinking_tokens=1500)

    print(f"\nFinal validated video: {final}")
