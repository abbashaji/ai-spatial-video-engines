"""
╔══════════════════════════════════════════════════════════════════════╗
║           AETHER ENGINE  —  Engine I of XI                          ║
║           Sequential Spatial Correction                              ║
║                                                                      ║
║  Paradigm:  Sequential — Generate → Check → Fix                     ║
║  Gemini:    Spatial Blueprint (2D/3D coordinate mapping)             ║
║  Grok:      Constrained video generation                             ║
║  Threshold: 15% positional deviation                                 ║
║  Correction: Full video re-render                                    ║
╚══════════════════════════════════════════════════════════════════════╝

ARCHITECTURE — 3 Sequential Stages:
  Stage 1: The Spatial Blueprint        (Gemini ER 1.5)
  Stage 2: The Logical Orchestrator     (Grok Imagine)
  Stage 3: The Verification Loop        (Gemini Temporal Reasoning)

WHAT THIS SOLVES:
  - Objects drifting between frames (coordinate locking)
  - Off-screen objects returning to wrong positions (SceneStateBuffer)
  - Clear separation: Gemini owns spatial truth, Grok owns aesthetics

WHAT THIS DOES NOT SOLVE:
  - Full re-render required for any violation (expensive)
  - No pre-render validation
  - No physics causality
  - No character cognition

See README.md § Engine I for full documentation.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from google import genai                          # pip install google-genai
from google.genai import types
from gemini_er_client import (                    # shared ER 1.5 adapter
    GeminiERClient, ThinkingPreset, ER15_MODEL,
    BoundingBox2D, SpatialPoint, GeminiERResponse,
    er15_to_blender)
import httpx                           # pip install httpx

logger = logging.getLogger("aether")
logging.basicConfig(level=logging.INFO, format="%(name)s [%(levelname)s] %(message)s")


# ══════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in world space (meters)."""
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float


@dataclass
class ObjectTrajectory:
    """
    Motion description for a single tracked object from T0 to T_final.
    Positions are in world space, right-hand Y-up coordinate system.
    """
    object_id: str
    label: str
    t0_position: dict[str, float]       # {"x": ..., "y": ..., "z": ...}
    t_final_position: dict[str, float]
    bounding_box: BoundingBox
    motion_path: list[dict[str, float]] = field(default_factory=list)


@dataclass
class SpatialMap:
    """
    The canonical output of Stage 1 — Gemini's blueprint of the scene.
    This is the ground truth that Grok must respect.
    """
    scene_description: str
    objects: list[ObjectTrajectory]
    camera_trajectory: list[dict[str, float]]
    duration_seconds: float = 10.0
    fps: int = 24


@dataclass
class FrameAuditResult:
    """Result of comparing a single frame's rendered position against the GCT."""
    frame_index: int
    object_id: str
    predicted_position: dict[str, float]
    actual_position: dict[str, float]
    deviation_pct: float
    requires_correction: bool


# ══════════════════════════════════════════════════════════════════════
#  SCENE STATE BUFFER
#  Solves the "Goldfish Effect" for Engine I
# ══════════════════════════════════════════════════════════════════════

class SceneStateBuffer:
    """
    Stores the authoritative 3D world position of every tracked object.

    KEY INSIGHT: When the camera pans away from an object, that object
    has NOT ceased to exist. It remains at its last known world-space
    position. When the camera returns, the object is restored from this
    buffer rather than re-hallucinated by the generative model.

    This solves the "Goldfish Effect" — objects that change state
    or position when re-entering frame after being off-screen.
    """

    def __init__(self) -> None:
        self._state: dict[str, ObjectTrajectory] = {}
        self._history: list[dict[str, Any]] = []   # timestamped snapshots

    def register(self, obj: ObjectTrajectory) -> None:
        """Register a new tracked object."""
        self._state[obj.object_id] = obj
        logger.debug("SceneBuffer registered: %s (%s)", obj.object_id, obj.label)

    def update_position(self, object_id: str, t: float, position: dict[str, float]) -> None:
        """Record the observed position of an object at time t."""
        if object_id not in self._state:
            raise KeyError(f"Object '{object_id}' not registered in SceneStateBuffer.")
        self._state[object_id].motion_path.append({"t": t, **position})
        self._history.append({
            "timestamp_wall": __import__("time").time(),
            "object_id": object_id,
            "t": t,
            "position": position,
        })

    def get_expected_position(self, object_id: str, t: float) -> dict[str, float]:
        """
        Linear interpolation between T0 and T_final for a given timestamp.
        This is the AUTHORITATIVE position — what Gemini says must be true.
        """
        obj = self._state[object_id]
        ratio = min(t / 10.0, 1.0)
        return {
            axis: obj.t0_position[axis] + (
                obj.t_final_position[axis] - obj.t0_position[axis]
            ) * ratio
            for axis in ("x", "y", "z")
        }

    def to_constraint_json(self) -> str:
        """Serialize all object states as Physical Constraints for Grok."""
        return json.dumps({
            oid: {
                "label": obj.label,
                "bounding_box": vars(obj.bounding_box),
                "t0": obj.t0_position,
                "t_final": obj.t_final_position,
            }
            for oid, obj in self._state.items()
        }, indent=2)


# ══════════════════════════════════════════════════════════════════════
#  GROK SYSTEM INSTRUCTIONS
#  The creative split: Gemini owns spatial truth, Grok owns aesthetics
# ══════════════════════════════════════════════════════════════════════

GROK_SYSTEM_INSTRUCTIONS = """
## AETHER PHYSICAL CONSTRAINTS PROTOCOL

You are a cinematic video generation engine operating under strict spatial governance.
A separate spatial reasoning system (Gemini ER 1.5) has pre-computed the authoritative
3D positions, bounding boxes, and motion trajectories for every object in this scene.
These are passed to you in the `physical_constraints` field.

### NON-NEGOTIABLE RULES

1. OBJECT PERMANENCE
   Every registered object MUST remain within its bounding box at all times.
   If your generative process would move an object outside its bounds,
   suppress that motion entirely.

2. IDENTITY LOCK
   A registered object cannot change its fundamental nature.
   A "coffee cup" cannot become a "vase," even under artistic reinterpretation.
   Visual style (texture, color grading, material) may evolve; geometry and
   identity may not.

3. TRAJECTORY ADHERENCE
   Objects with defined motion paths must follow them.
   Interpolate smoothly between t0 and t_final coordinates.
   Do not invent trajectories.

4. CAMERA AGNOSTICISM
   Object world-space coordinates are camera-independent.
   When the camera pans, tracks, or rotates, objects do not drift.
   Their world positions remain as specified; only their screen-space
   projections change.

### YOUR CREATIVE LATITUDE

- Lighting mood, color grading, atmosphere, lens flare, grain, vignette
- Texture richness, material quality, subsurface scattering
- Micro-motions within bounding boxes (breathing, fabric ripple, steam curl)
- Camera personality: handheld tension, dolly elegance, drone detachment
- Background elements NOT listed in `physical_constraints`
- Timing and rhythm of cuts (if multi-shot)
- Sound design metadata / sync cues

### CONSTRAINT PARSING

Read `physical_constraints` as:
  { "obj_001": { "label": "...", "bounding_box": {...}, "t0": {...}, "t_final": {...} } }

Before rendering each frame at time T, compute each object's expected position via
linear interpolation between t0 and t_final, and treat that as an immovable anchor.

### TONE

You are a world-class cinematographer who respects physics.
Your creativity lives in light, texture, and camera poetry — never in defying geometry.
"""


# ══════════════════════════════════════════════════════════════════════
#  AETHER ENGINE — Main Orchestrator
# ══════════════════════════════════════════════════════════════════════

class AetherEngine:
    """
    Engine I: Sequential Spatial Correction.

    Three-stage pipeline:
      Stage 1 · map_scene()                — Gemini ER 1.5 spatial blueprint
      Stage 2 · generate_constrained_video()— Grok Imagine with constraints
      Stage 3 · validate_consistency()     — Gemini temporal audit + correction

    Usage:
        engine = AetherEngine(
            gemini_api_key="YOUR_GEMINI_KEY",
            grok_api_key="YOUR_GROK_KEY")
        spatial_map = engine.map_scene("A café table at golden hour...")
        raw_video   = engine.generate_constrained_video("Slow push-in...")
        final_video = engine.validate_consistency(max_correction_passes=3)
    """

    DEVIATION_THRESHOLD = 0.15   # 15% — Engine I threshold

    def __init__(
        self,
        gemini_api_key: str,
        grok_api_key: str,
        grok_base_url: str = "https://api.x.ai/v1") -> None:
        self.gemini = GeminiERClient(
            api_key=gemini_api_key, model=ER15_MODEL,
            default_thinking=ThinkingPreset.NONE)
        self.grok_client = httpx.Client(
            base_url=grok_base_url,
            headers={
                "Authorization": f"Bearer {grok_api_key}",
                "Content-Type": "application/json",
            },
            timeout=120)
        self.scene_buffer = SceneStateBuffer()
        self._spatial_map: SpatialMap | None = None
        self._raw_video_url: str | None = None

    # ── Stage 1: Spatial Blueprint ─────────────────────────────────────

    def map_scene(
        self,
        prompt: str,
        keyframe_path: str | None = None) -> SpatialMap:
        """
        Stage 1: Call Gemini ER 1.5 to produce a Spatial JSON Map of the scene.

        Args:
            prompt:         Text description of the scene
            keyframe_path:  Optional path to a starting reference image

        Returns:
            SpatialMap containing all object trajectories and camera path.
            Also populates the SceneStateBuffer.
        """
        logger.info("Stage 1 · Generating spatial blueprint via Gemini ER 1.5")

        gemini_prompt = self._build_spatial_prompt(prompt, keyframe_path is not None)
        content_parts: list[Any] = []

        if keyframe_path:
            with open(keyframe_path, "rb") as f:
                content_parts.append({"mime_type": "image/jpeg", "data": f.read()})

        content_parts.append(gemini_prompt)

        response = self.gemini.generate_content(content_parts)
        raw_json = _extract_json(response.text)
        spatial_map = self._parse_spatial_map(raw_json, prompt)
        self._spatial_map = spatial_map

        # Register all objects in the state buffer
        for obj in spatial_map.objects:
            self.scene_buffer.register(obj)

        logger.info(
            "Stage 1 complete · %d objects registered", len(spatial_map.objects)
        )
        return spatial_map

    # ── Stage 2: Constrained Video Generation ─────────────────────────

    def generate_constrained_video(
        self,
        creative_prompt: str,
        style_tags: list[str] | None = None,
        resolution: str = "720p") -> str:
        """
        Stage 2: Pass Spatial JSON as Physical Constraints to Grok Imagine.

        Args:
            creative_prompt: Cinematic brief for Grok (tone, camera, style)
            style_tags:      List of style descriptors e.g. ["cinematic", "golden hour"]
            resolution:      Output resolution ("720p" | "1080p")

        Returns:
            URL of the raw generated video.
        """
        if self._spatial_map is None:
            raise RuntimeError("Call map_scene() before generate_constrained_video().")

        logger.info("Stage 2 · Generating constrained video via Grok Imagine")

        grok_payload = {
            "model": "grok-imagine-video",   # [FUTURE API]
            "resolution": resolution,
            "duration_seconds": self._spatial_map.duration_seconds,
            "fps": self._spatial_map.fps,
            "system_instructions": GROK_SYSTEM_INSTRUCTIONS,
            "prompt": self._build_constrained_prompt(creative_prompt, style_tags or []),
            "physical_constraints": json.loads(self.scene_buffer.to_constraint_json()),
            "camera_trajectory": self._spatial_map.camera_trajectory,
            "cinematic_parameters": {
                "color_grading": "cinematic",
                "depth_of_field": "auto",
                "motion_blur": True,
                "camera_direction": "narrative-driven",
            },
        }

        response = self.grok_client.post("/video/generate", json=grok_payload)
        response.raise_for_status()
        self._raw_video_url = response.json()["video_url"]

        logger.info("Stage 2 complete · raw video at %s", self._raw_video_url)
        return self._raw_video_url

    # ── Stage 3: Verification Loop ─────────────────────────────────────

    def validate_consistency(self, max_correction_passes: int = 3) -> str:
        """
        Stage 3: Use Gemini ER 1.5 temporal reasoning to audit each frame.
        Triggers Constraint-Corrected Edits via Grok if deviation > 15%.

        Args:
            max_correction_passes: Maximum re-render attempts before giving up

        Returns:
            URL of the final, consistency-validated video.
        """
        if not self._raw_video_url:
            raise RuntimeError("Call generate_constrained_video() first.")

        logger.info("Stage 3 · Running temporal consistency audit")
        video_url = self._raw_video_url

        for pass_num in range(1, max_correction_passes + 1):
            audit_results = self._run_temporal_audit(video_url)
            violations = [r for r in audit_results if r.requires_correction]

            if not violations:
                logger.info(
                    "Stage 3 · All frames consistent (pass %d)", pass_num
                )
                break

            logger.warning(
                "Stage 3 · %d violations detected on pass %d — requesting corrections",
                len(violations), pass_num)
            video_url = self._apply_corrections(video_url, violations)
        else:
            logger.warning(
                "Stage 3 · Max correction passes reached; returning best result."
            )

        logger.info("Stage 3 complete · final video at %s", video_url)
        return video_url

    # ── Private Helpers ────────────────────────────────────────────────

    def _run_temporal_audit(self, video_url: str) -> list[FrameAuditResult]:
        """Ask Gemini ER 1.5 to audit frame-by-frame object positions."""
        audit_prompt = f"""
You are auditing a generated video for spatial consistency.
Video URL: {video_url}
Scene state (authoritative positions): {self.scene_buffer.to_constraint_json()}

For each tracked object, at each keyframe (every 1 second), report:
- object_id, frame_index (0-based at 24fps), actual_x/y/z position.
Return ONLY a JSON array matching this schema:
[{{"object_id":"...","frame_index":0,"actual_position":{{"x":0,"y":0,"z":0}}}}]
"""
        response = self.gemini.generate_content(audit_prompt)
        raw = _extract_json(response.text)
        frame_reports: list[dict] = json.loads(raw)

        results = []
        fps = self._spatial_map.fps if self._spatial_map else 24
        duration = self._spatial_map.duration_seconds if self._spatial_map else 10.0

        for report in frame_reports:
            t = (report["frame_index"] / fps) * (duration / duration)
            predicted = self.scene_buffer.get_expected_position(report["object_id"], t)
            actual = report["actual_position"]
            deviation = _calc_deviation(predicted, actual)

            results.append(FrameAuditResult(
                frame_index=report["frame_index"],
                object_id=report["object_id"],
                predicted_position=predicted,
                actual_position=actual,
                deviation_pct=deviation,
                requires_correction=deviation > self.DEVIATION_THRESHOLD))

        return results

    def _apply_corrections(
        self,
        video_url: str,
        violations: list[FrameAuditResult]) -> str:
        """Send correction deltas back to Grok for targeted frame re-generation."""
        correction_payload = {
            "model": "grok-imagine-video",   # [FUTURE API]
            "source_video_url": video_url,
            "corrections": [
                {
                    "frame_index": v.frame_index,
                    "object_id": v.object_id,
                    "target_position": v.predicted_position,
                    "current_position": v.actual_position,
                }
                for v in violations
            ],
            "preserve_style": True,
            "system_instructions": GROK_SYSTEM_INSTRUCTIONS,
        }
        response = self.grok_client.post("/video/correct", json=correction_payload)
        response.raise_for_status()
        return response.json()["video_url"]

    @staticmethod
    def _build_spatial_prompt(prompt: str, has_keyframe: bool) -> str:
        keyframe_note = (
            "Use the attached keyframe as the starting state." if has_keyframe else ""
        )
        return f"""
You are a spatial reasoning engine analyzing a scene for video generation.
Scene description: "{prompt}"
{keyframe_note}

Produce a SPATIAL JSON MAP with this exact schema:
{{
  "objects": [
    {{
      "id": "obj_001",
      "label": "coffee cup",
      "t0": {{"x": 0.3, "y": 0.0, "z": 0.5}},
      "t_final": {{"x": 0.3, "y": 0.0, "z": 0.5}},
      "bounding_box": {{
        "x_min": 0.25, "y_min": -0.05, "z_min": 0.45,
        "x_max": 0.35, "y_max": 0.10, "z_max": 0.55
      }},
      "motion_path": []
    }}
  ],
  "camera_trajectory": [
    {{"t": 0.0, "x": 0, "y": 1.7, "z": -3.0}},
    {{"t": 10.0, "x": 2.0, "y": 1.7, "z": -3.0}}
  ]
}}

All coordinates are in meters, right-handed coordinate system, Y-up.
Return ONLY the JSON block, no prose.
"""

    @staticmethod
    def _build_constrained_prompt(
        creative_prompt: str,
        style_tags: list[str]) -> str:
        tags = ", ".join(style_tags) if style_tags else "cinematic, naturalistic"
        return (
            f"{creative_prompt} "
            f"Style: {tags}. "
            "CRITICAL: Every object must remain strictly within its provided "
            "coordinate bounds. Object positions are locked by the "
            "physical_constraints payload — do not reinterpret, relocate, "
            "or morph any registered object."
        )

    @staticmethod
    def _parse_spatial_map(raw_json: str, scene_desc: str) -> SpatialMap:
        data = json.loads(raw_json)
        objects = []
        for obj in data.get("objects", []):
            bb = obj["bounding_box"]
            objects.append(ObjectTrajectory(
                object_id=obj["id"],
                label=obj["label"],
                t0_position=obj["t0"],
                t_final_position=obj["t_final"],
                bounding_box=BoundingBox(**bb),
                motion_path=obj.get("motion_path", [])))
        return SpatialMap(
            scene_description=scene_desc,
            objects=objects,
            camera_trajectory=data.get("camera_trajectory", []))


# ══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> str:
    """Strip markdown fences and return raw JSON string."""
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    return match.group(1).strip() if match else text.strip()


def _calc_deviation(predicted: dict, actual: dict) -> float:
    """Euclidean deviation as a fraction of predicted vector magnitude."""
    dx = predicted["x"] - actual["x"]
    dy = predicted["y"] - actual["y"]
    dz = predicted["z"] - actual["z"]
    mag = math.sqrt(
        predicted["x"]**2 + predicted["y"]**2 + predicted["z"]**2
    ) or 1.0
    return math.sqrt(dx**2 + dy**2 + dz**2) / mag


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = AetherEngine(
        gemini_api_key="YOUR_GEMINI_KEY",
        grok_api_key="YOUR_GROK_KEY")

    # Stage 1 — map the scene
    spatial_map = engine.map_scene(
        prompt=(
            "A café table at golden hour. A coffee cup sits on the left side. "
            "A notebook lies open on the right. A hand reaches in from frame-right."
        ),
        keyframe_path=None,   # Optional: "starting_frame.jpg"
    )

    # Stage 2 — generate with constraints
    raw_video = engine.generate_constrained_video(
        creative_prompt=(
            "Slow cinematic push-in as steam rises from the cup. "
            "Warm amber light. Intimate, quiet rebellious energy."
        ),
        style_tags=["cinematic", "golden hour", "35mm grain", "melancholic"])

    # Stage 3 — verify and auto-correct
    final_video = engine.validate_consistency(max_correction_passes=3)
    print(f"Final validated video: {final_video}")
