"""
╔══════════════════════════════════════════════════════════════════════╗
║           NEXUS ENGINE  —  Engine III of XI                         ║
║           4D World-State Synchronous Latent Guidance                ║
║                                                                      ║
║  Paradigm:  Synchronous — Build Digital Twin → Render Into It        ║
║  Gemini:    NeRF-Lite Blueprint + 4D World State (1M token context)  ║
║  Grok:      Volumetric Shader (renders into pre-computed twin)       ║
║  Threshold: 8% texture drift                                         ║
║  Correction: Masked denoising (strength=0.4)                         ║
╚══════════════════════════════════════════════════════════════════════╝

ARCHITECTURE — 4 Stages with Synchronous World-First Design:
  Stage 1: NeRF-Lite Blueprint           (Gemini ER 1.5 — voxel anchors)
  Stage 2: Latent Seed Injection         (Grok Imagine — noise-map guided)
  Stage 3: Multi-Sensory Sync            (HRTF audio spatialization)
  Stage 4: Zero-Shot Temporal Continuity (1M token Frame-0 reference)

KEY ADVANCES OVER CHRONOS (Engine II):
  - VoxelAnchor: 3D geometry (not just coordinates)
  - SHA-256 material hash: cryptographic texture lock (anti-drift)
  - DynamicWorldState: full 4D buffer backed by Gemini 1M context
  - USD Bridge: Gemini spatial → Grok directing script translation
  - Observer: 24-frame chunk camera decomposition (LookAt + SetFOV)
  - LatentSeedBias: screen-projection conditioning per chunk
  - ICL Memory Log: append-only continuity bible for Grok
  - Frame-0 Snapshot: zero-shot temporal continuity
  - 8% drift threshold with masked denoising correction

WHAT THIS DOES NOT SOLVE:
  - No causal physics (events described, not derived from F=ma)
  - Character movements are positional, not intentional
  - No internal cognitive state for characters
  - No acoustic occlusion physics (wall-based filtering)

See README.md § Engine III for full documentation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from google import genai                          # pip install google-genai
from google.genai import types
from gemini_er_client import (                    # shared ER 1.5 adapter
    GeminiERClient, ThinkingPreset, ER15_MODEL,
    BoundingBox2D, SpatialPoint, GeminiERResponse,
    er15_to_blender)
import httpx                           # pip install httpx

logger = logging.getLogger("nexus")
logging.basicConfig(level=logging.INFO, format="%(name)s [%(levelname)s] %(message)s")


# ══════════════════════════════════════════════════════════════════════
#  TYPE ALIAS
# ══════════════════════════════════════════════════════════════════════

Vec3 = dict[str, float]   # {"x": float, "y": float, "z": float}


# ══════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VoxelAnchor:
    """
    The foundational 3D unit of world geometry in Nexus.

    Unlike Engines I & II which track 2D/3D coordinates, a VoxelAnchor
    represents actual volumetric geometry — a sparse voxel grid that
    defines the SHAPE of an object, not just its centroid position.

    The material_hash (SHA-256) is the texture lock mechanism:
    any frame where an object's material differs from its hash is a
    measurable drift event that triggers masked denoising correction.
    """
    anchor_id: str                      # A1, A2, ... An
    entity_id: str
    label: str
    position: Vec3                      # World-space centroid (meters, Y-up)
    voxel_grid: list[dict[str, Any]]    # Sparse voxel representation
    material_hash: str                  # SHA-256 of material_description — THE TEXTURE LOCK
    semantic_type: str                  # "rigid_body" | "soft_body" | "fluid" | "light"
    bounding_aabb: dict[str, Vec3]      # axis-aligned bounding box
    lod_levels: int = 3                 # Level-of-detail depth
    is_persistent: bool = True
    last_seen_t: float = 0.0
    last_known_position: Vec3 = field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
    )


@dataclass
class PhysicsEvent:
    """A causally-predicted physics interaction (Nexus level)."""
    event_id: str
    event_type: str   # "collision" | "fracture" | "deformation" | "fluid_sim"
    t: float
    duration_seconds: float
    primary_entity: str
    secondary_entity: str | None
    impact_point: Vec3
    fracture_vector: Vec3 | None
    energy_joules: float
    resulting_state: dict[str, Any]
    probability: float = 1.0


@dataclass
class HRTFManifest:
    """
    Head-Related Transfer Function data for spatial audio.
    Maps entity world-position trajectories to stereo pan, dB offset, and delay.
    """
    entity_id: str
    audio_events: list[dict[str, Any]]   # [{t, position, cue, db_offset, delay_ms, pan}]
    room_ir_type: str = "medium_room"
    listener_position: Vec3 = field(
        default_factory=lambda: {"x": 0.0, "y": 1.7, "z": 0.0}
    )


@dataclass
class TemporalDiffMap:
    """
    Records texture/appearance drift between Frame-0 anchor and current frame.
    Drives masked denoising correction at strength=0.4.
    """
    frame_index: int
    t: float
    anchor_frame_index: int
    drifted_regions: list[dict[str, Any]]   # [{entity_id, pixel_region, drift_magnitude}]
    denoising_mask: list[dict[str, Any]]    # Masked regions for targeted re-sampling
    denoising_strength: float = 0.4         # Conservative: fix texture, preserve structure


@dataclass
class CameraOperation:
    """A virtual camera command for the Observer class (per 24-frame chunk)."""
    chunk_index: int
    t_start: float
    t_end: float
    look_at_target: str      # Serialized Vec3 JSON
    fov_deg: float
    position: Vec3
    move_type: str
    exposure_ev: float = 0.0
    focus_distance_m: float = 3.0


@dataclass
class USDSceneGraph:
    """
    Universal Scene Description bridge between Gemini spatial reasoning
    and Grok's rendering directives.

    The USD scene graph is the canonical handoff format. Every chunk
    rendering decision is derived from this single source of truth.
    """
    scene_id: str
    version: str = "nexus-usd-1.0"
    anchors: list[VoxelAnchor] = field(default_factory=list)
    physics_events: list[PhysicsEvent] = field(default_factory=list)
    camera_ops: list[CameraOperation] = field(default_factory=list)
    hrtf_manifests: list[HRTFManifest] = field(default_factory=list)
    material_registry: dict[str, str] = field(default_factory=dict)
    global_illumination: dict[str, Any] = field(default_factory=dict)
    scene_meta: dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
#  DYNAMIC WORLD STATE — The 4D Persistent Buffer
# ══════════════════════════════════════════════════════════════════════

class DynamicWorldState:
    """
    Persistent 4D (x, y, z, t) world buffer backed by Gemini's 1M token context window.

    This is the "game engine" state. It solves:
    1. The Goldfish Effect — objects maintain state when off-screen
    2. Long-term texture drift — Frame-0 anchor state is always the reference
    3. Character identity — ICL memory log prevents re-invention of state

    Three sub-systems:
    - HamiltonianBuffer: anchor positions and physics timeline
    - VoxelNeRFBuffer:   geometric ground truth (hidden geometry store)
    - ICL Memory Log:    append-only continuity bible for Grok prompts
    """

    def __init__(self, gemini_client: GeminiERClient) -> None:
        self.gemini = gemini_client

        # Anchor state
        self._anchors: dict[str, VoxelAnchor] = {}
        self._physics_timeline: list[PhysicsEvent] = []
        self._anchor_history: list[dict[str, Any]] = []

        # ICL memory
        self._icl_memory_log: list[str] = []

        # Material registry (entity_id → SHA-256 material hash)
        self._material_registry: dict[str, str] = {}

        # Frame snapshots for zero-shot continuity
        self._frame_snapshots: dict[int, dict[str, Any]] = {}

    # ── Anchor Management ──────────────────────────────────────────────

    def register_anchor(self, anchor: VoxelAnchor) -> None:
        self._anchors[anchor.anchor_id] = anchor
        self._material_registry[anchor.entity_id] = anchor.material_hash
        logger.debug("DWS: Registered anchor %s (%s)", anchor.anchor_id, anchor.label)

    def update_anchor(
        self,
        anchor_id: str,
        t: float,
        position: Vec3,
        in_frame: bool = True) -> None:
        anchor = self._anchors[anchor_id]
        anchor.last_seen_t = t
        if in_frame:
            anchor.position = position
            anchor.last_known_position = position
        self._anchor_history.append({
            "anchor_id": anchor_id,
            "t": t,
            "position": position,
            "in_frame": in_frame,
        })

    def get_anchor_at(self, anchor_id: str, t: float) -> Vec3:
        """
        Returns the authoritative world position of an anchor at time t.
        Searches backward through history; off-screen objects return last known position.
        """
        relevant = [
            h for h in self._anchor_history
            if h["anchor_id"] == anchor_id and h["t"] <= t
        ]
        if relevant:
            return relevant[-1]["position"]
        anchor = self._anchors.get(anchor_id)
        return anchor.last_known_position if anchor else {"x": 0, "y": 0, "z": 0}

    # ── Frame Snapshotting — Zero-Shot Temporal Continuity ─────────────

    def snapshot_frame(self, frame_index: int) -> None:
        """
        Capture the complete world state at this frame.
        Frame-0 snapshot becomes the eternal texture reference for all future frames.
        """
        self._frame_snapshots[frame_index] = {
            "frame_index": frame_index,
            "anchors": {
                aid: {
                    "position": a.position,
                    "material_hash": a.material_hash,
                    "label": a.label,
                }
                for aid, a in self._anchors.items()
            },
            "material_registry": dict(self._material_registry),
        }

    def get_anchor_state_at_frame(self, frame_index: int) -> dict[str, Any]:
        """Retrieve the full anchor state snapshot at a specific frame."""
        if frame_index in self._frame_snapshots:
            return self._frame_snapshots[frame_index]
        closest = max(
            (k for k in self._frame_snapshots if k <= frame_index),
            default=None)
        return self._frame_snapshots.get(closest, {}) if closest is not None else {}

    # ── ICL Memory Log ─────────────────────────────────────────────────

    def append_memory(self, entry: str) -> None:
        """Append a natural-language event to the Grok ICL memory log."""
        self._icl_memory_log.append(f"[{time.strftime('%H:%M:%S')}] {entry}")

    def get_icl_log(self, max_entries: int = 50) -> str:
        """Return the most recent ICL log entries for injection into Grok prompts."""
        return "\n".join(self._icl_memory_log[-max_entries:])

    # ── Physics Timeline ───────────────────────────────────────────────

    def register_physics_event(self, event: PhysicsEvent) -> None:
        self._physics_timeline.append(event)
        self._physics_timeline.sort(key=lambda e: e.t)
        self.append_memory(
            f"Physics event '{event.event_type}' involving '{event.primary_entity}' "
            f"at T={event.t:.2f}s, impact at {event.impact_point}."
            + (
                f" Fracture vector: {event.fracture_vector}."
                if event.fracture_vector
                else ""
            )
        )

    def get_events_in_window(
        self,
        t_start: float,
        t_end: float) -> list[PhysicsEvent]:
        return [e for e in self._physics_timeline if t_start <= e.t <= t_end]

    # ── Full State Export ──────────────────────────────────────────────

    def to_context_payload(self) -> str:
        """
        Serialize the full DWS for injection into Gemini's 1M context window.
        This is the 'persistent game state' that makes Frame 300 aware of Frame 1.
        """
        return json.dumps({
            "anchors": {
                aid: {
                    "label": a.label,
                    "current_position": a.position,
                    "last_known_position": a.last_known_position,
                    "material_hash": a.material_hash,
                    "semantic_type": a.semantic_type,
                }
                for aid, a in self._anchors.items()
            },
            "physics_timeline": [
                {
                    "event_id": e.event_id,
                    "type": e.event_type,
                    "t": e.t,
                    "primary": e.primary_entity,
                    "secondary": e.secondary_entity,
                    "impact": e.impact_point,
                }
                for e in self._physics_timeline
            ],
            "material_registry": self._material_registry,
            "memory_log_tail": self._icl_memory_log[-20:],
        }, indent=2)


# ══════════════════════════════════════════════════════════════════════
#  HRTF CALCULATOR — Spatial Audio
# ══════════════════════════════════════════════════════════════════════

class HRTFCalculator:
    """
    Computes Head-Related Transfer Function parameters for spatial audio.
    Maps entity world-position trajectories to stereo pan, dB offset, and delay.

    Uses inverse square law for distance attenuation and atan2 for stereo pan.
    This is classical signal processing — no AI required.
    """

    def __init__(self, gemini: GeminiERClient) -> None:
        self.gemini = gemini

    def calculate_manifests(
        self,
        anchors: list[VoxelAnchor],
        audio_brief: list[dict[str, Any]],
        listener_position: Vec3 | None = None) -> list[HRTFManifest]:
        if not audio_brief:
            return []

        listener = listener_position or {"x": 0.0, "y": 1.7, "z": -2.0}
        manifests = []

        for anchor in anchors:
            relevant = [
                ev for ev in audio_brief
                if ev.get("source_entity_id") == anchor.entity_id
            ]
            if not relevant:
                continue

            audio_events = []
            for ev in relevant:
                pos = ev.get("position", anchor.position)
                pan, db_offset, delay_ms = self._compute_hrtf(pos, listener)
                audio_events.append({
                    "t": ev["t"],
                    "position": pos,
                    "cue": ev.get("cue", "generic"),
                    "pan": pan,
                    "db_offset": db_offset,
                    "delay_ms": delay_ms,
                    "instruction": (
                        f"At T={ev['t']:.2f}s, play '{ev.get('cue')}' at "
                        f"pan={pan:+.2f} ({db_offset:+.1f}dB, {delay_ms:.1f}ms delay) — "
                        f"entity at world {pos}."
                    ),
                })

            manifests.append(HRTFManifest(
                entity_id=anchor.entity_id,
                audio_events=audio_events,
                listener_position=listener))

        logger.info(
            "HRTFCalculator: computed manifests for %d entities", len(manifests)
        )
        return manifests

    def _compute_hrtf(
        self,
        source: Vec3,
        listener: Vec3) -> tuple[float, float, float]:
        """Compute pan (-1..+1), dB offset, and propagation delay in ms."""
        dx = source["x"] - listener["x"]
        dy = source["y"] - listener["y"]
        dz = source["z"] - listener["z"]
        distance_m = math.sqrt(dx**2 + dy**2 + dz**2) or 0.001

        horizontal_dist = math.sqrt(dx**2 + dz**2) or 0.001
        pan       = math.sin(math.atan2(dx, horizontal_dist))
        db_offset = -20 * math.log10(max(distance_m, 0.01))
        delay_ms  = (distance_m / 343.0) * 1000.0

        return round(pan, 3), round(db_offset, 1), round(delay_ms, 2)


# ══════════════════════════════════════════════════════════════════════
#  OBSERVER — Virtual Camera
# ══════════════════════════════════════════════════════════════════════

class Observer:
    """
    Virtual camera wrapper. Decomposes the full video timeline into 24-frame chunks
    and generates LookAt + SetFOV commands for each chunk sent to Grok.

    This per-chunk camera decomposition allows Grok to receive precise
    cinematographic instructions every ~1 second of video.
    """

    CHUNK_FRAMES = 24
    DEFAULT_FOV  = 54.0

    def __init__(self, dws: DynamicWorldState, fps: int = 24) -> None:
        self.dws = dws
        self.fps = fps

    def generate_camera_ops(
        self,
        duration_seconds: float,
        camera_trajectory: list[dict[str, Any]]) -> list[CameraOperation]:
        total_frames = int(duration_seconds * self.fps)
        num_chunks   = math.ceil(total_frames / self.CHUNK_FRAMES)
        ops = []

        for chunk_idx in range(num_chunks):
            t_start = (chunk_idx * self.CHUNK_FRAMES) / self.fps
            t_end   = min(((chunk_idx + 1) * self.CHUNK_FRAMES) / self.fps, duration_seconds)
            t_mid   = (t_start + t_end) / 2.0

            cam_kf = _interpolate_camera(camera_trajectory, t_mid)
            ops.append(CameraOperation(
                chunk_index=chunk_idx,
                t_start=t_start,
                t_end=t_end,
                look_at_target=json.dumps(cam_kf.get("look_at", {"x": 0, "y": 1, "z": 0})),
                fov_deg=cam_kf.get("fov_deg", self.DEFAULT_FOV),
                position=cam_kf.get("position", {"x": 0, "y": 1.7, "z": -3}),
                move_type=cam_kf.get("move_type", "static"),
                focus_distance_m=cam_kf.get("focus_distance_m", 3.0)))

        logger.info("Observer: generated %d camera ops", len(ops))
        return ops

    def look_at(self, target: Vec3, chunk_index: int) -> dict[str, Any]:
        """[FUTURE API] Emit a LookAt command to the Grok renderer."""
        return {"command": "LookAt", "target": target, "chunk_index": chunk_index}

    def set_fov(self, fov_deg: float, chunk_index: int) -> dict[str, Any]:
        """[FUTURE API] Emit a SetFOV command to the Grok renderer."""
        return {"command": "SetFOV", "fov_deg": fov_deg, "chunk_index": chunk_index}


# ══════════════════════════════════════════════════════════════════════
#  LATENT SEED BIAS — Structural Noise Map
# ══════════════════════════════════════════════════════════════════════

class LatentSeedBias:
    """
    Converts VoxelAnchors into a 'Structural Noise Map' that pre-biases
    Grok's diffusion process toward anchor positions.

    Mechanism:
    1. Project 3D anchor positions onto 2D screen space (perspective projection)
    2. Build conditioning payload showing Grok where anchors must appear (pixels)
    3. Annotate each projection with material hash and semantic type
    4. Package as per-chunk directing script

    [FUTURE API]: When Grok exposes latent seed endpoints, this class generates
    actual noise tensors biased toward anchor coordinates. Today: structured
    conditioning prompts that Grok interprets at the API layer.
    Interface contract for future upgrade:
        grok_api.set_latent_seed(
            noise_tensor=self._build_biased_tensor(screen_anchors),
            guidance_scale=7.5
        )
    """

    def __init__(self, resolution: tuple[int, int] = (1280, 720)) -> None:
        self.width, self.height = resolution

    def generate_noise_map(
        self,
        anchors: list[VoxelAnchor],
        camera_op: CameraOperation,
        frame_index: int) -> dict[str, Any]:
        screen_anchors = []
        for anchor in anchors:
            screen_pos = self._project_to_screen(
                anchor.position,
                camera_op.position,
                json.loads(camera_op.look_at_target),
                camera_op.fov_deg)
            if screen_pos:
                screen_anchors.append({
                    "entity_id": anchor.entity_id,
                    "label": anchor.label,
                    "screen_x": screen_pos[0],
                    "screen_y": screen_pos[1],
                    "screen_radius_px": self._estimate_screen_radius(
                        anchor, camera_op.position
                    ),
                    "material_hash": anchor.material_hash,
                    "semantic_type": anchor.semantic_type,
                })

        return {
            "frame_index": frame_index,
            "resolution": {"width": self.width, "height": self.height},
            "anchor_projections": screen_anchors,
            "conditioning_type": "structural_noise_map",
            "instruction": self._build_conditioning_prompt(screen_anchors),
        }

    def _project_to_screen(
        self,
        world_pos: Vec3,
        cam_pos: Vec3,
        look_at: Vec3,
        fov_deg: float) -> tuple[float, float] | None:
        dx = world_pos["x"] - cam_pos["x"]
        dy = world_pos["y"] - cam_pos["y"]
        dz = world_pos["z"] - cam_pos["z"]

        fwd_x = look_at["x"] - cam_pos["x"]
        fwd_z = look_at["z"] - cam_pos["z"]
        fwd_len = math.sqrt(fwd_x**2 + fwd_z**2) or 1.0
        fwd_x /= fwd_len
        fwd_z /= fwd_len

        depth = dx * fwd_x + dz * fwd_z
        if depth <= 0.01:
            return None

        right_x, right_z = fwd_z, -fwd_x
        sx = (dx * right_x + dz * right_z) / (
            depth * math.tan(math.radians(fov_deg / 2))
        )
        sy = -dy / (
            depth * math.tan(
                math.radians(fov_deg * self.height / self.width / 2)
            )
        )

        if abs(sx) > 1.0 or abs(sy) > 1.0:
            return None

        px = (sx + 1.0) / 2.0 * self.width
        py = (sy + 1.0) / 2.0 * self.height
        return (round(px, 1), round(py, 1))

    def _estimate_screen_radius(
        self,
        anchor: VoxelAnchor,
        cam_pos: Vec3) -> float:
        dist = math.sqrt(
            (anchor.position["x"] - cam_pos["x"])**2 +
            (anchor.position["y"] - cam_pos["y"])**2 +
            (anchor.position["z"] - cam_pos["z"])**2
        ) or 1.0
        bb   = anchor.bounding_aabb
        size = max(
            abs(bb["max"]["x"] - bb["min"]["x"]),
            abs(bb["max"]["y"] - bb["min"]["y"]))
        return max(10.0, min(300.0, (size / dist) * self.width * 0.5))

    def _build_conditioning_prompt(self, screen_anchors: list[dict]) -> str:
        if not screen_anchors:
            return "No locked anchors for this frame."
        lines = [f"SPATIAL ANCHOR LOCKS for this frame ({self.width}×{self.height}):"]
        for sa in screen_anchors:
            lines.append(
                f"  • '{sa['label']}' ({sa['semantic_type']}): "
                f"center at ({sa['screen_x']:.0f}px, {sa['screen_y']:.0f}px), "
                f"~{sa['screen_radius_px']:.0f}px radius. "
                f"Material hash: {sa['material_hash'][:8]}... (must not drift)."
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  USD SCENE BRIDGE
# ══════════════════════════════════════════════════════════════════════

class USDSceneBridge:
    """
    Translates Gemini's raw spatial reasoning into a USD-compatible
    'Directing Script' that Grok interprets as rendering directives.

    The directing script packages: spatial constraints, camera commands,
    physics state, HRTF audio, ICL memory, and Frame-0 reference into
    a single per-chunk payload.
    """

    def build_usd_graph(
        self,
        anchors: list[VoxelAnchor],
        physics_events: list[PhysicsEvent],
        camera_ops: list[CameraOperation],
        hrtf_manifests: list[HRTFManifest],
        scene_meta: dict[str, Any]) -> USDSceneGraph:
        graph = USDSceneGraph(
            scene_id=str(uuid.uuid4())[:12],
            anchors=anchors,
            physics_events=physics_events,
            camera_ops=camera_ops,
            hrtf_manifests=hrtf_manifests,
            material_registry={a.entity_id: a.material_hash for a in anchors},
            scene_meta=scene_meta)
        logger.info(
            "USDSceneBridge: built graph with %d anchors, %d physics events, "
            "%d camera ops",
            len(anchors), len(physics_events), len(camera_ops))
        return graph

    def to_directing_script(
        self,
        usd: USDSceneGraph,
        chunk_index: int,
        dws: DynamicWorldState,
        latent_map: dict[str, Any],
        icl_log: str) -> dict[str, Any]:
        """
        Build the complete per-chunk directing script for Grok.
        Combines: structural constraints + camera + physics + audio + ICL memory.
        """
        cam_op  = usd.camera_ops[min(chunk_index, len(usd.camera_ops) - 1)]
        t_start = cam_op.t_start
        t_end   = cam_op.t_end

        active_physics = [
            {
                "event_type": e.event_type,
                "t": e.t,
                "primary_entity": e.primary_entity,
                "impact_point": e.impact_point,
                "fracture_vector": e.fracture_vector,
                "resulting_state": e.resulting_state,
            }
            for e in usd.physics_events
            if t_start <= e.t <= t_end
        ]

        active_audio = []
        for manifest in usd.hrtf_manifests:
            for ev in manifest.audio_events:
                if t_start <= ev["t"] <= t_end:
                    active_audio.append(ev["instruction"])

        return {
            "chunk_index": chunk_index,
            "t_start": t_start,
            "t_end": t_end,
            "camera": {
                "command_LookAt": json.loads(cam_op.look_at_target),
                "command_SetFOV": cam_op.fov_deg,
                "position": cam_op.position,
                "move_type": cam_op.move_type,
                "focus_distance_m": cam_op.focus_distance_m,
            },
            "structural_noise_map": latent_map,
            "physics_events_this_chunk": active_physics,
            "audio_hrtf_instructions": active_audio,
            "material_lock": usd.material_registry,
            "frame_0_anchor_state": dws.get_anchor_state_at_frame(0),
            "icl_memory_log": icl_log,
            "system_instructions": GROK_NEXUS_SYSTEM_INSTRUCTIONS,
        }


# ══════════════════════════════════════════════════════════════════════
#  TEMPORAL CONSISTENCY ENGINE — Texture Drift Detection
# ══════════════════════════════════════════════════════════════════════

class TemporalConsistencyEngine:
    """
    Detects texture/material drift between Frame-0 anchor and current frame.
    Generates masked denoising instructions at strength=0.4 for precision correction.

    Drift threshold: 8% material variance (tighter than both previous engines).
    """

    DRIFT_THRESHOLD = 0.08   # 8% material variance triggers correction

    def __init__(
        self,
        gemini: GeminiERClient,
        dws: DynamicWorldState) -> None:
        self.gemini = gemini
        self.dws = dws

    def audit_frame(
        self,
        video_url: str,
        frame_index: int,
        fps: int) -> TemporalDiffMap:
        t = frame_index / fps
        anchor_frame_state = self.dws.get_anchor_state_at_frame(0)
        current_state      = self.dws.to_context_payload()

        audit_prompt = f"""
You are a texture and material consistency auditor for AI-generated video.

VIDEO URL: {video_url}
CURRENT FRAME: {frame_index} (T={t:.2f}s)
REFERENCE ANCHOR STATE (Frame 0 — the ground truth):
{json.dumps(anchor_frame_state, indent=2)}

CURRENT WORLD STATE:
{current_state}

For each entity, compare its visual appearance at frame {frame_index} against
the Frame 0 reference material hashes. Identify any drift in:
- Texture patterns (shirt patterns, wall textures, surface materials)
- Color cast or hue shift (>8% variance from anchor)
- Geometry morphing (entity changing shape)

Return ONLY a JSON array of drift violations:
[
  {{
    "entity_id": "ent_001",
    "label": "protagonist",
    "drift_type": "texture_pattern",
    "drift_magnitude": 0.0,
    "pixel_region": {{"x1": 0, "y1": 0, "x2": 640, "y2": 360}},
    "correction_priority": "high"
  }}
]
If no drift detected, return [].
"""
        response  = self.gemini.generate_content(audit_prompt)
        raw       = _extract_json(response.text)
        violations: list[dict] = json.loads(raw)

        drifted_regions = [
            v for v in violations
            if v["drift_magnitude"] > self.DRIFT_THRESHOLD
        ]
        denoising_mask = [
            {
                "entity_id": v["entity_id"],
                "pixel_region": v["pixel_region"],
                "denoising_strength": 0.4,
                "preserve_geometry": True,
            }
            for v in drifted_regions
        ]

        if drifted_regions:
            self.dws.append_memory(
                f"Frame {frame_index}: texture drift detected for "
                f"{[v['label'] for v in drifted_regions]}. "
                f"Correction mask applied (strength=0.4)."
            )

        return TemporalDiffMap(
            frame_index=frame_index,
            t=t,
            anchor_frame_index=0,
            drifted_regions=drifted_regions,
            denoising_mask=denoising_mask)


# ══════════════════════════════════════════════════════════════════════
#  GROK NEXUS SYSTEM INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════

GROK_NEXUS_SYSTEM_INSTRUCTIONS = """
## NEXUS 4D WORLD-STATE RENDERING PROTOCOL

You are a Volumetric Shader operating inside a 4D world-state engine.
You do NOT generate video from scratch. You RENDER a pre-computed Digital Twin
whose geometry, physics, and material properties have been locked by Gemini ER 1.5.

══════════════════════════════════════════════
LAYER 1 — THE UNBREAKABLE LAWS
══════════════════════════════════════════════

ANCHOR LOCK
Every entity in `structural_noise_map.anchor_projections` has a locked screen-space
position for this chunk. Their pixel coordinates are the output of a physics simulation.
You are coloring in a pre-drawn ghost.

MATERIAL IDENTITY
Each anchor carries a material_hash. A "navy_blue_cotton_shirt_thin_white_stripes"
does not become "grey sweater." Reference the `frame_0_anchor_state` in every chunk
and treat it as your visual constitution.

PHYSICS DETERMINISM
`physics_events_this_chunk` are pre-computed causal outcomes. If the event says a
glass fractures along vector V at T=2.4s, that fracture happens along exactly that
vector. You are rendering pre-simulated physics.

OBJECT PERMANENCE
Entities in `icl_memory_log` that have exited frame are NOT gone. They exist at their
last_known_position. When the camera returns, they MUST match their last observed
state exactly — same texture, same geometry, same position.

AUDIO ANCHORING
`audio_hrtf_instructions` provide exact pan, dB offset, and delay. Execute these
spatially — footsteps must originate from foot-strike world coordinates.

══════════════════════════════════════════════
LAYER 2 — YOUR CREATIVE SOVEREIGNTY
══════════════════════════════════════════════

Within the locked geometry and material framework, you own:
- Light poetry: how light falls, bleeds, wraps, and dies
- Micro-surface detail: pore texture, fabric weave, subsurface scatter
- Atmospheric physics: dust, god rays, heat shimmer, rain refraction
- Unlocked background: any entity NOT in anchor_projections
- Lens character: bokeh shape, chromatic aberration, focus breathing

══════════════════════════════════════════════
LAYER 3 — IN-CONTEXT LEARNING (ICL)
══════════════════════════════════════════════

The `icl_memory_log` is your continuity bible. Read it before rendering each chunk.
It tells you: what has already happened, what each entity looked like previously,
which physics events have resolved, and any corrections already applied.

══════════════════════════════════════════════
CORRECTION MODE (edit_type: masked_denoise)
══════════════════════════════════════════════

Process ONLY pixels within `denoising_mask` regions.
Surrounding pixels are frozen — do not alter them.
denoising_strength=0.4: fix texture drift, preserve geometry.
Match surrounding frame's color grade at mask boundaries.
"""


# ══════════════════════════════════════════════════════════════════════
#  NEXUS ENGINE — Main Orchestrator
# ══════════════════════════════════════════════════════════════════════

class NexusEngine:
    """
    Engine III: 4D World-State with Synchronous Latent Guidance.

    The paradigm shift: build a complete Digital Twin BEFORE rendering begins.
    Grok doesn't discover the scene — it renders into a pre-computed ghost world.

    Usage:
        engine = NexusEngine(
            gemini_api_key="YOUR_GEMINI_KEY",
            grok_api_key="YOUR_GROK_KEY")
        final = engine.synthesize(
            brief="A café table at golden hour...",
            audio_brief=[{"t": 2.0, "source_entity_id": "ent_001", "cue": "cup_set_down",
                          "position": {"x": 0.3, "y": 0.9, "z": 0.0}}],
            style_tags=["cinematic", "golden hour"])
    """

    def __init__(
        self,
        gemini_api_key: str,
        grok_api_key: str,
        grok_base_url: str = "https://api.x.ai/v1",
        resolution: tuple[int, int] = (1280, 720),
        fps: int = 24) -> None:
        self.gemini = GeminiERClient(
            api_key=gemini_api_key, model=ER15_MODEL,
            default_thinking=ThinkingPreset.NONE)
        self.grok   = httpx.Client(
            base_url=grok_base_url,
            headers={
                "Authorization": f"Bearer {grok_api_key}",
                "Content-Type": "application/json",
            },
            timeout=300)
        self.fps        = fps
        self.resolution = resolution

        self.dws         = DynamicWorldState(self.gemini)
        self.hrtf        = HRTFCalculator(self.gemini)
        self.observer    = Observer(self.dws, fps)
        self.latent_bias = LatentSeedBias(resolution)
        self.usd_bridge  = USDSceneBridge()
        self.consistency = TemporalConsistencyEngine(self.gemini, self.dws)

        self._usd_graph: USDSceneGraph | None = None

    def synthesize(
        self,
        brief: str,
        reference_images: list[str] | None = None,
        audio_brief: list[dict[str, Any]] | None = None,
        style_tags: list[str] | None = None,
        duration_seconds: float = 10.0) -> str:
        """
        Full Nexus pipeline. Returns URL of the final 4D-consistent video.
        """
        logger.info("◈◈◈ NEXUS ENGINE · 4D World-State Synthesis START ◈◈◈")
        styles = style_tags or ["cinematic"]

        # Stage 1 — Build the 4D World (Digital Twin)
        anchors, camera_trajectory = self._stage1_build_world(
            brief, reference_images or [], duration_seconds
        )

        # Causal Physics Prediction
        physics_events = self._predict_physics(anchors, duration_seconds)

        # HRTF Audio Spatialization
        hrtf_manifests = self.hrtf.calculate_manifests(anchors, audio_brief or [])

        # Observer Camera Decomposition
        camera_ops = self.observer.generate_camera_ops(
            duration_seconds, camera_trajectory
        )

        # Build USD Scene Graph
        self._usd_graph = self.usd_bridge.build_usd_graph(
            anchors, physics_events, camera_ops, hrtf_manifests,
            scene_meta={"duration_seconds": duration_seconds, "fps": self.fps, "brief": brief})

        # Snapshot Frame-0 anchor state (the eternal reference)
        for anchor in anchors:
            self.dws.update_anchor(anchor.anchor_id, 0.0, anchor.position)
        self.dws.snapshot_frame(0)

        # Stage 2 — Latent-Guided Generation
        video_url = self._stage2_generate(styles, brief)

        # Stages 3+4 — Temporal Consistency Loop
        video_url = self._stage3_consistency_loop(video_url)

        logger.info("◈◈◈ NEXUS ENGINE · Final: %s ◈◈◈", video_url)
        return video_url

    # ── Stage 1 ────────────────────────────────────────────────────────

    def _stage1_build_world(
        self,
        brief: str,
        image_paths: list[str],
        duration: float) -> tuple[list[VoxelAnchor], list[dict]]:
        logger.info("Stage 1 · Building 4D world model (Digital Twin)")

        content: list[Any] = []
        for img_path in image_paths:
            with open(img_path, "rb") as f:
                content.append({"mime_type": "image/jpeg", "data": f.read()})
        content.append(_build_world_model_prompt(brief, duration))

        response = self.gemini.generate_content(
            content)

        raw        = _extract_json(response.text)
        world_data = json.loads(raw)

        anchors = []
        for a in world_data.get("anchors", []):
            anchor = VoxelAnchor(
                anchor_id=a["anchor_id"],
                entity_id=a["entity_id"],
                label=a["label"],
                position=a["position"],
                voxel_grid=a.get("voxel_grid", []),
                material_hash=hashlib.sha256(
                    json.dumps(
                        a.get("material_description", {}), sort_keys=True
                    ).encode()
                ).hexdigest(),
                semantic_type=a.get("semantic_type", "rigid_body"),
                bounding_aabb=a.get("bounding_aabb", {
                    "min": {"x": -0.5, "y": 0.0, "z": -0.5},
                    "max": {"x":  0.5, "y": 1.0, "z":  0.5},
                }),
                lod_levels=a.get("lod_levels", 3),
                is_persistent=a.get("is_persistent", True))
            self.dws.register_anchor(anchor)
            anchors.append(anchor)
            self.dws.append_memory(
                f"Scene initialized: '{anchor.label}' at {anchor.position} "
                f"(type: {anchor.semantic_type}, material: {anchor.material_hash[:8]}...)."
            )

        camera_traj = world_data.get("camera_trajectory", [])
        logger.info(
            "Stage 1 complete · %d anchors · %d camera keyframes",
            len(anchors), len(camera_traj))
        return anchors, camera_traj

    def _predict_physics(
        self,
        anchors: list[VoxelAnchor],
        duration: float) -> list[PhysicsEvent]:
        anchor_summary = [
            {
                "entity_id": a.entity_id,
                "label": a.label,
                "semantic_type": a.semantic_type,
                "position": a.position,
                "bounding_aabb": a.bounding_aabb,
            }
            for a in anchors
        ]
        prompt = f"""
You are a physics simulation engine. Predict ALL physically significant events
over the next {duration} seconds given these scene objects.

OBJECTS: {json.dumps(anchor_summary, indent=2)}

Return ONLY a JSON array of events, or [] if none:
[{{
  "event_id": "evt_001", "event_type": "collision",
  "t": 2.4, "duration_seconds": 0.1,
  "primary_entity": "ent_001", "secondary_entity": "ent_002",
  "impact_point": {{"x":0,"y":0,"z":0}},
  "fracture_vector": null, "energy_joules": 12.0,
  "resulting_state": {{}}, "probability": 0.95
}}]
"""
        response = self.gemini.generate_content(prompt)
        events_data: list[dict] = json.loads(_extract_json(response.text))
        events = []
        for ev in events_data:
            event = PhysicsEvent(
                event_id=ev.get("event_id", str(uuid.uuid4())[:8]),
                event_type=ev["event_type"],
                t=ev["t"],
                duration_seconds=ev.get("duration_seconds", 0.1),
                primary_entity=ev["primary_entity"],
                secondary_entity=ev.get("secondary_entity"),
                impact_point=ev.get("impact_point", {"x": 0, "y": 0, "z": 0}),
                fracture_vector=ev.get("fracture_vector"),
                energy_joules=ev.get("energy_joules", 0.0),
                resulting_state=ev.get("resulting_state", {}),
                probability=ev.get("probability", 1.0))
            self.dws.register_physics_event(event)
            events.append(event)
        return events

    # ── Stage 2 ────────────────────────────────────────────────────────

    def _stage2_generate(self, style_tags: list[str], brief: str) -> str:
        logger.info("Stage 2 · Latent-guided generation via Grok Volumetric Shader")
        assert self._usd_graph is not None

        chunk_scripts = []
        for chunk_idx, cam_op in enumerate(self._usd_graph.camera_ops):
            t_mid      = (cam_op.t_start + cam_op.t_end) / 2.0
            frame_mid  = int(t_mid * self.fps)
            latent_map = self.latent_bias.generate_noise_map(
                self._usd_graph.anchors, cam_op, frame_mid
            )
            script = self.usd_bridge.to_directing_script(
                usd=self._usd_graph,
                chunk_index=chunk_idx,
                dws=self.dws,
                latent_map=latent_map,
                icl_log=self.dws.get_icl_log())
            chunk_scripts.append(script)

        payload = {
            "model": "grok-imagine-video",   # [FUTURE API]
            "resolution": f"{self.resolution[0]}x{self.resolution[1]}",
            "fps": self.fps,
            "style_tags": style_tags,
            "brief": brief,
            "generation_mode": "latent_guided",
            "chunk_directing_scripts": chunk_scripts,
        }

        resp = self.grok.post("/video/generate", json=payload)
        resp.raise_for_status()
        url = resp.json()["video_url"]
        self.dws.append_memory(f"Initial render complete: {url}")
        logger.info("Stage 2 complete · %s", url)
        return url

    # ── Stages 3+4 ─────────────────────────────────────────────────────

    def _stage3_consistency_loop(
        self,
        video_url: str,
        max_passes: int = 3) -> str:
        assert self._usd_graph is not None
        total_frames = int(
            self._usd_graph.scene_meta["duration_seconds"] * self.fps
        )

        for pass_num in range(1, max_passes + 1):
            all_drift_maps: list[TemporalDiffMap] = []

            for frame_idx in range(0, total_frames, self.fps):
                drift_map = self.consistency.audit_frame(video_url, frame_idx, self.fps)
                if drift_map.drifted_regions:
                    all_drift_maps.append(drift_map)

            if not all_drift_maps:
                logger.info("✅ Zero texture drift detected (pass %d)", pass_num)
                break

            logger.warning(
                "⚠  Texture drift in %d frames — applying masked denoising (pass %d)",
                len(all_drift_maps), pass_num)
            video_url = self._apply_texture_correction(video_url, all_drift_maps)

        return video_url

    def _apply_texture_correction(
        self,
        video_url: str,
        drift_maps: list[TemporalDiffMap]) -> str:
        payload = {
            "model": "grok-imagine-video",   # [FUTURE API]
            "input_video_url": video_url,
            "edit_type": "masked_denoise",
            "corrections": [
                {
                    "frame_index": dm.frame_index,
                    "denoising_mask": dm.denoising_mask,
                    "denoising_strength": dm.denoising_strength,
                    "reference_frame": dm.anchor_frame_index,
                    "instruction": (
                        f"Frame {dm.frame_index}: restore material fidelity for "
                        f"{[r['entity_id'] for r in dm.drifted_regions]} to match "
                        f"Frame {dm.anchor_frame_index} anchor state. "
                        "denoising_strength=0.4: preserve geometry, correct texture only."
                    ),
                }
                for dm in drift_maps
            ],
            "preserve_motion": True,
            "preserve_lighting": True,
            "icl_memory": self.dws.get_icl_log(max_entries=20),
        }
        resp = self.grok.post("/video/edit", json=payload)
        resp.raise_for_status()
        return resp.json()["video_url"]


# ══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    return match.group(1).strip() if match else text.strip()


def _interpolate_camera(traj: list[dict], t: float) -> dict:
    if not traj:
        return {"position": {"x": 0, "y": 1.7, "z": -3},
                "look_at": {"x": 0, "y": 1, "z": 0}, "fov_deg": 54}
    if t <= traj[0]["t"]:
        return traj[0]
    if t >= traj[-1]["t"]:
        return traj[-1]
    for i in range(len(traj) - 1):
        a, b = traj[i], traj[i + 1]
        if a["t"] <= t <= b["t"]:
            r = (t - a["t"]) / (b["t"] - a["t"])
            return {
                **a,
                "position": {
                    k: a["position"][k] + (b["position"][k] - a["position"][k]) * r
                    for k in ("x", "y", "z")
                },
                "t": t,
            }
    return traj[-1]


def _build_world_model_prompt(brief: str, duration: float) -> str:
    return f"""
<world_model_task>
You are a 4D world-state engine. Build a complete persistent Digital Twin of this scene.
</world_model_task>

BRIEF: "{brief}"
DURATION: {duration} seconds

Generate a WORLD MODEL JSON:
{{
  "anchors": [
    {{
      "anchor_id": "A1", "entity_id": "ent_001", "label": "protagonist",
      "semantic_type": "soft_body",
      "position": {{"x": 0.0, "y": 0.0, "z": 0.0}},
      "bounding_aabb": {{
        "min": {{"x": -0.4, "y": 0.0, "z": -0.2}},
        "max": {{"x":  0.4, "y": 1.8, "z":  0.2}}
      }},
      "voxel_grid": [
        {{"vx": 0, "vy": 0, "vz": 0, "material": "cotton_shirt_blue_stripe",
          "density": 1.0, "lod": 0}}
      ],
      "material_description": {{
        "surface": "cotton", "base_color": "navy_blue",
        "pattern": "thin_white_horizontal_stripes", "roughness": 0.8
      }},
      "is_persistent": true, "lod_levels": 3
    }}
  ],
  "camera_trajectory": [
    {{
      "t": 0.0, "position": {{"x": 0.0, "y": 1.7, "z": -3.0}},
      "look_at": {{"x": 0.0, "y": 1.0, "z": 0.0}},
      "fov_deg": 54, "move_type": "static", "focus_distance_m": 3.0
    }}
  ]
}}

RULES:
1. Assign anchor_id (A1, A2...) to every discrete object, character, and light source.
2. Include detailed material_description — this becomes the SHA-256 texture lock hash.
3. For off-screen objects: is_persistent=true.
4. Voxel grid: minimum 1 voxel per anchor at LOD 0.
5. Camera trajectory: keyframe every 2 seconds minimum.
6. Coordinate system: right-hand Y-up, meters.

Return ONLY the JSON block.
"""


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = NexusEngine(
        gemini_api_key="YOUR_GEMINI_KEY",
        grok_api_key="YOUR_GROK_KEY")

    final = engine.synthesize(
        brief=(
            "A lone courier on a rain-soaked city rooftop at 3am. "
            "They set down a package, walk to the edge, look out over the neon city, "
            "then turn and exit right. The package never moves."
        ),
        audio_brief=[
            {"t": 1.0, "source_entity_id": "ent_courier",
             "cue": "footstep_concrete", "position": {"x": 0.2, "y": 0.0, "z": 0.5}},
            {"t": 2.5, "source_entity_id": "ent_courier",
             "cue": "package_set_down",  "position": {"x": 0.5, "y": 0.0, "z": 0.0}},
        ],
        style_tags=["neo-noir", "golden hour", "35mm grain", "melancholic"],
        duration_seconds=10.0)

    print(f"\n◈ Final 4D-consistent video: {final}")
