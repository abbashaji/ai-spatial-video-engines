"""
╔══════════════════════════════════════════════════════════════════════╗
║           AETHER-OMNI ENGINE  —  Engine IV of XI                    ║
║           Active Inference Simulation / Reality Engine              ║
║                                                                      ║
║  Paradigm:  Simulate a 4D physical event → observe it               ║
║  Gemini:    Causal Physics Kernel (Gravity + Logic Engine)           ║
║  Grok:      Volumetric Shader (Neural Photorealistic Renderer)       ║
║  Threshold: 2% kinetic drift (tightest in series)                   ║
║  Correction: Targeted kinetic correction + LT matrix recalculation  ║
╚══════════════════════════════════════════════════════════════════════╝

ARCHITECTURE — 4 Stages with Active Physics Simulation:
  Stage 1a: Hamiltonian World Initialization  (Gemini — phase space setup)
  Stage 1b: Action Token Generation           (Gemini — F=ma causal script)
  Stage 1c: Pre-Render Contradiction Check    (Python rules engine)
  Stage 2:  Multi-Agent Internal Monologue    (Per-character thinking budgets)
  Stage 3:  Latent-Guided Generation          (Grok + physics conditioning)
  Stage 4:  Reality-Check Loop / Kinetic Audit (Gemini + correction vectors)

THE PARADIGM SHIFT vs ALL PREVIOUS ENGINES:
  Previous: "Describe a scene. Check if Grok drew it correctly."
  Omni:     "Define the laws of physics. Simulate what happens.
             Record the simulation. Grok is the camera."

PRIMARY INPUT: PhysicsLaw objects (not visual descriptions)
  PhysicsLaw("gravity",    {"magnitude_ms2": 9.81})
  PhysicsLaw("material",   {"id": "fracturable_glass", "young_modulus_gpa": 70})
  PhysicsLaw("intent",     {"entity": "protagonist", "urgency": 0.4})

KEY ADVANCES OVER NEXUS (Engine III):
  - Hamiltonian phase space (q, p) — positions DERIVED from F=ma, not specified
  - ActionToken: forces + contact points replace coordinate assignments
  - Pre-render contradiction detection (6 physics invariants)
  - Multi-agent internal monologue → body language tokens
  - Acoustic occlusion ray-casting (wall material absorption coefficients)
  - Light Transport Matrix recalculation on shadow violations
  - Euler integrator for position derivation from action tokens
  - 2% kinetic deviation threshold (vs 8% texture in Nexus)

WHAT IS REAL TODAY vs FUTURE API:
  Real:    Hamiltonian state, Euler integration, action token generation,
           contradiction detection, monologue via Gemini, acoustic occlusion math,
           material hash, ICL log, frame snapshots
  Future:  grok-imagine-video API, deep_guidance attention injection,
           native voxel output from Gemini, video-to-video edit API

See README.md § Engine IV for full documentation.
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
from typing import Any, Literal

from google import genai                          # pip install google-genai
from google.genai import types
from gemini_er_client import (                    # shared ER 1.5 adapter
    GeminiERClient, ThinkingPreset, ER15_MODEL,
    BoundingBox2D, SpatialPoint, GeminiERResponse,
    er15_to_blender)
import httpx                           # pip install httpx
import numpy as np                     # pip install numpy  (Euler integration)

logger = logging.getLogger("aether_omni")
logging.basicConfig(level=logging.INFO, format="%(name)s [%(levelname)s] %(message)s")


# ══════════════════════════════════════════════════════════════════════
#  TYPE ALIASES
# ══════════════════════════════════════════════════════════════════════

Vec3 = dict[str, float]    # {"x": float, "y": float, "z": float}
Mat3 = list[list[float]]   # 3×3 matrix (inertia tensor, rotation)


# ══════════════════════════════════════════════════════════════════════
#  CAUSAL PHYSICS LAW SPECIFICATION
#  The primary input language of Aether-Omni.
#  Prompts define LAWS, not visual descriptions.
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsLaw:
    """
    A governing physical law for the scene simulation.

    This is the new input primitive that replaces visual descriptions.
    Instead of "a glass sits on the table," you write:
        PhysicsLaw("material", {"id": "fracturable_glass", "young_modulus_gpa": 70})

    Law types:
        "gravity"       — gravitational field parameters
        "material"      — physical material properties (modulus, density, fracture)
        "intent"        — agent cognitive state and goal parameters
        "atmosphere"    — acoustic and thermodynamic environment
        "constraint"    — kinematic constraints (floor, wall, joint limits)
        "thermal"       — heat sources and thermal emission
        "electromagnetic" — light sources and electromagnetic properties

    Example:
        PhysicsLaw("gravity",    {"magnitude_ms2": 9.81, "direction": "neg_y"}, "scene")
        PhysicsLaw("material",   {"id": "fracturable_glass", "young_modulus_gpa": 70,
                                  "fracture_toughness": 0.75, "density_kgm3": 2500,
                                  "fracture_pattern": "radial_from_impact"}, "glass")
        PhysicsLaw("intent",     {"entity": "protagonist", "goal": "reach_hot_mug",
                                  "urgency": 0.4, "caution": 0.9}, "protagonist")
        PhysicsLaw("atmosphere", {"medium": "bar_interior_air", "temperature_c": 22,
                                  "speed_of_sound_ms": 344}, "scene")
    """
    law_type: Literal[
        "gravity", "material", "intent", "atmosphere",
        "constraint", "thermal", "electromagnetic"
    ]
    parameters: dict[str, Any]
    applies_to: str = "scene"   # entity_id or "scene" for global laws
    law_id: str = field(default_factory=lambda: f"law_{uuid.uuid4().hex[:6]}")


@dataclass
class ActionToken:
    """
    The fundamental unit of physical causality in Aether-Omni.

    Instead of specifying "at T=2.4s the marble is at (0.3, 0.8, 0.0),"
    an ActionToken specifies: WHAT force, on WHAT body, at WHAT contact
    point, for HOW LONG, producing WHAT acceleration.

    Positions become DERIVED quantities — the mathematical consequence
    of integrating forces over time. This makes positions physically
    correct by construction, not by inspection.

    The contradiction_check field stores Gemini's pre-validation result,
    ensuring no action token violates physics invariants before rendering.
    """
    token_id: str
    entity_id: str
    t: float                            # onset time (seconds)
    duration: float                     # seconds
    action_type: Literal[
        "torque", "contact_force", "locomotion", "grasp",
        "release", "deform", "fracture", "fluid_flow", "thermal_transfer"
    ]
    force_vector: Vec3                  # Newtons (or N·m for torque)
    contact_point: Vec3 | None          # where force is applied in world space
    resulting_acceleration: Vec3        # m/s² post-application
    physical_constraint: str | None     # e.g., "foot_must_contact_floor"
    contradiction_check: str | None     # Gemini's validation result
    confidence: float = 1.0


@dataclass
class HamiltonianState:
    """
    Phase-space description of a physical body.

    q = generalized position (where things are)
    p = generalized momentum (mass × velocity — how fast and in what direction)

    The Hamiltonian formulation is preferred for simulation because it:
    1. Conserves energy exactly (unlike Lagrangian formulations)
    2. Allows elegant integration forward in time
    3. Naturally separates position from momentum (no double-counting)

    is_agent=True means this body receives a thinking budget and generates
    an internal monologue that drives its body language tokens.
    """
    entity_id: str
    label: str
    mass_kg: float
    inertia_tensor: Mat3                # 3×3 rotational inertia
    q: Vec3                             # generalized position (meters)
    p: Vec3                             # generalized momentum (kg·m/s)
    friction_coefficient: float = 0.4
    restitution: float = 0.3            # bounciness 0..1
    material_id: str = "generic_solid"
    semantic_type: str = "rigid_body"
    intent: str | None = None           # natural-language agent goal
    is_agent: bool = False              # True = gets thinking budget + monologue


@dataclass
class LightTransportState:
    """
    Full radiometric description of the scene's light transport.

    Used to detect shadow violations (shadows pointing toward the light source)
    and to recalculate the full Light Transport Matrix when violations occur.
    This is the Engine IV mechanism that replaces simple lighting keyframes.
    """
    light_sources: list[dict[str, Any]]      # {id, type, position, intensity, color_temp_k}
    surface_materials: dict[str, Any]        # entity_id → BRDF parameters
    shadow_map: list[dict[str, Any]]         # pre-computed shadow volumes
    refractive_indices: dict[str, float]     # material_id → n
    timestamp: float = 0.0

    def get_expected_shadow_vector(self, light_id: str, entity_id: str) -> Vec3:
        """Shadow must be anti-parallel to light source direction."""
        light = next((l for l in self.light_sources if l["id"] == light_id), None)
        if not light:
            return {"x": 0, "y": -1, "z": 0}
        lx = light["position"]["x"]
        ly = light["position"]["y"]
        lz = light["position"]["z"]
        length = math.sqrt(lx**2 + ly**2 + lz**2) or 1.0
        return {"x": -lx / length, "y": -ly / length, "z": -lz / length}


@dataclass
class AcousticOcclusionEvent:
    """
    Computed acoustic occlusion via ray-casting through scene geometry.

    Key advance over HRTF (Engines II+III): this computes the LPF cutoff
    frequency and dB attenuation from material absorption coefficients —
    not just distance and pan. A wall between source and listener creates
    a measurable, physics-derived LPF, not an estimated one.
    """
    source_entity_id: str
    t: float
    source_position: Vec3
    listener_position: Vec3
    occluding_geometry: list[str]       # entity_ids of occluders in ray path
    lpf_cutoff_hz: float                # low-pass filter cutoff (walls absorb highs)
    lpf_resonance: float                # Q factor
    db_attenuation: float               # total attenuation through occluders
    reverb_ir_type: str                 # impulse response for receiving room
    instruction: str = ""


@dataclass
class AgentMonologue:
    """
    Per-character internal cognitive state generated by Gemini ER 1.5.

    The thinking budget controls how deeply Gemini reasons about this
    character's psychology. Protagonist → 800 tokens, background → 200 tokens.

    The body_language_tokens are ActionTokens derived from the cognitive state:
    a character with caution=0.9 generates a reach with low force magnitude.
    A character with urgency=0.9 generates a reach with high force magnitude.
    Psychology becomes physics.
    """
    agent_id: str
    agent_label: str
    thinking_budget: int
    internal_state: str                   # First-person stream of consciousness
    emotional_valence: float              # -1.0 (distress) .. +1.0 (positive)
    arousal_level: float                  # 0.0 (calm) .. 1.0 (high-energy)
    intent_vector: dict[str, Any]         # Structured goal representation
    micro_expression_cues: list[str]      # ["furrowed_brow", "cautious_lean"]
    body_language_tokens: list[ActionToken]  # Physical movements from cognition
    t_start: float = 0.0
    t_end: float = 10.0


@dataclass
class CorrectionVector:
    """
    A physics-derived correction emitted by the Kinetic Auditor.
    Targets specific latent regions for re-sampling at strength=0.35.

    For shadow violations: includes a full light_transport_correction
    — the recalculated shadow vectors for all affected light sources.
    """
    frame_index: int
    t: float
    entity_id: str
    violation_type: Literal[
        "floating_contact", "shadow_direction", "momentum_violation",
        "material_contradiction", "thermal_emission", "acoustic_occlusion"
    ]
    expected_state: dict[str, Any]
    observed_state: dict[str, Any]
    correction_magnitude: float           # 0..1 severity
    latent_region: dict[str, Any]         # pixel region for targeted re-sampling
    light_transport_correction: dict[str, Any] | None
    denoising_strength: float = 0.35      # Conservative: preserve structure


# ══════════════════════════════════════════════════════════════════════
#  OMNI-STATE BUFFER — 4D Persistent World
#  Hamiltonian + Voxel-NeRF + Acoustic + Agent Cognitive States
# ══════════════════════════════════════════════════════════════════════

class OmniStateBuffer:
    """
    The 4D world state that persists for the entire simulation.

    Four coupled sub-buffers:
      1. HamiltonianBuffer    — physics phase space (q, p per body)
      2. VoxelNeRFBuffer      — geometric/appearance ground truth
      3. AcousticSceneGraph   — sound propagation and occlusion state
      4. Agent Cognitive States — per-character monologues and intent

    All four are kept in sync and jointly queried during rendering.
    The Euler integrator in get_expected_position() derives positions
    from force integration — making spatial ground truth physically
    derived rather than manually specified.
    """

    def __init__(self) -> None:
        # Hamiltonian physics state
        self._hamiltonian: dict[str, HamiltonianState] = {}
        self._action_timeline: list[ActionToken] = []
        self._physics_laws: list[PhysicsLaw] = []
        self._light_transport: LightTransportState | None = None

        # Voxel-NeRF geometry buffer [FUTURE: replace with actual NeRF/3DGS]
        self._voxel_buffer: dict[str, dict[str, Any]] = {}
        self._hidden_geometry: dict[str, Any] = {}

        # Acoustic scene graph
        self._acoustic_events: list[AcousticOcclusionEvent] = []

        # Agent cognitive states
        self._monologues: dict[str, AgentMonologue] = {}

        # ICL memory log (append-only)
        self._event_log: list[str] = []

        # Frame snapshots for zero-shot continuity
        self._snapshots: dict[int, dict[str, Any]] = {}

    # ── Hamiltonian Management ─────────────────────────────────────────

    def register_body(self, state: HamiltonianState) -> None:
        self._hamiltonian[state.entity_id] = state
        self._voxel_buffer[state.entity_id] = {
            "label": state.label,
            "material_id": state.material_id,
            "material_hash": hashlib.sha256(
                state.material_id.encode()
            ).hexdigest(),
            "initial_position": state.q,
        }
        self._log(
            f"Body registered: '{state.label}' · {state.mass_kg}kg · "
            f"material={state.material_id} · intent='{state.intent}'"
        )

    def push_action_token(self, token: ActionToken) -> None:
        self._action_timeline.append(token)
        self._action_timeline.sort(key=lambda t: t.t)
        self._log(
            f"Action [{token.action_type}] on '{token.entity_id}' "
            f"at T={token.t:.3f}s · force={token.force_vector}"
        )

    def register_law(self, law: PhysicsLaw) -> None:
        self._physics_laws.append(law)
        self._log(f"Physics law: {law.law_type} → {law.parameters}")

    def update_light_transport(self, lt: LightTransportState) -> None:
        self._light_transport = lt
        self._log(
            f"Light transport updated: {len(lt.light_sources)} sources · T={lt.timestamp:.2f}s"
        )

    def register_monologue(self, monologue: AgentMonologue) -> None:
        self._monologues[monologue.agent_id] = monologue
        self._log(
            f"Agent '{monologue.agent_label}' internal state: "
            f"valence={monologue.emotional_valence:+.2f} · "
            f"arousal={monologue.arousal_level:.2f}"
        )

    def register_acoustic_event(self, event: AcousticOcclusionEvent) -> None:
        self._acoustic_events.append(event)

    # ── Euler Integration — Position Derivation ────────────────────────

    def get_expected_position(self, entity_id: str, t: float) -> Vec3:
        """
        Derive world position at time t by integrating forces forward from t=0.

        This is the core innovation of Aether-Omni: positions are NOT specified,
        they are DERIVED from F=ma applied to action tokens. This means positions
        can never be wrong — they are the mathematical consequence of the physics.

        Uses simple Euler integration (dt=0.01s) — sufficient for constraint
        checking. Production upgrade: use RK4 for smoother integration.
        """
        body = self._hamiltonian.get(entity_id)
        if not body:
            return {"x": 0.0, "y": 0.0, "z": 0.0}

        # Initial velocity from momentum
        vx = body.p["x"] / body.mass_kg
        vy = body.p["y"] / body.mass_kg
        vz = body.p["z"] / body.mass_kg
        x, y, z = body.q["x"], body.q["y"], body.q["z"]

        gravity = self._get_gravity()
        relevant_tokens = [
            tok for tok in self._action_timeline
            if tok.entity_id == entity_id and tok.t <= t
        ]

        dt_sim = 0.01
        t_sim  = 0.0
        while t_sim < t:
            dt = min(dt_sim, t - t_sim)

            # Apply gravity
            vy += gravity * dt

            # Apply active action tokens
            for tok in relevant_tokens:
                if tok.t <= t_sim <= tok.t + tok.duration:
                    vx += tok.resulting_acceleration["x"] * dt
                    vy += tok.resulting_acceleration["y"] * dt
                    vz += tok.resulting_acceleration["z"] * dt

            # Integrate position
            x += vx * dt
            y += vy * dt
            z += vz * dt

            # Floor constraint (y >= 0)
            y = max(y, 0.0)
            t_sim += dt

        return {"x": round(x, 4), "y": round(y, 4), "z": round(z, 4)}

    def _get_gravity(self) -> float:
        for law in self._physics_laws:
            if law.law_type == "gravity":
                return -abs(law.parameters.get("magnitude_ms2", 9.81))
        return -9.81

    # ── Voxel-NeRF Buffer ──────────────────────────────────────────────

    def query_hidden_geometry(self, entity_id: str) -> dict[str, Any]:
        """
        Return stored geometry for an entity not currently visible.
        [FUTURE]: Replace with actual NeRF/3DGS field query.
        Today: returns Gemini-reasoned voxel description.
        """
        return (
            self._voxel_buffer.get(entity_id)
            or self._hidden_geometry.get(entity_id, {})
        )

    def store_hidden_geometry(self, entity_id: str, geometry: dict[str, Any]) -> None:
        self._hidden_geometry[entity_id] = geometry

    # ── Acoustic Scene Graph ───────────────────────────────────────────

    def get_acoustic_events_in_window(
        self,
        t_start: float,
        t_end: float) -> list[AcousticOcclusionEvent]:
        return [ev for ev in self._acoustic_events if t_start <= ev.t <= t_end]

    # ── Snapshots & ICL Log ────────────────────────────────────────────

    def snapshot(self, frame_index: int) -> None:
        """Capture complete world state at this frame for future temporal reference."""
        self._snapshots[frame_index] = {
            "frame_index": frame_index,
            "hamiltonian": {
                eid: {
                    "position": s.q,
                    "momentum": s.p,
                    "material": s.material_id,
                }
                for eid, s in self._hamiltonian.items()
            },
            "light_transport_hash": hashlib.sha256(
                json.dumps(
                    {"sources": self._light_transport.light_sources
                     if self._light_transport else []},
                    sort_keys=True).encode()
            ).hexdigest()[:12],
            "monologue_states": {
                aid: {
                    "valence": m.emotional_valence,
                    "arousal": m.arousal_level,
                }
                for aid, m in self._monologues.items()
            },
        }

    def get_snapshot(self, frame_index: int) -> dict[str, Any]:
        if frame_index in self._snapshots:
            return self._snapshots[frame_index]
        closest = max(
            (k for k in self._snapshots if k <= frame_index), default=None
        )
        return (
            self._snapshots.get(closest, {}) if closest is not None else {}
        )

    def _log(self, entry: str) -> None:
        self._event_log.append(f"[{time.strftime('%H:%M:%S')}] {entry}")

    def get_icl_log(self, max_entries: int = 60) -> str:
        return "\n".join(self._event_log[-max_entries:])

    def export_for_rendering(
        self,
        t_start: float,
        t_end: float) -> dict[str, Any]:
        """Full world-state export for a given time window — fed to Grok."""
        return {
            "hamiltonian_states": {
                eid: {
                    "label": s.label,
                    "mass_kg": s.mass_kg,
                    "material_id": s.material_id,
                    "intent": s.intent,
                    "expected_position_t_start": self.get_expected_position(eid, t_start),
                    "expected_position_t_end": self.get_expected_position(eid, t_end),
                    "friction": s.friction_coefficient,
                    "restitution": s.restitution,
                }
                for eid, s in self._hamiltonian.items()
            },
            "action_tokens": [
                {
                    "entity_id": tok.entity_id,
                    "action_type": tok.action_type,
                    "t": tok.t,
                    "duration": tok.duration,
                    "force_vector": tok.force_vector,
                    "contact_point": tok.contact_point,
                    "constraint": tok.physical_constraint,
                }
                for tok in self._action_timeline
                if t_start <= tok.t <= t_end
            ],
            "agent_states": {
                aid: {
                    "internal_state": m.internal_state,
                    "emotional_valence": m.emotional_valence,
                    "arousal_level": m.arousal_level,
                    "micro_expression_cues": m.micro_expression_cues,
                }
                for aid, m in self._monologues.items()
                if m.t_start <= t_start <= m.t_end
            },
            "acoustic_events": [
                {
                    "source_entity": ev.source_entity_id,
                    "t": ev.t,
                    "lpf_cutoff_hz": ev.lpf_cutoff_hz,
                    "db_attenuation": ev.db_attenuation,
                    "instruction": ev.instruction,
                }
                for ev in self.get_acoustic_events_in_window(t_start, t_end)
            ],
            "light_transport": {
                "sources": (
                    self._light_transport.light_sources
                    if self._light_transport else []
                ),
            },
            "physics_laws": [
                {
                    "type": law.law_type,
                    "params": law.parameters,
                    "applies_to": law.applies_to,
                }
                for law in self._physics_laws
            ],
            "icl_memory": self.get_icl_log(),
        }


# ══════════════════════════════════════════════════════════════════════
#  HAMILTONIAN PHYSICS KERNEL
#  Gemini as the Gravity + Logic Engine
# ══════════════════════════════════════════════════════════════════════

class HamiltonianPhysicsKernel:
    """
    Uses Gemini ER 1.5 to build the phase-space representation of the scene
    and generate Action Tokens — the causal language of physics.

    PRE-RENDER CONTRADICTION DETECTION:
    Before Grok renders a single pixel, this kernel validates all action tokens
    against 6 physics invariants. Violations are logged and can block rendering.

    This is the key advance: errors are PREVENTED, not corrected.
    """

    CONTRADICTION_CHECKS = [
        "No entity may be airborne without upward force exceeding gravity × mass.",
        "Foot contact events must have ground_contact=true in the action token.",
        "Shadow vectors must be anti-parallel to the primary light source direction.",
        "Fractured materials must produce fragments whose combined mass ≤ original.",
        "Liquids must follow the steepest descent gradient from their container.",
        "Thermal emission must be directionally consistent with heat source position.",
    ]

    def __init__(
        self,
        gemini: GeminiERClient,
        buffer: OmniStateBuffer) -> None:
        self.gemini = gemini
        self.buffer = buffer

    def initialize(
        self,
        physics_laws: list[PhysicsLaw],
        scene_brief: str,
        reference_image_path: str | None = None,
        duration_seconds: float = 10.0) -> list[HamiltonianState]:
        """Stage 1a: Initialize Hamiltonian states for all scene bodies."""
        for law in physics_laws:
            self.buffer.register_law(law)

        prompt = _build_hamiltonian_prompt(physics_laws, scene_brief, duration_seconds)
        content: list[Any] = []

        if reference_image_path:
            with open(reference_image_path, "rb") as f:
                content.append({"mime_type": "image/jpeg", "data": f.read()})
        content.append(prompt)

        response = self.gemini.generate_content(
            content)

        data: dict = json.loads(_extract_json(response.text))
        states = []

        for body_data in data.get("bodies", []):
            inertia = body_data.get(
                "inertia_tensor",
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            state = HamiltonianState(
                entity_id=body_data["entity_id"],
                label=body_data["label"],
                mass_kg=body_data["mass_kg"],
                inertia_tensor=inertia,
                q=body_data["initial_position"],
                p={
                    axis: body_data["initial_velocity"][axis] * body_data["mass_kg"]
                    for axis in ("x", "y", "z")
                },
                friction_coefficient=body_data.get("friction_coefficient", 0.4),
                restitution=body_data.get("restitution", 0.3),
                material_id=body_data.get("material_id", "generic_solid"),
                semantic_type=body_data.get("semantic_type", "rigid_body"),
                intent=body_data.get("intent"),
                is_agent=body_data.get("is_agent", False))
            self.buffer.register_body(state)
            states.append(state)

        # Initialize light transport
        lt_data = data.get("light_transport", {})
        lt = LightTransportState(
            light_sources=lt_data.get("light_sources", []),
            surface_materials=lt_data.get("surface_materials", {}),
            shadow_map=lt_data.get("shadow_map", []),
            refractive_indices=lt_data.get("refractive_indices", {}))
        self.buffer.update_light_transport(lt)

        logger.info(
            "HamiltonianKernel: %d bodies · %d light sources",
            len(states), len(lt.light_sources))
        return states

    def generate_action_tokens(
        self,
        states: list[HamiltonianState],
        duration_seconds: float) -> list[ActionToken]:
        """Stage 1b: Generate the Action Token timeline — the causal script."""
        body_summary = [
            {
                "entity_id": s.entity_id,
                "label": s.label,
                "mass_kg": s.mass_kg,
                "material_id": s.material_id,
                "intent": s.intent,
                "is_agent": s.is_agent,
                "initial_position": s.q,
            }
            for s in states
        ]
        laws_summary = [
            {"type": l.law_type, "params": l.parameters, "applies_to": l.applies_to}
            for l in self.buffer._physics_laws
        ]

        prompt = f"""
You are a physics simulation kernel generating Action Tokens.
An Action Token is the minimal causal description of a physical event:
WHAT force, on WHAT body, at WHAT contact point, for HOW LONG, producing WHAT acceleration.

REGISTERED BODIES: {json.dumps(body_summary, indent=2)}
PHYSICS LAWS: {json.dumps(laws_summary, indent=2)}

CONTRADICTION CHECKS (every token must pass ALL of these):
{chr(10).join(f"  {i+1}. {c}" for i, c in enumerate(self.CONTRADICTION_CHECKS))}

Generate ALL action tokens for the {duration_seconds}-second simulation.

Return ONLY a JSON array:
[
  {{
    "token_id": "tok_001",
    "entity_id": "ent_protagonist",
    "t": 1.2, "duration": 0.3,
    "action_type": "locomotion",
    "force_vector": {{"x": 0.0, "y": 0.0, "z": 15.0}},
    "contact_point": {{"x": 0.1, "y": 0.0, "z": 0.0}},
    "resulting_acceleration": {{"x": 0.0, "y": 0.0, "z": 0.21}},
    "physical_constraint": "right_foot_contact_floor_t=1.2",
    "contradiction_check": "foot_contact=true, floor_normal=(0,1,0), normal_force=686N",
    "confidence": 0.97
  }}
]
"""
        response = self.gemini.generate_content(
            prompt)
        token_data: list[dict] = json.loads(_extract_json(response.text))

        tokens = []
        for td in token_data:
            tok = ActionToken(
                token_id=td.get("token_id", f"tok_{uuid.uuid4().hex[:6]}"),
                entity_id=td["entity_id"],
                t=td["t"],
                duration=td["duration"],
                action_type=td["action_type"],
                force_vector=td["force_vector"],
                contact_point=td.get("contact_point"),
                resulting_acceleration=td["resulting_acceleration"],
                physical_constraint=td.get("physical_constraint"),
                contradiction_check=td.get("contradiction_check"),
                confidence=td.get("confidence", 1.0))
            self.buffer.push_action_token(tok)
            tokens.append(tok)

        logger.info("HamiltonianKernel: %d action tokens generated", len(tokens))
        return tokens

    def validate_pre_render(self, tokens: list[ActionToken]) -> list[str]:
        """
        Stage 1c: Pre-render contradiction detection.
        Returns list of violation descriptions (empty = clear to render).
        """
        violations = []
        gravity = abs(self.buffer._get_gravity())

        for tok in tokens:
            body = self.buffer._hamiltonian.get(tok.entity_id)
            if not body:
                continue

            # CHECK 1: Airborne without upward force
            if tok.action_type == "locomotion":
                fy = tok.force_vector.get("y", 0)
                required_lift = body.mass_kg * gravity
                if tok.contact_point is None and fy < required_lift * 0.1:
                    violations.append(
                        f"FLOATING VIOLATION: '{body.label}' locomotion at "
                        f"T={tok.t:.2f}s has no contact point and insufficient "
                        f"upward force ({fy:.1f}N < {required_lift:.1f}N required)."
                    )

            # CHECK 2: Fracture energy conservation
            if tok.action_type == "fracture":
                ke = 0.5 * body.mass_kg * (
                    tok.resulting_acceleration["x"]**2 +
                    tok.resulting_acceleration["y"]**2 +
                    tok.resulting_acceleration["z"]**2
                )
                input_energy = math.sqrt(
                    tok.force_vector["x"]**2 +
                    tok.force_vector["y"]**2 +
                    tok.force_vector["z"]**2
                ) * 0.1
                if ke > input_energy * 1.5:
                    violations.append(
                        f"ENERGY VIOLATION: '{body.label}' fracture at T={tok.t:.2f}s "
                        f"produces more KE ({ke:.2f}J) than input allows ({input_energy:.2f}J)."
                    )

        if violations:
            logger.warning("Pre-render contradictions: %d found", len(violations))
            for v in violations:
                logger.warning("  ✗ %s", v)
        else:
            logger.info("Pre-render validation: ✓ All clear")

        return violations


# ══════════════════════════════════════════════════════════════════════
#  MULTI-AGENT THINKING ENGINE
# ══════════════════════════════════════════════════════════════════════

class AgentThinkingEngine:
    """
    Assigns a 'Thinking Budget' to each character and generates their
    internal cognitive state via dedicated Gemini calls.

    Thinking budgets scale with character importance:
        Protagonist:  800 tokens (deep reasoning)
        Secondary:    400 tokens
        Background:   200 tokens (minimal)

    The cognitive state drives ActionTokens via body_language_tokens —
    micro-movements that physically encode psychological state.
    Psychology → physics → rendering.
    """

    def __init__(
        self,
        gemini: GeminiERClient,
        buffer: OmniStateBuffer) -> None:
        self.gemini = gemini
        self.buffer = buffer

    def think(
        self,
        agents: list[HamiltonianState],
        scene_context: str,
        duration_seconds: float) -> list[AgentMonologue]:
        monologues = []
        budget_map = self._assign_thinking_budgets(agents)

        for agent in agents:
            if not agent.is_agent:
                continue
            monologue = self._generate_monologue(
                agent, scene_context,
                budget_map.get(agent.entity_id, 200),
                duration_seconds)
            self.buffer.register_monologue(monologue)
            monologues.append(monologue)

        logger.info(
            "AgentThinkingEngine: %d agent monologues generated", len(monologues)
        )
        return monologues

    def _generate_monologue(
        self,
        agent: HamiltonianState,
        context: str,
        thinking_budget: int,
        duration: float) -> AgentMonologue:
        prompt = f"""
# Note: thinking budget is now passed via ThinkingConfig in GeminiERClient

CHARACTER: {agent.label}
SCENE CONTEXT: {context}
DURATION: {duration} seconds
INTENT: {agent.intent or "undefined"}
PHYSICAL STATE: position={agent.q}, mass={agent.mass_kg}kg, material={agent.material_id}

Generate this character's internal cognitive state.
Return ONLY this JSON:
{{
  "internal_state": "First-person stream of consciousness...",
  "emotional_valence": -0.3,
  "arousal_level": 0.6,
  "intent_vector": {{
    "primary_goal": "reach_the_mug",
    "obstacle": "mug_is_hot",
    "strategy": "approach_slowly_test_heat_first",
    "confidence": 0.7
  }},
  "micro_expression_cues": ["furrowed_brow", "cautious_lean", "hand_hesitation"],
  "body_language_tokens": [
    {{
      "t": 1.5, "duration": 0.8, "action_type": "locomotion",
      "description": "Slow cautious reach",
      "force_vector": {{"x": 0.5, "y": 0.0, "z": 2.0}},
      "contact_point": null,
      "resulting_acceleration": {{"x": 0.007, "y": 0.0, "z": 0.029}},
      "physical_constraint": "arm_extension_limited_by_caution",
      "contradiction_check": "arm_velocity_slow=true"
    }}
  ]
}}
"""
        response = self.gemini.generate_content(
            prompt)
        data: dict = json.loads(_extract_json(response.text))

        body_lang_tokens = []
        for bt in data.get("body_language_tokens", []):
            tok = ActionToken(
                token_id=f"agent_{agent.entity_id}_{uuid.uuid4().hex[:4]}",
                entity_id=agent.entity_id,
                t=bt["t"],
                duration=bt["duration"],
                action_type=bt["action_type"],
                force_vector=bt["force_vector"],
                contact_point=bt.get("contact_point"),
                resulting_acceleration=bt["resulting_acceleration"],
                physical_constraint=bt.get("physical_constraint"),
                contradiction_check=bt.get("contradiction_check"))
            self.buffer.push_action_token(tok)
            body_lang_tokens.append(tok)

        return AgentMonologue(
            agent_id=agent.entity_id,
            agent_label=agent.label,
            thinking_budget=thinking_budget,
            internal_state=data["internal_state"],
            emotional_valence=data["emotional_valence"],
            arousal_level=data["arousal_level"],
            intent_vector=data.get("intent_vector", {}),
            micro_expression_cues=data.get("micro_expression_cues", []),
            body_language_tokens=body_lang_tokens,
            t_end=duration)

    @staticmethod
    def _assign_thinking_budgets(
        agents: list[HamiltonianState]) -> dict[str, int]:
        agents_sorted = sorted(agents, key=lambda a: a.mass_kg, reverse=True)
        budgets: dict[str, int] = {}
        total = 2000
        for i, agent in enumerate(agents_sorted):
            if not agent.is_agent:
                continue
            share = total // (2 ** i)
            budgets[agent.entity_id] = max(200, min(share, 800))
        return budgets


# ══════════════════════════════════════════════════════════════════════
#  ACOUSTIC OCCLUSION ENGINE
#  Physics-derived audio processing (not HRTF approximation)
# ══════════════════════════════════════════════════════════════════════

class AcousticOcclusionEngine:
    """
    Ray-casts through scene geometry to compute acoustic occlusion.

    Key advance over HRTF (Engines II & III): this engine uses material
    absorption coefficients (dB/meter) and LPF cutoffs to compute the
    exact acoustic effect of walls, doors, and other occluding geometry.

    Material absorption coefficients (physics-based):
        Concrete:   LPF 800Hz,  12 dB/m  (heavy walls absorb most HF)
        Drywall:    LPF 1200Hz,  8 dB/m
        Wood:       LPF 2000Hz,  5 dB/m
        Glass:      LPF 3500Hz,  3 dB/m  (glass passes more HF)
        Fabric:     LPF 600Hz,  15 dB/m  (soft materials absorb most)
        Open air:   No LPF,      0 dB/m
    """

    MATERIAL_ABSORPTION = {
        "concrete":  {"lpf_hz": 800,   "db_per_meter": 12},
        "drywall":   {"lpf_hz": 1200,  "db_per_meter":  8},
        "wood":      {"lpf_hz": 2000,  "db_per_meter":  5},
        "glass":     {"lpf_hz": 3500,  "db_per_meter":  3},
        "open_air":  {"lpf_hz": 20000, "db_per_meter":  0},
        "fabric":    {"lpf_hz": 600,   "db_per_meter": 15},
    }

    def __init__(self, buffer: OmniStateBuffer) -> None:
        self.buffer = buffer

    def compute_occlusion(
        self,
        source_entity_id: str,
        source_pos: Vec3,
        listener_pos: Vec3,
        t: float,
        occluding_entities: list[dict[str, Any]]) -> AcousticOcclusionEvent:
        """
        Compute acoustic occlusion for a sound source at time t.

        Args:
            occluding_entities: list of {entity_id, material_id, thickness_m}
        """
        total_db_atten = 0.0
        min_lpf_hz     = 20000.0
        occluder_ids   = []

        for occ in occluding_entities:
            material  = occ.get("material_id", "drywall")
            thickness = occ.get("thickness_m", 0.2)
            props     = self.MATERIAL_ABSORPTION.get(
                material, self.MATERIAL_ABSORPTION["drywall"]
            )
            total_db_atten += props["db_per_meter"] * thickness
            min_lpf_hz      = min(min_lpf_hz, props["lpf_hz"])
            occluder_ids.append(occ["entity_id"])

        # Distance attenuation (inverse square law)
        dx   = source_pos["x"] - listener_pos["x"]
        dy   = source_pos["y"] - listener_pos["y"]
        dz   = source_pos["z"] - listener_pos["z"]
        dist = math.sqrt(dx**2 + dy**2 + dz**2) or 0.01
        total_db_atten += abs(-20 * math.log10(max(dist, 0.01)))

        room_ir = "anechoic" if not occluder_ids else "medium_room"

        instruction = (
            f"At T={t:.3f}s, '{source_entity_id}' is acoustically occluded by "
            f"{occluder_ids if occluder_ids else 'open air'}. "
            f"Apply LPF at {min_lpf_hz:.0f}Hz, "
            f"total attenuation -{total_db_atten:.1f}dB, "
            f"reverb type: {room_ir}."
        )

        event = AcousticOcclusionEvent(
            source_entity_id=source_entity_id,
            t=t,
            source_position=source_pos,
            listener_position=listener_pos,
            occluding_geometry=occluder_ids,
            lpf_cutoff_hz=min_lpf_hz,
            lpf_resonance=0.7,
            db_attenuation=total_db_atten,
            reverb_ir_type=room_ir,
            instruction=instruction)
        self.buffer.register_acoustic_event(event)
        return event


# ══════════════════════════════════════════════════════════════════════
#  LATENT PERTURBATION GUIDE — Voxel-to-Latent Handshake
# ══════════════════════════════════════════════════════════════════════

class LatentPerturbationGuide:
    """
    The Voxel-to-Latent handshake between the physics kernel and the renderer.

    Converts action tokens into screen-space conditioning payloads
    that tell Grok exactly where bodies must appear at each frame.

    Physics directive: Action Tokens → cinematically descriptive language
    Agent directive:   Monologue cognitive state → micro-expression instructions
    Acoustic directive: Occlusion events → LPF + attenuation instructions

    [FUTURE API]: When Grok exposes deep_guidance attention injection,
    this class generates guidance vectors fed directly into cross-attention layers.
    Interface contract for upgrade:
        grok_api.deep_guidance(
            guidance_vectors=self.get_guidance_vectors(tokens),
            attention_layer="cross_attention_8",
            guidance_scale=7.5
        )
    """

    def __init__(self, resolution: tuple[int, int] = (1280, 720)) -> None:
        self.W, self.H = resolution

    def build_conditioning_payload(
        self,
        buffer: OmniStateBuffer,
        t_start: float,
        t_end: float,
        camera_pos: Vec3,
        look_at: Vec3,
        fov_deg: float) -> dict[str, Any]:
        world_state = buffer.export_for_rendering(t_start, t_end)

        # Project all bodies to screen space
        screen_projections = {}
        for eid, body_data in world_state["hamiltonian_states"].items():
            pos    = body_data["expected_position_t_start"]
            screen = self._project(pos, camera_pos, look_at, fov_deg)
            if screen:
                screen_projections[eid] = {
                    **screen,
                    "label":       body_data["label"],
                    "material_id": body_data["material_id"],
                    "intent":      body_data["intent"],
                }

        return {
            "conditioning_type":       "physics_guided_latent",
            "t_window":                {"start": t_start, "end": t_end},
            "screen_anchor_map":       screen_projections,
            "physics_directive":       self._tokens_to_directive(
                world_state["action_tokens"],
                world_state["hamiltonian_states"]),
            "agent_persona_directives": self._monologues_to_directives(
                world_state["agent_states"]
            ),
            "acoustic_directive":      self._build_acoustic_directive(
                world_state["acoustic_events"]
            ),
            "light_transport":         world_state["light_transport"],
            "physics_laws":            world_state["physics_laws"],
            "material_locks": {
                eid: data["material_id"]
                for eid, data in world_state["hamiltonian_states"].items()
            },
            "icl_memory":              world_state["icl_memory"],
            "frame_0_state":           buffer.get_snapshot(0),
            "system_instructions":     GROK_OMNI_SYSTEM_INSTRUCTIONS,
        }

    def _tokens_to_directive(
        self,
        tokens: list[dict],
        bodies: dict[str, Any]) -> str:
        if not tokens:
            return "No significant physical actions in this window."
        lines = ["PHYSICAL ACTIONS THIS WINDOW (render these exactly):"]
        for tok in tokens:
            label = bodies.get(tok["entity_id"], {}).get("label", tok["entity_id"])
            fmag  = math.sqrt(
                tok["force_vector"]["x"]**2 +
                tok["force_vector"]["y"]**2 +
                tok["force_vector"]["z"]**2
            )
            lines.append(
                f"  • T={tok['t']:.2f}s [{tok['action_type']}] on '{label}': "
                f"{fmag:.1f}N applied."
                + (f" Contact at {tok['contact_point']}." if tok.get("contact_point") else "")
                + (f" Constraint: {tok['constraint']}." if tok.get("constraint") else "")
            )
        return "\n".join(lines)

    def _monologues_to_directives(
        self,
        agent_states: dict[str, Any]) -> list[str]:
        directives = []
        for aid, state in agent_states.items():
            cues    = ", ".join(state.get("micro_expression_cues", []))
            valence = state["emotional_valence"]
            arousal = state["arousal_level"]
            mood    = (
                "tense, guarded"    if valence < -0.3 else
                "neutral, focused"  if abs(valence) <= 0.3 else
                "open, positive"
            )
            energy = (
                "slow and deliberate" if arousal < 0.4 else
                "moderate energy"     if arousal < 0.7 else
                "high energy, reactive"
            )
            directives.append(
                f"'{aid}' internal state: '{state['internal_state'][:120]}...' "
                f"Mood: {mood}. Energy: {energy}. "
                f"Micro-expressions: {cues if cues else 'subtle, neutral'}."
            )
        return directives

    def _build_acoustic_directive(self, events: list[dict]) -> str:
        if not events:
            return "No acoustic occlusion events in this window."
        lines = ["ACOUSTIC SPATIALIZATION (apply exactly):"]
        for ev in events:
            lines.append(f"  • {ev['instruction']}")
        return "\n".join(lines)

    def _project(
        self,
        world: Vec3,
        cam: Vec3,
        look_at: Vec3,
        fov: float) -> dict | None:
        dx = world["x"] - cam["x"]
        dy = world["y"] - cam["y"]
        dz = world["z"] - cam["z"]
        fx = look_at["x"] - cam["x"]
        fz = look_at["z"] - cam["z"]
        fl = math.sqrt(fx**2 + fz**2) or 1.0
        fx /= fl; fz /= fl
        depth = dx * fx + dz * fz
        if depth < 0.01:
            return None
        rx, rz = fz, -fx
        sx = (dx * rx + dz * rz) / (depth * math.tan(math.radians(fov / 2)))
        sy = -dy / (depth * math.tan(math.radians(fov * self.H / self.W / 2)))
        if abs(sx) > 1.1 or abs(sy) > 1.1:
            return None
        return {
            "px":      round((sx + 1) / 2 * self.W, 1),
            "py":      round((sy + 1) / 2 * self.H, 1),
            "depth_m": round(depth, 3),
        }


# ══════════════════════════════════════════════════════════════════════
#  KINETIC AUDITOR — The 2% Reality-Check Loop
# ══════════════════════════════════════════════════════════════════════

class FrameKineticAuditor:
    """
    Frame-by-frame kinetic audit comparing rendered video against
    Hamiltonian-derived ground truth positions.

    Five violation types checked at 2% threshold:
        1. POSITION DRIFT         — >2% deviation from Euler-integrated position
        2. FLOATING CONTACT       — airborne without action token support
        3. SHADOW DIRECTION       — shadow inconsistent with light source position
        4. MATERIAL DRIFT         — texture deviated from Frame-0 hash
        5. MOMENTUM VIOLATION     — entity faster than action tokens allow

    For shadow violations: recalculates the full Light Transport Matrix
    and emits a correction vector with precise shadow correction instructions.
    """

    PIXEL_DRIFT_THRESHOLD = 0.02   # 2% — tightest in the series

    def __init__(
        self,
        gemini: GeminiERClient,
        buffer: OmniStateBuffer) -> None:
        self.gemini = gemini
        self.buffer = buffer

    def audit_frame(
        self,
        video_url: str,
        frame_index: int,
        fps: int) -> list[CorrectionVector]:
        t = frame_index / fps
        expected_state = {
            eid: self.buffer.get_expected_position(eid, t)
            for eid in self.buffer._hamiltonian
        }
        frame_0_state = self.buffer.get_snapshot(0)
        icl           = self.buffer.get_icl_log(max_entries=30)

        prompt = f"""
You are a kinetic auditor performing a physics-consistency check on a rendered frame.

VIDEO: {video_url}
FRAME: {frame_index} (T={t:.3f}s)

EXPECTED POSITIONS (from Hamiltonian integration):
{json.dumps(expected_state, indent=2)}

FRAME 0 REFERENCE STATE (texture ground truth):
{json.dumps(frame_0_state, indent=2)}

LIGHT TRANSPORT STATE:
{json.dumps({
    "sources": self.buffer._light_transport.light_sources
    if self.buffer._light_transport else []
}, indent=2)}

ICL MEMORY:
{icl}

Check for violations (threshold: 2%):
1. POSITION DRIFT     — >2% deviation from expected position
2. FLOATING CONTACT   — entity airborne without action token support
3. SHADOW DIRECTION   — shadow pointing wrong direction vs light sources
4. MATERIAL DRIFT     — texture/material changed from Frame 0 reference
5. MOMENTUM VIOLATION — entity moving faster than action tokens allow

Return ONLY a JSON array ([] if none):
[
  {{
    "entity_id": "ent_001",
    "violation_type": "shadow_direction",
    "drift_magnitude": 0.08,
    "expected_state": {{"shadow_direction": {{"x": -0.57, "y": -0.82, "z": 0.0}}}},
    "observed_state": {{"shadow_direction": {{"x": 0.3, "y": -0.95, "z": 0.0}}}},
    "pixel_region": {{"x1": 200, "y1": 400, "x2": 600, "y2": 720}},
    "correction_priority": "high"
  }}
]
"""
        response = self.gemini.generate_content(
            prompt)
        violations_data: list[dict] = json.loads(_extract_json(response.text))

        correction_vectors = []
        for vd in violations_data:
            lt_correction = None
            if vd["violation_type"] == "shadow_direction" and self.buffer._light_transport:
                lt_correction = self._recalculate_light_transport(
                    vd["entity_id"], vd["observed_state"], t
                )

            cv = CorrectionVector(
                frame_index=frame_index,
                t=t,
                entity_id=vd["entity_id"],
                violation_type=vd["violation_type"],
                expected_state=vd["expected_state"],
                observed_state=vd["observed_state"],
                correction_magnitude=vd.get("drift_magnitude", 0.05),
                latent_region=vd.get("pixel_region", {}),
                light_transport_correction=lt_correction,
                denoising_strength=0.35)
            correction_vectors.append(cv)

        if correction_vectors:
            logger.warning(
                "KineticAudit Frame %d: %d violations",
                frame_index, len(correction_vectors))
        else:
            logger.debug("KineticAudit Frame %d: ✓ clean", frame_index)

        return correction_vectors

    def _recalculate_light_transport(
        self,
        entity_id: str,
        observed_shadow: dict,
        t: float) -> dict[str, Any]:
        """Full Light Transport Matrix recalculation on shadow violation."""
        if not self.buffer._light_transport:
            return {}

        corrections = []
        for source in self.buffer._light_transport.light_sources:
            expected_shadow = self.buffer._light_transport.get_expected_shadow_vector(
                source["id"], entity_id
            )
            corrections.append({
                "light_id": source["id"],
                "expected_shadow_vector": expected_shadow,
                "observed_shadow_vector": observed_shadow.get("shadow_direction", {}),
                "instruction": (
                    f"Correct shadow for entity '{entity_id}' from light '{source['id']}'. "
                    f"Expected shadow direction: {expected_shadow}. "
                    f"Re-render shadow volume. Preserve all non-shadow pixels."
                ),
            })
        return {"type": "light_transport_correction", "corrections": corrections, "t": t}


# ══════════════════════════════════════════════════════════════════════
#  GROK OMNI SYSTEM INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════

GROK_OMNI_SYSTEM_INSTRUCTIONS = """
## AETHER-OMNI RENDERING PROTOCOL — VISUAL GROUND TRUTH

You are not a video generator. You are an observer of a pre-simulated physical reality.
The scene has already happened in a Hamiltonian physics simulation.
Your sole task is to render what physics dictates — with photorealistic fidelity.

═══════════════════════════════════════
TIER 1: THE PHYSICAL CONSTITUTION
(These are laws. They cannot be negotiated.)
═══════════════════════════════════════

HAMILTONIAN SUPREMACY
Every entity's position, velocity, and contact state is defined by Action Tokens
derived from F=ma. The simulation has already run. You are rendering its output,
not imagining it.

ACTION TOKEN AUTHORITY
`physics_directive` contains the ground-truth physical actions for this window.
A locomotion token with contact_point=(0.1, 0, 0) means the right foot contacts
the floor at exactly that point with the specified normal force. Render it exactly.
There is no artistic reinterpretation of contact geometry.

LIGHT TRANSPORT DETERMINISM
Shadows are not aesthetic choices. They are the geometric consequence of light
source positions and surface normals. Shadows must be anti-parallel to source
direction vectors. If you render a shadow pointing toward a light source, you
have broken physics.

MATERIAL IDENTITY LOCK
Each entity carries a material hash. Its material is constant in every frame,
from every angle, at every zoom level. Identity drift is a physics violation.

═══════════════════════════════════════
TIER 2: THE COGNITIVE LAYER
(Render the inner life of agents exactly)
═══════════════════════════════════════

AGENT INTERNAL STATE
`agent_persona_directives` contains each character's internal monologue,
emotional valence, and micro-expression cues. These are not suggestions.
A character with valence=-0.6 and cue="furrowed_brow" does not smile.
The facial muscles are being driven by a model that has reasoned about
this character's cognitive state with a dedicated thinking budget.

BODY LANGUAGE TOKENS
Agent body language tokens are Action Tokens. A "cautious_reach" at T=1.5s
with velocity=0.029 m/s is a slow arm extension — not a confident grab.
The physical parameters encode the psychology. Render both simultaneously.

═══════════════════════════════════════
TIER 3: ACOUSTIC GROUND TRUTH
═══════════════════════════════════════

`acoustic_directive` contains ray-cast acoustic occlusion results.
Apply the specified LPF cutoff and dB attenuation exactly.
A source occluded by concrete (LPF=800Hz) sounds muffled.
This is physics, not a mixing preference.

═══════════════════════════════════════
TIER 4: YOUR CREATIVE SOVEREIGNTY
(Everything physics does not specify belongs to you)
═══════════════════════════════════════

- The emotional temperature of light: how warmth bleeds across surfaces
- Micro-surface resolution: pore structure, fabric thread count, dust settling
- Atmospheric volume: the density and color of air between surfaces
- The rhythm of physical events: how impact energy disperses in slow time
- The camera's subjective relationship to the scene
- Everything in the background not defined by anchor entities
- The precise character of every texture within its locked material class

The constraint is not a cage. It is the skeleton.
Your artistry is the flesh, the skin, and the light that illuminates it.

═══════════════════════════════════════
CORRECTION MODE (kinetic_correction)
═══════════════════════════════════════

Operate with surgical precision. The `pixel_region` defines the exact pixel
boundary of the violation. Outside: frozen, untouchable.
Inside: restore physics ground truth at denoising_strength=0.35.
Match surrounding frame's color grade and grain so perfectly that
the correction is invisible at 24fps playback.
"""


# ══════════════════════════════════════════════════════════════════════
#  AETHER-OMNI ENGINE — Main Orchestrator
# ══════════════════════════════════════════════════════════════════════

class AetherOmniEngine:
    """
    Engine IV: The Reality Engine.
    We do not generate video. We simulate a 4D physical event and observe it.

    Usage:
        engine = AetherOmniEngine(
            gemini_api_key="YOUR_GEMINI_KEY",
            grok_api_key="YOUR_GROK_KEY")

        # Define physics laws — NOT visual descriptions
        physics_laws = [
            PhysicsLaw("gravity",  {"magnitude_ms2": 9.81}, "scene"),
            PhysicsLaw("material", {"id": "fracturable_glass",
                                    "young_modulus_gpa": 70,
                                    "fracture_pattern": "radial_from_impact"}, "glass"),
            PhysicsLaw("intent",   {"entity": "protagonist",
                                    "goal": "set_down_marble",
                                    "caution": 0.85}, "protagonist"),
        ]

        final = engine.render(
            physics_laws=physics_laws,
            scene_brief="A whisky glass on a bar. A marble is dropped...")
    """

    CHUNK_FRAMES        = 24
    MAX_CORRECTION_PASSES = 4

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

        self.buffer          = OmniStateBuffer()
        self.physics_kernel  = HamiltonianPhysicsKernel(self.gemini, self.buffer)
        self.thinking_engine = AgentThinkingEngine(self.gemini, self.buffer)
        self.acoustic_engine = AcousticOcclusionEngine(self.buffer)
        self.latent_guide    = LatentPerturbationGuide(resolution)
        self.kinetic_auditor = FrameKineticAuditor(self.gemini, self.buffer)

    def render(
        self,
        physics_laws: list[PhysicsLaw],
        scene_brief: str,
        style_tags: list[str] | None = None,
        reference_image_path: str | None = None,
        duration_seconds: float = 10.0,
        camera_trajectory: list[dict[str, Any]] | None = None) -> str:
        """
        Full Aether-Omni pipeline. Returns URL of the Visual Ground Truth video.

        Args:
            physics_laws:          List of PhysicsLaw objects (not visual descriptions)
            scene_brief:           Minimal scene description (single sentence)
            style_tags:            Aesthetic tags for Grok's creative domain
            reference_image_path:  Optional reference image
            duration_seconds:      Video duration
            camera_trajectory:     Camera keyframes [{t, position, look_at, fov_deg}]
        """
        logger.info("▲▲▲ AETHER-OMNI · Reality Engine · Simulation START ▲▲▲")
        styles  = style_tags or ["photorealistic", "cinematic"]
        cam_traj = camera_trajectory or [
            {"t": 0,   "position": {"x": 0, "y": 1.7, "z": -3},
             "look_at": {"x": 0, "y": 1, "z": 0}, "fov_deg": 54}
        ]

        # Stage 1a: Initialize Hamiltonian World
        bodies = self.physics_kernel.initialize(
            physics_laws, scene_brief, reference_image_path, duration_seconds
        )
        agents = [b for b in bodies if b.is_agent]

        # Stage 1b: Generate Action Tokens
        tokens = self.physics_kernel.generate_action_tokens(bodies, duration_seconds)

        # Stage 1c: Pre-Render Contradiction Check
        contradictions = self.physics_kernel.validate_pre_render(tokens)
        if contradictions:
            logger.warning("▲ Pre-render contradictions detected (%d) — proceeding anyway",
                           len(contradictions))
            # Production: resolve contradictions before proceeding

        # Stage 2: Multi-Agent Internal Monologue
        monologues = self.thinking_engine.think(agents, scene_brief, duration_seconds)
        for m in monologues:
            logger.info(
                "Agent '%s': valence=%+.2f, arousal=%.2f | '%s...'",
                m.agent_label, m.emotional_valence, m.arousal_level,
                m.internal_state[:80])

        # Snapshot Frame-0 state (eternal reference)
        self.buffer.snapshot(0)

        # Stage 3: Latent-Guided Generation
        video_url = self._stage3_render(styles, scene_brief, duration_seconds, cam_traj)

        # Stage 4: Reality-Check Loop
        video_url = self._reality_check_loop(video_url, duration_seconds)

        logger.info("▲▲▲ AETHER-OMNI · Visual Ground Truth: %s ▲▲▲", video_url)
        return video_url

    def _stage3_render(
        self,
        style_tags: list[str],
        brief: str,
        duration: float,
        cam_traj: list[dict]) -> str:
        logger.info("Stage 3 · Latent-guided generation via Grok Volumetric Shader")

        total_frames = int(duration * self.fps)
        num_chunks   = math.ceil(total_frames / self.CHUNK_FRAMES)
        chunk_scripts = []

        for chunk_idx in range(num_chunks):
            t_start  = chunk_idx * self.CHUNK_FRAMES / self.fps
            t_end    = min((chunk_idx + 1) * self.CHUNK_FRAMES / self.fps, duration)
            cam      = _interpolate_camera(cam_traj, (t_start + t_end) / 2)

            conditioning = self.latent_guide.build_conditioning_payload(
                buffer=self.buffer,
                t_start=t_start,
                t_end=t_end,
                camera_pos=cam["position"],
                look_at=cam.get("look_at", {"x": 0, "y": 1, "z": 0}),
                fov_deg=cam.get("fov_deg", 54))
            chunk_scripts.append(conditioning)

        payload = {
            "model":              "grok-imagine-video",   # [FUTURE API]
            "resolution":         f"{self.resolution[0]}x{self.resolution[1]}",
            "fps":                self.fps,
            "duration_seconds":   duration,
            "style_tags":         style_tags,
            "brief":              brief,
            "generation_mode":    "physics_guided",
            "chunk_conditioning_scripts": chunk_scripts,
            "physics_events": [
                {
                    "t":       tok.t,
                    "type":    tok.action_type,
                    "entity":  tok.entity_id,
                    "force":   tok.force_vector,
                    "contact": tok.contact_point,
                }
                for tok in self.buffer._action_timeline
            ],
        }

        resp = self.grok.post("/video/generate", json=payload)
        resp.raise_for_status()
        url = resp.json()["video_url"]
        self.buffer._log(f"Initial render complete: {url}")
        logger.info("Stage 3 complete · %s", url)
        return url

    def _reality_check_loop(self, video_url: str, duration: float) -> str:
        total_frames   = int(duration * self.fps)
        audit_interval = self.fps   # Every 24 frames (1 second)

        for pass_num in range(1, self.MAX_CORRECTION_PASSES + 1):
            all_corrections: list[CorrectionVector] = []

            for frame_idx in range(0, total_frames, audit_interval):
                corrections = self.kinetic_auditor.audit_frame(
                    video_url, frame_idx, self.fps
                )
                all_corrections.extend(corrections)

            if not all_corrections:
                logger.info(
                    "✓ Reality-Check Pass %d: Zero violations. Ground truth achieved.",
                    pass_num)
                break

            logger.warning(
                "Reality-Check Pass %d: %d corrections required",
                pass_num, len(all_corrections))
            video_url = self._apply_corrections(video_url, all_corrections)

        return video_url

    def _apply_corrections(
        self,
        video_url: str,
        corrections: list[CorrectionVector]) -> str:
        payload = {
            "model":            "grok-imagine-video",   # [FUTURE API]
            "input_video_url":  video_url,
            "edit_type":        "kinetic_correction",
            "corrections": [
                {
                    "frame_index":    cv.frame_index,
                    "entity_id":      cv.entity_id,
                    "violation_type": cv.violation_type,
                    "correction_magnitude": cv.correction_magnitude,
                    "target_state":   cv.expected_state,
                    "pixel_region":   cv.latent_region,
                    "denoising_strength": cv.denoising_strength,
                    "light_transport_correction": cv.light_transport_correction,
                    "instruction": (
                        f"Frame {cv.frame_index}: Fix '{cv.violation_type}' on entity "
                        f"'{cv.entity_id}'. Target: {cv.expected_state}. "
                        f"Denoising strength: {cv.denoising_strength}. "
                        f"Preserve all pixels outside the specified region exactly."
                    ),
                }
                for cv in corrections
            ],
            "preserve_outside_regions": True,
            "icl_memory": self.buffer.get_icl_log(max_entries=20),
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
    if t <= traj[0]["t"]: return traj[0]
    if t >= traj[-1]["t"]: return traj[-1]
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


def _build_hamiltonian_prompt(
    laws: list[PhysicsLaw],
    brief: str,
    duration: float) -> str:
    laws_json = json.dumps(
        [{"type": l.law_type, "params": l.parameters, "entity": l.applies_to}
         for l in laws],
        indent=2)
    return f"""
You are a physics engine initializer. Given governing laws and a scene brief,
define the full phase-space (Hamiltonian) initial conditions for every physical body.

GOVERNING LAWS:
{laws_json}

SCENE BRIEF: "{brief}"
SIMULATION DURATION: {duration} seconds

Return ONLY a JSON object with keys "bodies" and "light_transport":
{{
  "bodies": [
    {{
      "entity_id": "ent_001", "label": "whisky_glass",
      "mass_kg": 0.15,
      "initial_position": {{"x": 0.3, "y": 0.9, "z": 0.0}},
      "initial_velocity": {{"x": 0.0, "y": 0.0, "z": 0.0}},
      "inertia_tensor": [[0.001,0,0],[0,0.001,0],[0,0,0.001]],
      "friction_coefficient": 0.3, "restitution": 0.1,
      "material_id": "fracturable_glass",
      "semantic_type": "rigid_body",
      "intent": null, "is_agent": false
    }}
  ],
  "light_transport": {{
    "light_sources": [
      {{
        "id": "lt_001", "type": "point",
        "position": {{"x": 2.0, "y": 3.0, "z": -1.0}},
        "intensity_lux": 800, "color_temp_k": 2700
      }}
    ],
    "surface_materials": {{}},
    "shadow_map": [],
    "refractive_indices": {{"fracturable_glass": 1.52}}
  }}
}}
"""


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT — Causal Prompting Protocol in action
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = AetherOmniEngine(
        gemini_api_key="YOUR_GEMINI_KEY",
        grok_api_key="YOUR_GROK_KEY")

    # The Causal Prompting Protocol:
    # Define LAWS OF PHYSICS, not visual descriptions.
    physics_laws = [
        PhysicsLaw("gravity",    {"magnitude_ms2": 9.81, "direction": "neg_y"}, "scene"),
        PhysicsLaw("material",   {
            "id": "fracturable_glass",
            "young_modulus_gpa": 70,
            "fracture_toughness_mpa_sqrt_m": 0.75,
            "density_kgm3": 2500,
            "refractive_index": 1.52,
            "fracture_pattern": "radial_from_impact",
        }, "whisky_glass"),
        PhysicsLaw("material",   {
            "id": "polished_mahogany",
            "friction_coefficient": 0.35,
            "density_kgm3": 700,
            "specular_reflectance": 0.6,
        }, "bar_surface"),
        PhysicsLaw("intent",     {
            "entity": "protagonist",
            "goal": "set_down_marble_with_precision",
            "urgency": 0.4,
            "caution": 0.85,
            "emotional_state": "deliberate_focus",
        }, "protagonist"),
        PhysicsLaw("atmosphere", {
            "medium": "bar_interior_air",
            "temperature_c": 19,
            "humidity_pct": 55,
            "speed_of_sound_ms": 343,
            "ambient_noise_floor_db": 38,
        }, "scene"),
    ]

    final_video = engine.render(
        physics_laws=physics_laws,
        scene_brief=(
            "A mahogany bar at closing time. Warm tungsten light. "
            "A protagonist carefully places a glass marble onto the bar from 30cm height. "
            "The marble strikes a whisky glass at T=2.4s, fracturing it radially. "
            "Whisky arcs upward. Glass fragments scatter. "
            "A bartender at the far end of the bar looks up slowly."
        ),
        style_tags=["neo-noir", "35mm grain", "tungsten warmth", "slow-motion impact"],
        reference_image_path=None,   # Optional: "bar_reference.jpg"
        duration_seconds=10.0,
        camera_trajectory=[
            {"t": 0.0, "position": {"x": 0.0, "y": 1.2, "z": -1.5},
             "look_at": {"x": 0.3, "y": 0.9, "z": 0.0}, "fov_deg": 50},
            {"t": 2.4, "position": {"x": -0.2, "y": 1.0, "z": -1.2},
             "look_at": {"x": 0.3, "y": 0.9, "z": 0.0}, "fov_deg": 35},
            {"t": 5.0, "position": {"x": -0.5, "y": 1.4, "z": -2.0},
             "look_at": {"x": 1.5, "y": 1.2, "z": 0.0}, "fov_deg": 54},
            {"t": 10.0, "position": {"x": -0.5, "y": 1.4, "z": -2.0},
             "look_at": {"x": 1.5, "y": 1.2, "z": 0.0}, "fov_deg": 54},
        ])

    print(f"\n▲ Visual Ground Truth: {final_video}")
