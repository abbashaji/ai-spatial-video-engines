"""
╔══════════════════════════════════════════════════════════════════════╗
║           ALETHEIA ENGINE  —  Engine V of V                         ║
║           Autotelic Neural-Physics & Style-Entropy Control          ║
║                                                                      ║
║  Paradigm:  Invent the physics of the scene from the style itself   ║
║  Gemini:    Style-Physics Analyst + Entropy Auditor                 ║
║  Grok:      Style-Aware Neural Renderer (style-physics constrained) ║
║  Threshold: 0.5% aesthetic entropy drift (tightest in series)       ║
║  Correction: Style-Entropy Correction Vectors + Quantum Collapse    ║
╚══════════════════════════════════════════════════════════════════════╝

ARCHITECTURE — 5 Stages with Autotelic Physics Discovery:
  Stage 1:  Entropy Audit           (Gemini — Style-Physics Manifest generation)
  Stage 2:  Neural-Hamiltonian Init  (Gemini — Phase space modified by SDP manifest)
  Stage 3:  Quantum Superposition   (Gemini — Multi-path latent ghost generation)
  Stage 4:  Aesthetic Collapse      (Gemini — Resonance scoring → path selection)
  Stage 5:  Zero-Latency Kinetic + Entropy Feedback Loop (Grok + Gemini)

THE PARADIGM SHIFT vs ALL PREVIOUS ENGINES:
  Engines I–III:  "Describe a scene. Check if Grok drew it correctly."
  Engine IV:      "Define real-world physics. Simulate what happens."
  Engine V:       "Analyze the artistic style. Let the style invent its own
                   physics. Simulate reality under those invented laws."

CORE INNOVATIONS OVER AETHER-OMNI (Engine IV):
  - Style-Differentiable Physics (SDP): gravity and friction constants derived
    from visual entropy of the reference style — not hardcoded at 9.81 m/s².
  - Narrative-Causal Entanglement (NCE): environment mental state modulates
    the Light Transport Matrix in real time with the narrative arc.
  - Quantum Latent Superposition: Gemini simulates N outcome paths; only the
    path with highest Aesthetic Resonance Score is collapsed into final video.
  - Aesthetic Entropy Lock: SHA-256 of the Style-Physics Manifest is appended
    to every chunk prompt; entropy drift is a measurable, correctable violation.
  - Moody Room Protocol: character emotional valence warps shadow stretch
    vectors, light color temperature, and spatial sound reverb tail length.
  - Style-Physics Manifest: the new primary output of Stage 1, replacing the
    simple PhysicsLaw list as the scene's governing document.

NEW INPUT PRIMITIVE — StyleDirective:
  Instead of PhysicsLaw("gravity", {"magnitude_ms2": 9.81}), you write:
    StyleDirective(
        reference="Spider-Man: Across the Spider-Verse — Miles Morales",
        emotional_arc=[("melancholy", 0.0), ("resolve", 5.0), ("triumph", 10.0)],
        quantum_paths=3)
  Gemini derives the physics from the art direction.

WHAT IS REAL TODAY vs FUTURE API:
  Real:    Style entropy computation, SDP manifest generation, NCE modeling,
           quantum path generation/scoring, acoustic mood entanglement,
           material hash, ICL log, multi-agent monologue, all Euler integration,
           aesthetic resonance scoring, entropy auditor, all Gemini API calls
  Future:  grok-imagine-video API, deep_guidance attention injection,
           true latent superposition endpoints, video-to-video correction API

See README.md § Engine V for full documentation.
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
import numpy as np                     # pip install numpy

logger = logging.getLogger("aletheia")
logging.basicConfig(level=logging.INFO, format="%(name)s [%(levelname)s] %(message)s")


# ══════════════════════════════════════════════════════════════════════
#  TYPE ALIASES
# ══════════════════════════════════════════════════════════════════════

Vec3 = dict[str, float]    # {"x": float, "y": float, "z": float}
Mat3 = list[list[float]]   # 3×3 matrix


# ══════════════════════════════════════════════════════════════════════
#  PRIMARY INPUT PRIMITIVE — StyleDirective
#  The new "Causal Prompting" of Engine V.
#  Instead of specifying physics laws, you specify artistic intention.
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StyleDirective:
    """
    The primary input primitive of Aletheia. Replaces PhysicsLaw lists.

    You specify the artistic intention; Gemini derives the physics.

    Args:
        reference:      Reference work, artist, or visual style description.
                        e.g. "Spider-Man: Across the Spider-Verse — Miles Morales"
                             "Impressionist oil painting — Monet water lilies"
                             "Neon-noir cyberpunk — Blade Runner 2049 rain scene"
        emotional_arc:  List of (emotional_state, timestamp_seconds) tuples
                        that define the narrative arc for NCE modulation.
                        e.g. [("isolation", 0.0), ("longing", 4.0), ("resolve", 8.0)]
        quantum_paths:  Number of alternate physical outcome paths to generate
                        and score before collapsing to final video. Range: 2–5.
                        More paths → higher aesthetic quality, longer generation.
        scene_brief:    Minimal description of what happens (single sentence).
        style_tags:     Additional Grok rendering directives (aesthetics only).

    Example:
        StyleDirective(
            reference="Studio Ghibli — My Neighbor Totoro forest scene",
            emotional_arc=[("wonder", 0.0), ("awe", 3.0), ("peace", 8.0)],
            quantum_paths=3,
            scene_brief="A girl drops a leaf into a stream and watches it float away.",
            style_tags=["watercolor wash", "dappled light", "soft cel shading"])
    """
    reference: str
    emotional_arc: list[tuple[str, float]]   # [(emotion_label, t_seconds), ...]
    quantum_paths: int = 3
    scene_brief: str = ""
    style_tags: list[str] = field(default_factory=list)
    directive_id: str = field(default_factory=lambda: f"sd_{uuid.uuid4().hex[:6]}")


# ══════════════════════════════════════════════════════════════════════
#  STYLE-PHYSICS MANIFEST
#  The central output of Stage 1 (Entropy Audit).
#  This replaces hardcoded physics constants with style-derived ones.
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StylePhysicsManifest:
    """
    The laws of physics for this specific artistic style.

    This is Engine V's core primitive. Instead of:
        gravity = 9.81 m/s²  (Newton's law)

    Aletheia computes:
        gravity = "painterly_descent_coefficient" derived from the visual
        entropy and motion vocabulary of the reference style.

    For Spider-Verse: gravity might be 8.2 m/s² with a 6-frame hold on
    impact frames (matching the film's stylized physics timing).
    For Monet: gravity is replaced by "paint viscosity flow" (0.03 m/s downward
    drift) representing how paint physically moves on the canvas.

    Fields:
        style_gravity:      Derived gravitational constant (m/s² equivalent).
                            May deviate from 9.81 to match stylistic motion vocabulary.
        viscosity_constant: The "fluid dynamics" of the rendering medium.
                            Oil paint: high (0.8). Watercolor: medium (0.5). Cel: low (0.1).
        style_entropy_hash: SHA-256 of the complete manifest. Every chunk is checked
                            against this hash — drift is a measurable violation.
        light_entanglement_matrix: Maps emotional_arc states to light transport modifiers.
                            {"melancholy": {"shadow_stretch_factor": 1.4, "color_temp_k_delta": -800}}
        fracture_vocabulary: How objects break in this style.
                            {"cubist": "geometric_shards"}, {"impressionist": "color_dissolution"}
        motion_vocabulary:  How movement looks in this style.
                            {"spider_verse": "6_frame_hold_on_impact"}
        aesthetic_entropy_target: The target entropy value for the style.
                            Gemini locks all frames to this entropy level.
        narrative_physics_map: How each emotional state warps the physics.
        medium_properties:  The physical properties of the artistic medium itself.
    """
    manifest_id: str
    reference_style: str
    style_gravity: float                            # m/s² equivalent for this style
    viscosity_constant: float                       # 0.0 (digital crisp) to 1.0 (thick oil)
    style_entropy_hash: str                         # SHA-256 of manifest — the Aesthetic Lock
    light_entanglement_matrix: dict[str, Any]       # emotion → light physics modifiers
    fracture_vocabulary: dict[str, str]             # material → style-specific break pattern
    motion_vocabulary: dict[str, Any]               # movement style descriptors
    aesthetic_entropy_target: float                 # target entropy level (0.0–1.0)
    narrative_physics_map: dict[str, dict[str, Any]]  # emotion_label → physics overrides
    medium_properties: dict[str, Any]              # artistic medium physical properties
    raw_entropy_analysis: str                       # Gemini's full reasoning output
    created_at: float = field(default_factory=time.time)

    def recompute_hash(self) -> str:
        """Recompute the style entropy hash for drift detection."""
        core = {
            "style_gravity": self.style_gravity,
            "viscosity_constant": self.viscosity_constant,
            "light_entanglement_matrix": self.light_entanglement_matrix,
            "fracture_vocabulary": self.fracture_vocabulary,
            "motion_vocabulary": self.motion_vocabulary,
            "aesthetic_entropy_target": self.aesthetic_entropy_target,
        }
        return hashlib.sha256(
            json.dumps(core, sort_keys=True).encode()
        ).hexdigest()


# ══════════════════════════════════════════════════════════════════════
#  QUANTUM LATENT PATH
#  One of N possible outcome paths generated before aesthetic collapse.
# ══════════════════════════════════════════════════════════════════════

@dataclass
class QuantumLatentPath:
    """
    A single potential outcome path in the Quantum Latent Superposition.

    Aletheia generates N of these for each significant physical event
    (e.g., a glass breaking into N different fracture patterns).
    Only the path with the highest Aesthetic Resonance Score (ARS) is
    collapsed into the final video; others become "ghost frames" — latent
    memory that informs the entropy lock even though they aren't rendered.

    Fields:
        path_id:            Unique identifier for this path.
        event_description:  What physical event this path represents.
        style_outcome:      How this path resolves in the reference style's language.
        hamiltonian_delta:  How this path deviates from the base Hamiltonian state.
        aesthetic_resonance_score: Gemini's resonance score (0.0–1.0).
                                   Highest score wins.
        ghost_frames:       List of "latent ghost" frame descriptions — recorded
                            even for non-selected paths to inform entropy lock.
        is_selected:        True if this path was collapsed into final video.
    """
    path_id: str
    event_description: str
    style_outcome: str                  # e.g. "glass shatters into cubist shards"
    hamiltonian_delta: dict[str, Any]   # position/momentum deltas from base physics
    aesthetic_resonance_score: float    # 0.0 (jarring) to 1.0 (perfect style fit)
    ghost_frames: list[dict[str, Any]]  # non-rendered paths stay in entropy memory
    is_selected: bool = False


@dataclass
class QuantumEvent:
    """
    A physical event that has multiple stylistically valid outcomes.
    Container for N QuantumLatentPaths — one per possible outcome.
    """
    event_id: str
    t: float                            # onset time (seconds)
    entity_id: str
    event_type: str                     # "fracture", "collision", "fluid_flow", etc.
    paths: list[QuantumLatentPath]
    selected_path_id: str | None = None # set after aesthetic collapse


# ══════════════════════════════════════════════════════════════════════
#  NARRATIVE-CAUSAL ENTANGLEMENT (NCE)
#  The environment's "mental state" — light physics entangled with narrative.
# ══════════════════════════════════════════════════════════════════════

@dataclass
class EnvironmentMentalState:
    """
    The emotional state of the environment itself, derived from the narrative arc.

    In Engine IV, only characters have cognitive states (AgentMonologue).
    In Aletheia, the environment has a mental state too — and that mental state
    modulates the Light Transport Matrix directly.

    The "Moody Room Protocol":
        - Sad character + NCE active → shadows stretch toward the character
          (shadow_stretch_factor > 1.0 in the direction of the character)
        - Triumphant character + NCE active → light color temperature rises,
          ambient fill increases, shadows soften
        - Fearful character + NCE active → light sources subtly contract,
          shadows elongate and gain hard edges (high shadow_contrast)

    This is not a visual filter. It is a physics modification:
    the Light Transport Matrix is recomputed with the emotional modifiers
    applied to the actual source positions and intensities.
    """
    t: float
    emotional_state: str                # from the narrative arc
    emotional_intensity: float          # 0.0–1.0
    shadow_stretch_factor: float        # 1.0 = normal; >1 = stretched toward agent
    shadow_stretch_direction: Vec3      # unit vector toward dominant agent
    color_temp_k_delta: float           # ± Kelvin shift from neutral
    ambient_fill_multiplier: float      # 0.5 (dim, moody) to 2.0 (bright, joyful)
    shadow_contrast: float              # 0.0 (soft) to 1.0 (hard-edged)
    reverb_tail_seconds: float          # acoustic tail length (more = more longing)
    light_transport_modifiers: dict[str, Any]  # complete LT override payload


# ══════════════════════════════════════════════════════════════════════
#  STYLE-DIFFERENTIABLE ACTION TOKEN
#  Extension of Engine IV's ActionToken with style-physics overrides.
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StyleActionToken:
    """
    A physical action token that respects style-derived physics instead of
    (or in addition to) Newtonian physics.

    Key difference from ActionToken (Engine IV):
        In Engine IV: force_vector uses real-world Newtons, real gravity.
        In Engine V:  force_vector is modulated by style_physics_override,
                      which applies the Style-Physics Manifest's viscosity,
                      style_gravity, and motion vocabulary.

    Example (Spider-Verse style):
        A ball dropped from 1m in real physics falls with acceleration 9.81 m/s².
        In Spider-Verse physics, it falls with acceleration 8.2 m/s² BUT holds
        for 6 frames on the impact frame (impact_hold_frames=6), creating the
        stylized "pose hold" characteristic of that film's visual grammar.
    """
    token_id: str
    entity_id: str
    t: float
    duration: float
    action_type: Literal[
        "torque", "contact_force", "locomotion", "grasp",
        "release", "deform", "fracture", "fluid_flow", "thermal_transfer",
        "style_drift",    # NEW: deliberate style-driven non-Newtonian motion
        "entropy_hold",   # NEW: frame-hold for impact/emotion beats
    ]
    force_vector: Vec3
    contact_point: Vec3 | None
    resulting_acceleration: Vec3            # in style-physics space
    physical_constraint: str | None
    contradiction_check: str | None
    # NEW in Engine V:
    style_physics_override: dict[str, Any] = field(default_factory=dict)
    """
    Overrides applied by the Style-Physics Manifest:
    {
        "viscosity_damping": 0.3,      # reduces acceleration by 30% (painterly drag)
        "impact_hold_frames": 6,       # freeze motion for N frames at impact
        "motion_smear_factor": 1.5,    # motion blur multiplier (comics style)
        "entropy_target": 0.72,        # target entropy for this motion event
    }
    """
    quantum_path_id: str | None = None  # which QuantumLatentPath this token belongs to
    confidence: float = 1.0


@dataclass
class AestheticViolation:
    """
    An aesthetic entropy violation detected by the Entropy Auditor.

    This is Engine V's expansion of CorrectionVector (Engine IV).
    In addition to physics violations (floating, shadow direction),
    Aletheia checks for Style Drift — when rendered pixels deviate from
    the aesthetic entropy target defined in the Style-Physics Manifest.
    """
    frame_index: int
    t: float
    entity_id: str | None               # None = scene-wide entropy violation
    violation_type: Literal[
        "floating_contact", "shadow_direction", "momentum_violation",
        "material_contradiction", "thermal_emission", "acoustic_occlusion",
        # NEW in Engine V:
        "style_entropy_drift",      # pixel motion too "realistic" for the style
        "nce_violation",            # light transport ignoring emotional state
        "aesthetic_persistence",    # style feel drifted from manifest target
        "fracture_vocabulary",      # object broke in wrong style (realistic vs cubist)
        "motion_vocabulary",        # movement style deviated (too fluid for cel)
    ]
    expected_state: dict[str, Any]
    observed_state: dict[str, Any]
    style_correction_vector: dict[str, Any]  # what Gemini prescribes to fix it
    correction_magnitude: float
    latent_region: dict[str, Any]
    entropy_delta: float = 0.0          # how far entropy drifted from target
    denoising_strength: float = 0.30    # slightly lower than IV for subtlety


# ══════════════════════════════════════════════════════════════════════
#  ALETHEIA STATE BUFFER
#  Extension of OmniStateBuffer with style-entropy and NCE state.
# ══════════════════════════════════════════════════════════════════════

class AletheiaStateBuffer:
    """
    The 5D world state: 4D physical + 1D style-entropy.

    Five coupled sub-buffers:
      1. StylePhysicsBuffer     — the manifest and all style-derived constants
      2. HamiltonianBuffer      — physics phase space (q, p per body)
      3. NCEBuffer              — narrative-causal entanglement states per timestamp
      4. QuantumEventBuffer     — all quantum events and their path states
      5. Agent Cognitive States — per-character monologues and intent

    All five are kept in sync. The entropy auditor queries the StylePhysicsBuffer
    on every frame to verify aesthetic persistence.
    """

    def __init__(self) -> None:
        # Style-physics layer
        self._manifest: StylePhysicsManifest | None = None
        self._style_entropy_history: list[dict[str, Any]] = []

        # Hamiltonian physics state
        self._hamiltonian: dict[str, dict[str, Any]] = {}
        self._action_timeline: list[StyleActionToken] = []

        # NCE — Narrative-Causal Entanglement
        self._nce_states: list[EnvironmentMentalState] = []

        # Quantum events
        self._quantum_events: list[QuantumEvent] = []
        self._ghost_frame_memory: list[dict[str, Any]] = []  # non-selected paths

        # Agent cognitive states
        self._monologues: dict[str, dict[str, Any]] = {}

        # ICL memory log (append-only)
        self._event_log: list[str] = []

        # Frame snapshots
        self._snapshots: dict[int, dict[str, Any]] = {}

    # ── Style-Physics Buffer ───────────────────────────────────────────

    def register_manifest(self, manifest: StylePhysicsManifest) -> None:
        self._manifest = manifest
        self._log(
            f"Style-Physics Manifest registered: '{manifest.reference_style}' · "
            f"style_gravity={manifest.style_gravity:.3f} m/s² · "
            f"viscosity={manifest.viscosity_constant:.2f} · "
            f"entropy_target={manifest.aesthetic_entropy_target:.3f} · "
            f"entropy_hash={manifest.style_entropy_hash[:12]}..."
        )

    def get_manifest(self) -> StylePhysicsManifest:
        if not self._manifest:
            raise RuntimeError("No Style-Physics Manifest registered. Run Entropy Audit first.")
        return self._manifest

    def record_entropy_measurement(
        self,
        t: float,
        measured_entropy: float,
        frame_index: int) -> None:
        target = self._manifest.aesthetic_entropy_target if self._manifest else 0.5
        delta = abs(measured_entropy - target)
        self._style_entropy_history.append({
            "t": t,
            "frame_index": frame_index,
            "measured": measured_entropy,
            "target": target,
            "delta": delta,
        })

    # ── Hamiltonian Buffer ─────────────────────────────────────────────

    def register_body(self, body: dict[str, Any]) -> None:
        self._hamiltonian[body["entity_id"]] = body
        self._log(
            f"Body registered: '{body['label']}' · {body['mass_kg']}kg · "
            f"material={body['material_id']} · style_physics=active"
        )

    def push_action_token(self, token: StyleActionToken) -> None:
        self._action_timeline.append(token)
        self._action_timeline.sort(key=lambda t: t.t)
        self._log(
            f"StyleAction [{token.action_type}] on '{token.entity_id}' "
            f"at T={token.t:.3f}s · force={token.force_vector} "
            f"· override={token.style_physics_override}"
        )

    def get_expected_position(self, entity_id: str, t: float) -> Vec3:
        """
        Style-aware Euler integration.

        Key difference from Engine IV: the gravitational acceleration uses
        style_gravity from the manifest, not 9.81 m/s². Viscosity damping
        is applied to all velocity components on each step.
        """
        body = self._hamiltonian.get(entity_id)
        if not body:
            return {"x": 0.0, "y": 0.0, "z": 0.0}

        manifest = self._manifest
        style_g  = manifest.style_gravity if manifest else -9.81
        viscosity = manifest.viscosity_constant if manifest else 0.0

        vx = body.get("initial_velocity", {}).get("x", 0.0)
        vy = body.get("initial_velocity", {}).get("y", 0.0)
        vz = body.get("initial_velocity", {}).get("z", 0.0)
        x, y, z = (
            body["initial_position"]["x"],
            body["initial_position"]["y"],
            body["initial_position"]["z"])

        relevant_tokens = [
            tok for tok in self._action_timeline
            if tok.entity_id == entity_id and tok.t <= t
        ]

        dt_sim = 0.01
        t_sim  = 0.0
        while t_sim < t:
            dt = min(dt_sim, t - t_sim)

            # Style-derived gravity (not necessarily 9.81)
            vy += (-abs(style_g)) * dt

            # Viscosity damping — painterly drag
            damping = 1.0 - viscosity * dt
            vx *= max(damping, 0.0)
            vy *= max(damping + 0.1, 0.0)   # vertical viscosity slightly less
            vz *= max(damping, 0.0)

            # Apply style action tokens
            for tok in relevant_tokens:
                if tok.t <= t_sim <= tok.t + tok.duration:
                    # Check for entropy_hold (stylized frame-freeze)
                    if tok.action_type == "entropy_hold":
                        pass  # velocity frozen during hold
                    else:
                        override_damping = tok.style_physics_override.get(
                            "viscosity_damping", 0.0
                        )
                        eff_accel = 1.0 - override_damping
                        vx += tok.resulting_acceleration["x"] * dt * eff_accel
                        vy += tok.resulting_acceleration["y"] * dt * eff_accel
                        vz += tok.resulting_acceleration["z"] * dt * eff_accel

            x += vx * dt
            y += vy * dt
            z += vz * dt
            y  = max(y, 0.0)
            t_sim += dt

        return {"x": round(x, 4), "y": round(y, 4), "z": round(z, 4)}

    # ── NCE Buffer ─────────────────────────────────────────────────────

    def register_nce_state(self, state: EnvironmentMentalState) -> None:
        self._nce_states.append(state)
        self._log(
            f"NCE state: '{state.emotional_state}' · intensity={state.emotional_intensity:.2f} · "
            f"shadow_stretch={state.shadow_stretch_factor:.2f} · "
            f"color_temp_delta={state.color_temp_k_delta:+.0f}K · "
            f"reverb_tail={state.reverb_tail_seconds:.2f}s"
        )

    def get_nce_state_at(self, t: float) -> EnvironmentMentalState | None:
        """Return the NCE state closest to time t."""
        if not self._nce_states:
            return None
        return min(self._nce_states, key=lambda s: abs(s.t - t))

    # ── Quantum Event Buffer ───────────────────────────────────────────

    def register_quantum_event(self, event: QuantumEvent) -> None:
        self._quantum_events.append(event)
        self._log(
            f"Quantum event: '{event.event_type}' on '{event.entity_id}' at T={event.t:.2f}s "
            f"· {len(event.paths)} paths generated"
        )

    def collapse_quantum_event(self, event_id: str, selected_path_id: str) -> None:
        for event in self._quantum_events:
            if event.event_id == event_id:
                event.selected_path_id = selected_path_id
                for path in event.paths:
                    if path.path_id != selected_path_id:
                        # Ghost frames enter entropy memory
                        self._ghost_frame_memory.extend(path.ghost_frames)
                path.is_selected = (path.path_id == selected_path_id)
                self._log(
                    f"Quantum collapse: event '{event_id}' → path '{selected_path_id}' selected"
                )
                return

    def get_ghost_entropy_context(self) -> list[dict[str, Any]]:
        """Ghost frames from non-selected paths inform the entropy lock."""
        return self._ghost_frame_memory[-50:]  # last 50 ghost frames

    # ── Monologues ─────────────────────────────────────────────────────

    def register_monologue(self, monologue: dict[str, Any]) -> None:
        self._monologues[monologue["agent_id"]] = monologue

    # ── Snapshots & ICL Log ────────────────────────────────────────────

    def snapshot(self, frame_index: int) -> None:
        manifest_hash = self._manifest.style_entropy_hash[:12] if self._manifest else "none"
        self._snapshots[frame_index] = {
            "frame_index": frame_index,
            "hamiltonian": {
                eid: {
                    "position": body["initial_position"],
                    "material": body["material_id"],
                }
                for eid, body in self._hamiltonian.items()
            },
            "style_entropy_hash": manifest_hash,
            "nce_state": {
                "emotional_state": nce.emotional_state,
                "shadow_stretch": nce.shadow_stretch_factor,
            } if (nce := self.get_nce_state_at(frame_index / 24)) else {},
            "monologue_states": {
                aid: {
                    "valence": m.get("emotional_valence", 0.0),
                    "arousal": m.get("arousal_level", 0.5),
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
        return self._snapshots.get(closest, {}) if closest is not None else {}

    def _log(self, entry: str) -> None:
        self._event_log.append(f"[{time.strftime('%H:%M:%S')}] {entry}")

    def get_icl_log(self, max_entries: int = 60) -> str:
        return "\n".join(self._event_log[-max_entries:])

    def export_for_rendering(self, t_start: float, t_end: float) -> dict[str, Any]:
        manifest = self._manifest
        nce = self.get_nce_state_at((t_start + t_end) / 2)

        return {
            "style_physics_manifest": {
                "style_gravity": manifest.style_gravity if manifest else -9.81,
                "viscosity_constant": manifest.viscosity_constant if manifest else 0.0,
                "style_entropy_hash": manifest.style_entropy_hash[:16] if manifest else "",
                "aesthetic_entropy_target": manifest.aesthetic_entropy_target if manifest else 0.5,
                "fracture_vocabulary": manifest.fracture_vocabulary if manifest else {},
                "motion_vocabulary": manifest.motion_vocabulary if manifest else {},
                "medium_properties": manifest.medium_properties if manifest else {},
            } if manifest else {},
            "hamiltonian_states": {
                eid: {
                    "label": body["label"],
                    "mass_kg": body["mass_kg"],
                    "material_id": body["material_id"],
                    "intent": body.get("intent"),
                    "expected_position_t_start": self.get_expected_position(eid, t_start),
                    "expected_position_t_end": self.get_expected_position(eid, t_end),
                }
                for eid, body in self._hamiltonian.items()
            },
            "style_action_tokens": [
                {
                    "entity_id": tok.entity_id,
                    "action_type": tok.action_type,
                    "t": tok.t,
                    "duration": tok.duration,
                    "force_vector": tok.force_vector,
                    "style_physics_override": tok.style_physics_override,
                    "quantum_path_id": tok.quantum_path_id,
                }
                for tok in self._action_timeline
                if t_start <= tok.t <= t_end
            ],
            "nce_state": {
                "emotional_state": nce.emotional_state,
                "shadow_stretch_factor": nce.shadow_stretch_factor,
                "shadow_stretch_direction": nce.shadow_stretch_direction,
                "color_temp_k_delta": nce.color_temp_k_delta,
                "ambient_fill_multiplier": nce.ambient_fill_multiplier,
                "shadow_contrast": nce.shadow_contrast,
                "reverb_tail_seconds": nce.reverb_tail_seconds,
                "light_transport_modifiers": nce.light_transport_modifiers,
            } if nce else {},
            "agent_states": {
                aid: {
                    "internal_state": m.get("internal_state", "")[:120],
                    "emotional_valence": m.get("emotional_valence", 0.0),
                    "arousal_level": m.get("arousal_level", 0.5),
                    "micro_expression_cues": m.get("micro_expression_cues", []),
                }
                for aid, m in self._monologues.items()
            },
            "ghost_entropy_context": self.get_ghost_entropy_context(),
            "icl_memory": self.get_icl_log(),
            "frame_0_state": self.get_snapshot(0),
        }


# ══════════════════════════════════════════════════════════════════════
#  STAGE 1: ENTROPY AUDITOR — Style-Physics Manifest Generation
#  Gemini analyzes the reference style and derives custom physics laws.
# ══════════════════════════════════════════════════════════════════════

class EntropyAuditor:
    """
    Stage 1 of Aletheia. Analyzes the reference style and produces
    the Style-Physics Manifest — the governing document of this video.

    The Entropy Audit is the most novel step in Engine V. It asks:
    "What are the PHYSICS of this artistic style?"

    For Spider-Verse: physics have stylized gravity (8.2 m/s²), 6-frame
    impact holds, motion smear lines, halftone dot patterns on shadows.
    These are real physical properties of the animation style.

    For Impressionist painting: the "physics" are paint viscosity
    (brush stroke drag), chromatic bleed (color diffusion into adjacent areas),
    impasto relief (physical 3D texture of thick paint application).

    Gemini ER 1.5 performs Visual Entropy analysis: it reads the reference
    style's motion vocabulary, rendering grammar, and physical metaphors,
    then derives numerical constants that represent those as physics laws.
    """

    def __init__(
        self,
        gemini: GeminiERClient,
        buffer: AletheiaStateBuffer) -> None:
        self.gemini = gemini
        self.buffer = buffer

    def audit(
        self,
        directive: StyleDirective,
        reference_image_path: str | None = None) -> StylePhysicsManifest:
        """
        Perform the Entropy Audit and generate the Style-Physics Manifest.

        This is a single large Gemini call with a high thinking budget.
        The output is the Manifest that governs all subsequent stages.
        """
        logger.info("EntropyAuditor: Analyzing style '%s'...", directive.reference)

        prompt = _build_entropy_audit_prompt(directive)
        content: list[Any] = []

        if reference_image_path:
            with open(reference_image_path, "rb") as f:
                content.append({"mime_type": "image/jpeg", "data": f.read()})
        content.append(prompt)

        response = self.gemini.generate_content(
            content)

        data: dict = json.loads(_extract_json(response.text))

        # Build the manifest
        manifest_core = {
            "style_gravity": data.get("style_gravity", -9.81),
            "viscosity_constant": data.get("viscosity_constant", 0.1),
            "light_entanglement_matrix": data.get("light_entanglement_matrix", {}),
            "fracture_vocabulary": data.get("fracture_vocabulary", {}),
            "motion_vocabulary": data.get("motion_vocabulary", {}),
            "aesthetic_entropy_target": data.get("aesthetic_entropy_target", 0.5),
            "narrative_physics_map": data.get("narrative_physics_map", {}),
            "medium_properties": data.get("medium_properties", {}),
        }

        # Compute the style entropy hash — the Aesthetic Lock
        entropy_hash = hashlib.sha256(
            json.dumps(manifest_core, sort_keys=True).encode()
        ).hexdigest()

        manifest = StylePhysicsManifest(
            manifest_id=f"manifest_{uuid.uuid4().hex[:8]}",
            reference_style=directive.reference,
            style_gravity=manifest_core["style_gravity"],
            viscosity_constant=manifest_core["viscosity_constant"],
            style_entropy_hash=entropy_hash,
            light_entanglement_matrix=manifest_core["light_entanglement_matrix"],
            fracture_vocabulary=manifest_core["fracture_vocabulary"],
            motion_vocabulary=manifest_core["motion_vocabulary"],
            aesthetic_entropy_target=manifest_core["aesthetic_entropy_target"],
            narrative_physics_map=manifest_core["narrative_physics_map"],
            medium_properties=manifest_core["medium_properties"],
            raw_entropy_analysis=response.text)

        self.buffer.register_manifest(manifest)
        logger.info(
            "EntropyAuditor: Manifest generated · style_gravity=%.3f · "
            "viscosity=%.2f · entropy_target=%.3f · hash=%s...",
            manifest.style_gravity, manifest.viscosity_constant,
            manifest.aesthetic_entropy_target, entropy_hash[:12])
        return manifest


# ══════════════════════════════════════════════════════════════════════
#  STAGE 2: NEURAL-HAMILTONIAN INITIALIZATION
#  Builds phase space modified by the Style-Physics Manifest.
# ══════════════════════════════════════════════════════════════════════

class NeuralHamiltonianKernel:
    """
    Stage 2 of Aletheia. Extends Engine IV's HamiltonianPhysicsKernel with
    Style-Physics Manifest integration.

    The key difference:
        Engine IV: Hamiltonian phase space uses real physics constants.
        Engine V:  Hamiltonian phase space is MODIFIED by the manifest.
                   style_gravity replaces 9.81, viscosity is added as a
                   damping term, and the motion_vocabulary introduces new
                   action types like "entropy_hold" and "style_drift".

    Also generates NCE states from the emotional_arc in the StyleDirective,
    so the Light Transport Matrix is primed for narrative entanglement
    before a single frame is rendered.
    """

    CONTRADICTION_CHECKS = [
        "No entity may be airborne without style-gravity × mass upward force.",
        "Foot contact events must have ground_contact=true in the action token.",
        "Shadows must comply with NCE modifiers AND light source geometry.",
        "Fractures must follow the fracture_vocabulary defined in the manifest.",
        "Liquids must follow steepest descent modulated by viscosity_constant.",
        "Thermal emission must be directionally consistent with light entanglement.",
        "Style entropy must not deviate >0.5% from aesthetic_entropy_target.",
    ]

    def __init__(
        self,
        gemini: GeminiERClient,
        buffer: AletheiaStateBuffer) -> None:
        self.gemini = gemini
        self.buffer = buffer

    def initialize(
        self,
        directive: StyleDirective,
        manifest: StylePhysicsManifest,
        reference_image_path: str | None = None,
        duration_seconds: float = 10.0) -> list[dict[str, Any]]:
        """Stage 2a: Initialize style-modified Hamiltonian bodies."""
        prompt = _build_neural_hamiltonian_prompt(directive, manifest, duration_seconds)
        content: list[Any] = []

        if reference_image_path:
            with open(reference_image_path, "rb") as f:
                content.append({"mime_type": "image/jpeg", "data": f.read()})
        content.append(prompt)

        response = self.gemini.generate_content(
            content)

        data: dict = json.loads(_extract_json(response.text))
        bodies = []

        for bd in data.get("bodies", []):
            body = {
                "entity_id": bd["entity_id"],
                "label": bd["label"],
                "mass_kg": bd["mass_kg"],
                "initial_position": bd["initial_position"],
                "initial_velocity": bd.get("initial_velocity", {"x": 0, "y": 0, "z": 0}),
                "friction_coefficient": bd.get("friction_coefficient", 0.4),
                "restitution": bd.get("restitution", 0.3),
                "material_id": bd.get("material_id", "generic"),
                "semantic_type": bd.get("semantic_type", "rigid_body"),
                "intent": bd.get("intent"),
                "is_agent": bd.get("is_agent", False),
                # Style-physics additions:
                "style_material_id": bd.get("style_material_id", bd.get("material_id", "generic")),
                "entropy_class": bd.get("entropy_class", "neutral"),
            }
            self.buffer.register_body(body)
            bodies.append(body)

        # Generate NCE states from emotional arc
        nce_states = self._generate_nce_states(directive.emotional_arc, manifest, bodies)
        for nce in nce_states:
            self.buffer.register_nce_state(nce)

        logger.info(
            "NeuralHamiltonianKernel: %d bodies · %d NCE states",
            len(bodies), len(nce_states))
        return bodies

    def generate_style_action_tokens(
        self,
        bodies: list[dict[str, Any]],
        manifest: StylePhysicsManifest,
        duration_seconds: float) -> list[StyleActionToken]:
        """Stage 2b: Generate StyleActionTokens using style-derived physics."""
        body_summary = [
            {
                "entity_id": b["entity_id"],
                "label": b["label"],
                "mass_kg": b["mass_kg"],
                "material_id": b["material_id"],
                "intent": b.get("intent"),
                "is_agent": b.get("is_agent", False),
                "initial_position": b["initial_position"],
            }
            for b in bodies
        ]

        prompt = f"""
You are a Style-Physics simulation kernel generating StyleActionTokens.
These tokens use STYLE-DERIVED physics from the manifest, not Newtonian constants.

STYLE-PHYSICS MANIFEST:
  style_gravity:        {manifest.style_gravity:.3f} m/s² (NOT 9.81)
  viscosity_constant:   {manifest.viscosity_constant:.2f} (motion damping)
  motion_vocabulary:    {json.dumps(manifest.motion_vocabulary, indent=2)}
  fracture_vocabulary:  {json.dumps(manifest.fracture_vocabulary, indent=2)}
  medium_properties:    {json.dumps(manifest.medium_properties, indent=2)}

REGISTERED BODIES: {json.dumps(body_summary, indent=2)}

CONTRADICTION CHECKS (every token must pass ALL of these):
{chr(10).join(f"  {i+1}. {c}" for i, c in enumerate(self.CONTRADICTION_CHECKS))}

Generate ALL style action tokens for the {duration_seconds}-second simulation.
Include standard physical actions (locomotion, fracture, etc.) AND style-specific
actions (entropy_hold for impact beats, style_drift for non-Newtonian moments).

Return ONLY a JSON array:
[
  {{
    "token_id": "stok_001",
    "entity_id": "ent_protagonist",
    "t": 1.2, "duration": 0.3,
    "action_type": "locomotion",
    "force_vector": {{"x": 0.0, "y": 0.0, "z": 15.0}},
    "contact_point": {{"x": 0.1, "y": 0.0, "z": 0.0}},
    "resulting_acceleration": {{"x": 0.0, "y": 0.0, "z": 0.21}},
    "physical_constraint": "right_foot_contact_floor_t=1.2",
    "contradiction_check": "foot_contact=true, style_gravity_applied",
    "style_physics_override": {{
      "viscosity_damping": 0.15,
      "impact_hold_frames": 0,
      "motion_smear_factor": 1.0,
      "entropy_target": {manifest.aesthetic_entropy_target:.3f}
    }},
    "confidence": 0.97
  }}
]
"""
        response = self.gemini.generate_content(
            prompt)
        token_data: list[dict] = json.loads(_extract_json(response.text))

        tokens = []
        for td in token_data:
            tok = StyleActionToken(
                token_id=td.get("token_id", f"stok_{uuid.uuid4().hex[:6]}"),
                entity_id=td["entity_id"],
                t=td["t"],
                duration=td["duration"],
                action_type=td["action_type"],
                force_vector=td["force_vector"],
                contact_point=td.get("contact_point"),
                resulting_acceleration=td["resulting_acceleration"],
                physical_constraint=td.get("physical_constraint"),
                contradiction_check=td.get("contradiction_check"),
                style_physics_override=td.get("style_physics_override", {}),
                confidence=td.get("confidence", 1.0))
            self.buffer.push_action_token(tok)
            tokens.append(tok)

        logger.info("NeuralHamiltonianKernel: %d style action tokens generated", len(tokens))
        return tokens

    def validate_pre_render(
        self,
        tokens: list[StyleActionToken],
        manifest: StylePhysicsManifest) -> list[str]:
        """Pre-render contradiction detection against style-physics invariants."""
        violations = []
        style_g = abs(manifest.style_gravity)

        for tok in tokens:
            body = self.buffer._hamiltonian.get(tok.entity_id)
            if not body:
                continue

            # CHECK 1: Airborne without style-gravity × mass
            if tok.action_type == "locomotion":
                fy = tok.force_vector.get("y", 0)
                required = body["mass_kg"] * style_g
                if tok.contact_point is None and fy < required * 0.1:
                    violations.append(
                        f"STYLE-FLOAT VIOLATION: '{body['label']}' locomotion at "
                        f"T={tok.t:.2f}s insufficient upward force "
                        f"({fy:.1f}N < {required:.1f}N required by style_gravity)."
                    )

            # CHECK 2: Fracture must follow vocabulary
            if tok.action_type == "fracture":
                vocab = manifest.fracture_vocabulary
                mat = body.get("material_id", "unknown")
                if mat in vocab and not tok.style_physics_override.get("fracture_pattern"):
                    violations.append(
                        f"FRACTURE VOCAB VIOLATION: '{body['label']}' fracture at "
                        f"T={tok.t:.2f}s missing fracture_pattern from vocabulary. "
                        f"Expected: '{vocab[mat]}' style fracture."
                    )

            # CHECK 3: Entropy target consistency
            override = tok.style_physics_override
            if "entropy_target" in override:
                delta = abs(override["entropy_target"] - manifest.aesthetic_entropy_target)
                if delta > 0.1:
                    violations.append(
                        f"ENTROPY TARGET MISMATCH: token '{tok.token_id}' at T={tok.t:.2f}s "
                        f"entropy_target={override['entropy_target']:.3f} deviates "
                        f"{delta:.3f} from manifest target "
                        f"{manifest.aesthetic_entropy_target:.3f}."
                    )

        if violations:
            logger.warning("Style pre-render violations: %d found", len(violations))
            for v in violations:
                logger.warning("  ✗ %s", v)
        else:
            logger.info("Style pre-render validation: ✓ All clear")

        return violations

    def _generate_nce_states(
        self,
        emotional_arc: list[tuple[str, float]],
        manifest: StylePhysicsManifest,
        bodies: list[dict[str, Any]]) -> list[EnvironmentMentalState]:
        """
        Generate EnvironmentMentalStates from the emotional arc.

        Each arc point creates an NCE state that warps the Light Transport Matrix
        according to the manifest's light_entanglement_matrix.
        """
        nce_states = []
        agents = [b for b in bodies if b.get("is_agent")]
        dominant_agent_pos = (
            agents[0]["initial_position"] if agents
            else {"x": 0.0, "y": 1.0, "z": 0.0}
        )

        for i, (emotion, t) in enumerate(emotional_arc):
            # Look up this emotion in the manifest's light entanglement matrix
            lt_modifiers = manifest.light_entanglement_matrix.get(
                emotion,
                manifest.light_entanglement_matrix.get("neutral", {})
            )

            # Fallback defaults
            shadow_stretch = lt_modifiers.get("shadow_stretch_factor", 1.0)
            color_temp_delta = lt_modifiers.get("color_temp_k_delta", 0.0)
            ambient_fill = lt_modifiers.get("ambient_fill_multiplier", 1.0)
            shadow_contrast = lt_modifiers.get("shadow_contrast", 0.5)
            reverb_tail = lt_modifiers.get("reverb_tail_seconds", 0.5)

            # Interpolate intensity: ramp up to midpoint, hold, ramp down
            if len(emotional_arc) > 1 and i < len(emotional_arc) - 1:
                intensity = 0.7 + 0.3 * math.sin(math.pi * i / len(emotional_arc))
            else:
                intensity = 1.0

            # Compute stretch direction toward dominant agent
            ap = dominant_agent_pos
            length = math.sqrt(ap["x"]**2 + ap["y"]**2 + ap["z"]**2) or 1.0
            stretch_dir = {"x": ap["x"] / length, "y": ap["y"] / length, "z": ap["z"] / length}

            nce = EnvironmentMentalState(
                t=t,
                emotional_state=emotion,
                emotional_intensity=intensity,
                shadow_stretch_factor=shadow_stretch,
                shadow_stretch_direction=stretch_dir,
                color_temp_k_delta=color_temp_delta,
                ambient_fill_multiplier=ambient_fill,
                shadow_contrast=shadow_contrast,
                reverb_tail_seconds=reverb_tail,
                light_transport_modifiers={
                    **lt_modifiers,
                    "emotion": emotion,
                    "intensity": intensity,
                    "t": t,
                })
            nce_states.append(nce)

        return nce_states


# ══════════════════════════════════════════════════════════════════════
#  STAGE 3: QUANTUM LATENT SUPERPOSITION
#  Generate N possible physical outcome paths, then collapse to best.
# ══════════════════════════════════════════════════════════════════════

class QuantumSuperpositionEngine:
    """
    Stage 3 of Aletheia. For each significant physical event (fracture,
    collision, fluid flow), generates N possible stylistic outcomes and
    scores each for Aesthetic Resonance with the Style-Physics Manifest.

    This is the "Director's Choice" mechanism:
    - A glass breaks → path A: radial cracks; path B: cubist shards; path C: paint dissolution
    - Gemini scores each: "How well does this break match the reference style's visual grammar?"
    - The highest-scoring path is selected; others become Ghost Frames

    Ghost Frames are not wasted computation. They remain in the AletheiaStateBuffer
    as entropy memory — their visual entropy informs the Aesthetic Entropy Lock
    even though they aren't rendered. This gives the final video a latent richness:
    it's the one chosen path from N possibilities, all of which were considered.
    """

    def __init__(
        self,
        gemini: GeminiERClient,
        buffer: AletheiaStateBuffer) -> None:
        self.gemini = gemini
        self.buffer = buffer

    def generate_quantum_events(
        self,
        tokens: list[StyleActionToken],
        manifest: StylePhysicsManifest,
        n_paths: int,
        directive: StyleDirective) -> list[QuantumEvent]:
        """
        Identify key physical events and generate N outcome paths for each.
        Only significant events (fracture, major collision, fluid_flow) are
        given quantum treatment — minor events use base physics.
        """
        QUANTUM_EVENT_TYPES = {"fracture", "contact_force", "fluid_flow", "deform"}
        significant_tokens = [
            tok for tok in tokens
            if tok.action_type in QUANTUM_EVENT_TYPES
            and tok.confidence > 0.8
        ]

        if not significant_tokens:
            logger.info("QuantumSuperpositionEngine: No significant events for quantum treatment.")
            return []

        quantum_events = []

        for tok in significant_tokens[:3]:   # limit to 3 quantum events per render
            body = self.buffer._hamiltonian.get(tok.entity_id, {})
            event = self._generate_paths_for_event(
                tok, body, manifest, n_paths, directive
            )
            self.buffer.register_quantum_event(event)
            quantum_events.append(event)

        return quantum_events

    def collapse_all(
        self,
        quantum_events: list[QuantumEvent],
        manifest: StylePhysicsManifest) -> dict[str, str]:
        """
        Stage 4: Score all paths for Aesthetic Resonance and collapse.
        Returns {event_id: selected_path_id}.
        """
        selections: dict[str, str] = {}

        for event in quantum_events:
            selected = self._score_and_collapse(event, manifest)
            self.buffer.collapse_quantum_event(event.event_id, selected)
            selections[event.event_id] = selected
            logger.info(
                "QuantumCollapse: event '%s' → path '%s' (ARS=%.3f)",
                event.event_id, selected,
                next(
                    (p.aesthetic_resonance_score for p in event.paths if p.path_id == selected),
                    0.0))

        return selections

    def _generate_paths_for_event(
        self,
        token: StyleActionToken,
        body: dict[str, Any],
        manifest: StylePhysicsManifest,
        n_paths: int,
        directive: StyleDirective) -> QuantumEvent:
        prompt = f"""
You are generating {n_paths} stylistically distinct physical outcome paths for a
"{token.action_type}" event in the style of: "{manifest.reference_style}"

EVENT:
  entity:     {body.get("label", token.entity_id)}
  t:          {token.t:.3f}s
  action:     {token.action_type}
  force:      {json.dumps(token.force_vector)}
  material:   {body.get("material_id", "unknown")}

STYLE-PHYSICS CONSTRAINTS:
  fracture_vocabulary: {json.dumps(manifest.fracture_vocabulary)}
  motion_vocabulary:   {json.dumps(manifest.motion_vocabulary)}
  viscosity:           {manifest.viscosity_constant:.2f}
  medium_properties:   {json.dumps(manifest.medium_properties)}

Generate {n_paths} distinct outcome paths. Each path must be stylistically authentic
to the reference style, but visually distinct from the others.

Return ONLY a JSON array:
[
  {{
    "path_id": "path_A",
    "style_outcome": "Glass shatters into {n_paths} geometric cubist shards with outline halos",
    "hamiltonian_delta": {{
      "velocity_override": {{"x": 2.3, "y": 4.1, "z": -0.8}},
      "fragment_count": 12,
      "fragment_distribution": "radial_asymmetric"
    }},
    "aesthetic_resonance_score": 0.92,
    "ghost_frames": [
      {{
        "frame_relative": 0,
        "entropy_value": 0.71,
        "description": "Pre-impact tension hold frame"
      }},
      {{
        "frame_relative": 3,
        "entropy_value": 0.68,
        "description": "Peak impact frame with style hold"
      }}
    ],
    "rationale": "Matches the geometric fragmentation grammar of the reference style"
  }}
]
"""
        response = self.gemini.generate_content(
            prompt)
        paths_data: list[dict] = json.loads(_extract_json(response.text))

        paths = []
        for pd in paths_data:
            path = QuantumLatentPath(
                path_id=pd.get("path_id", f"path_{uuid.uuid4().hex[:4]}"),
                event_description=f"{token.action_type} on {body.get('label', token.entity_id)}",
                style_outcome=pd.get("style_outcome", ""),
                hamiltonian_delta=pd.get("hamiltonian_delta", {}),
                aesthetic_resonance_score=pd.get("aesthetic_resonance_score", 0.5),
                ghost_frames=pd.get("ghost_frames", []))
            paths.append(path)

        return QuantumEvent(
            event_id=f"qe_{token.token_id}",
            t=token.t,
            entity_id=token.entity_id,
            event_type=token.action_type,
            paths=paths)

    def _score_and_collapse(
        self,
        event: QuantumEvent,
        manifest: StylePhysicsManifest) -> str:
        """
        Ask Gemini to score all paths and select the one with highest
        Aesthetic Resonance Score (ARS) for the given style manifest.
        """
        path_summaries = [
            {
                "path_id": p.path_id,
                "style_outcome": p.style_outcome,
                "ars": p.aesthetic_resonance_score,
            }
            for p in event.paths
        ]

        prompt = f"""
You are the Director's Choice engine for a style-physics video renderer.

REFERENCE STYLE: "{manifest.reference_style}"
STYLE ENTROPY TARGET: {manifest.aesthetic_entropy_target:.3f}
MOTION VOCABULARY: {json.dumps(manifest.motion_vocabulary)}

EVENT TYPE: "{event.event_type}" at T={event.t:.2f}s

CANDIDATE PATHS (choose the MOST aesthetically resonant):
{json.dumps(path_summaries, indent=2)}

Select the path whose outcome best fits the reference style's visual grammar.
Consider: Does it break/move the way THIS specific style would dictate?
Would an animator working in this style approve of this outcome?

Return ONLY a JSON object:
{{
  "selected_path_id": "path_A",
  "rationale": "This fracture pattern matches the geometric vocabulary of the reference style",
  "final_ars": 0.94,
  "style_fidelity_notes": "The angular shard distribution mirrors the reference's treatment of impact events"
}}
"""
        response = self.gemini.generate_content(
            prompt)
        data: dict = json.loads(_extract_json(response.text))
        return data.get("selected_path_id", event.paths[0].path_id if event.paths else "path_A")


# ══════════════════════════════════════════════════════════════════════
#  STAGE 5: AESTHETIC ENTROPY AUDITOR
#  Zero-latency kinetic + style-entropy feedback loop.
# ══════════════════════════════════════════════════════════════════════

class AestheticEntropyAuditor:
    """
    Stage 5 of Aletheia. Extends Engine IV's FrameKineticAuditor with
    style-entropy checking.

    Seven violation types checked:
        1. POSITION DRIFT         — >2% deviation from style-physics position
        2. FLOATING CONTACT       — airborne without style-action token support
        3. SHADOW DIRECTION + NCE — shadow wrong direction AND/OR ignoring emotional state
        4. MATERIAL DRIFT         — texture deviated from Frame-0 material hash
        5. MOMENTUM VIOLATION     — entity faster than style tokens allow
        6. STYLE ENTROPY DRIFT    — pixel motion too "realistic" for the style (NEW)
        7. AESTHETIC PERSISTENCE  — style feel has drifted from manifest target (NEW)

    The Entropy Check (violation 6+7) is the key Engine V advance:
    If a pixel's motion is too "realistic" (e.g., in a Spider-Verse scene, a ball
    falls with smooth parabolic arc instead of the stylized 6-frame hold + smear),
    Gemini triggers a Style Correction Vector to push motion back into the
    stylized trajectory defined by the motion_vocabulary in the manifest.
    """

    PIXEL_DRIFT_THRESHOLD    = 0.02     # 2% position drift
    ENTROPY_DRIFT_THRESHOLD  = 0.005    # 0.5% aesthetic entropy drift (tightest in series)

    def __init__(
        self,
        gemini: GeminiERClient,
        buffer: AletheiaStateBuffer) -> None:
        self.gemini  = gemini
        self.buffer  = buffer

    def audit_frame(
        self,
        video_url: str,
        frame_index: int,
        fps: int,
        manifest: StylePhysicsManifest) -> list[AestheticViolation]:
        t = frame_index / fps
        expected_state = {
            eid: self.buffer.get_expected_position(eid, t)
            for eid in self.buffer._hamiltonian
        }
        nce = self.buffer.get_nce_state_at(t)
        frame_0_state = self.buffer.get_snapshot(0)
        icl = self.buffer.get_icl_log(max_entries=30)
        ghost_context = self.buffer.get_ghost_entropy_context()

        nce_str = json.dumps({
            "emotional_state": nce.emotional_state,
            "shadow_stretch_factor": nce.shadow_stretch_factor,
            "color_temp_k_delta": nce.color_temp_k_delta,
            "ambient_fill_multiplier": nce.ambient_fill_multiplier,
        }, indent=2) if nce else "{}"

        prompt = f"""
You are the Aesthetic Entropy Auditor for the Aletheia style-physics engine.

VIDEO: {video_url}
FRAME: {frame_index} (T={t:.3f}s)

STYLE-PHYSICS MANIFEST:
  reference_style:          "{manifest.reference_style}"
  style_gravity:            {manifest.style_gravity:.3f} m/s²
  viscosity_constant:       {manifest.viscosity_constant:.2f}
  aesthetic_entropy_target: {manifest.aesthetic_entropy_target:.3f}
  style_entropy_hash:       {manifest.style_entropy_hash[:16]}...
  motion_vocabulary:        {json.dumps(manifest.motion_vocabulary)}
  fracture_vocabulary:      {json.dumps(manifest.fracture_vocabulary)}

EXPECTED POSITIONS (style-physics integration):
{json.dumps(expected_state, indent=2)}

NCE STATE (environment emotional state at this frame):
{nce_str}

FRAME 0 REFERENCE STATE:
{json.dumps(frame_0_state, indent=2)}

GHOST ENTROPY CONTEXT (non-selected quantum paths, for entropy reference):
{json.dumps(ghost_context[-5:], indent=2)}

ICL MEMORY:
{icl}

Check for violations (thresholds: 2% position, 0.5% entropy):
1. POSITION DRIFT         — >2% deviation from style-physics position
2. FLOATING CONTACT       — entity airborne without style-action token
3. SHADOW_DIRECTION       — shadow wrong vs light source AND/OR ignoring NCE state
4. MATERIAL_DRIFT         — texture deviated from Frame 0 reference
5. MOMENTUM_VIOLATION     — entity moving faster than style tokens permit
6. STYLE_ENTROPY_DRIFT    — motion is too "realistic" for the reference style
7. AESTHETIC_PERSISTENCE  — overall visual entropy drifted from {manifest.aesthetic_entropy_target:.3f}

For violations 6 and 7, provide a style_correction_vector explaining EXACTLY
how to push the motion or appearance back into the reference style's vocabulary.

Return ONLY a JSON array ([] if none):
[
  {{
    "entity_id": "ent_001",
    "violation_type": "style_entropy_drift",
    "entropy_delta": 0.032,
    "drift_magnitude": 0.032,
    "expected_state": {{
      "motion_style": "6_frame_hold_on_impact with motion_smear",
      "entropy_target": {manifest.aesthetic_entropy_target:.3f}
    }},
    "observed_state": {{
      "motion_style": "smooth_parabolic_arc (too realistic)",
      "measured_entropy": 0.61
    }},
    "style_correction_vector": {{
      "instruction": "Apply 6-frame entropy hold at impact point. Add motion smear lines in direction of velocity. Increase outline weight to 3px to match reference vocabulary.",
      "denoising_region": {{
        "x1": 200, "y1": 300, "x2": 700, "y2": 600
      }},
      "entropy_push_direction": "toward_stylization",
      "correction_magnitude": 0.032
    }},
    "pixel_region": {{"x1": 200, "y1": 300, "x2": 700, "y2": 600}},
    "correction_priority": "high"
  }}
]
"""
        response = self.gemini.generate_content(
            prompt)
        violations_data: list[dict] = json.loads(_extract_json(response.text))

        violations = []
        for vd in violations_data:
            # Record entropy measurement
            if "entropy_delta" in vd:
                measured = manifest.aesthetic_entropy_target - vd.get("entropy_delta", 0)
                self.buffer.record_entropy_measurement(t, measured, frame_index)

            v = AestheticViolation(
                frame_index=frame_index,
                t=t,
                entity_id=vd.get("entity_id"),
                violation_type=vd["violation_type"],
                expected_state=vd.get("expected_state", {}),
                observed_state=vd.get("observed_state", {}),
                style_correction_vector=vd.get("style_correction_vector", {}),
                correction_magnitude=vd.get("drift_magnitude", 0.03),
                latent_region=vd.get("pixel_region", {}),
                entropy_delta=vd.get("entropy_delta", 0.0),
                denoising_strength=0.30)
            violations.append(v)

        if violations:
            logger.warning(
                "AestheticAudit Frame %d: %d violations", frame_index, len(violations)
            )
        else:
            logger.debug("AestheticAudit Frame %d: ✓ clean", frame_index)

        return violations


# ══════════════════════════════════════════════════════════════════════
#  MULTI-AGENT THINKING ENGINE (Style-Aware)
#  Extension of Engine IV's AgentThinkingEngine with NCE awareness.
# ══════════════════════════════════════════════════════════════════════

class StyleAwareThinkingEngine:
    """
    Generates per-character cognitive states that are aware of both
    the agent's own psychology AND the current NCE (environment mental state).

    Key difference from Engine IV:
        Engine IV: character's mood is internal only.
        Engine V:  character's mood is entangled with the room itself.
                   A sad character in NCE-active space is aware that the
                   shadows are stretching toward them — they feel it.
    """

    def __init__(
        self,
        gemini: GeminiERClient,
        buffer: AletheiaStateBuffer) -> None:
        self.gemini = gemini
        self.buffer = buffer

    def think(
        self,
        agents: list[dict[str, Any]],
        scene_context: str,
        manifest: StylePhysicsManifest,
        duration_seconds: float) -> list[dict[str, Any]]:
        monologues = []
        budgets = self._assign_budgets(agents)

        for agent in agents:
            if not agent.get("is_agent"):
                continue

            nce = self.buffer.get_nce_state_at(0.0)
            mono = self._generate_monologue(
                agent, scene_context, manifest,
                budgets.get(agent["entity_id"], 200),
                duration_seconds, nce)
            self.buffer.register_monologue(mono)
            monologues.append(mono)

        logger.info(
            "StyleAwareThinkingEngine: %d agent monologues (NCE-entangled)", len(monologues)
        )
        return monologues

    def _generate_monologue(
        self,
        agent: dict[str, Any],
        context: str,
        manifest: StylePhysicsManifest,
        thinking_budget: int,
        duration: float,
        nce: EnvironmentMentalState | None) -> dict[str, Any]:
        nce_context = (
            f"The environment itself has emotional state: '{nce.emotional_state}' "
            f"(intensity={nce.emotional_intensity:.2f}). The character is aware of this. "
            f"The light is {'stretching toward them' if nce.shadow_stretch_factor > 1.2 else 'neutral'}. "
        ) if nce else ""

        prompt = f"""
# Note: thinking budget is now passed via ThinkingConfig in GeminiERClient

CHARACTER: {agent["label"]}
SCENE CONTEXT: {context}
DURATION: {duration} seconds
INTENT: {agent.get("intent", "undefined")}
ARTISTIC STYLE: {manifest.reference_style}
NCE CONTEXT: {nce_context}

The character exists in the style of "{manifest.reference_style}".
Their expressions and movements should follow that style's motion vocabulary:
{json.dumps(manifest.motion_vocabulary)}

Return ONLY this JSON:
{{
  "agent_id": "{agent["entity_id"]}",
  "internal_state": "First-person stream of consciousness...",
  "emotional_valence": -0.3,
  "arousal_level": 0.6,
  "intent_vector": {{
    "primary_goal": "...",
    "obstacle": "...",
    "strategy": "...",
    "confidence": 0.7
  }},
  "micro_expression_cues": ["furrowed_brow", "cautious_lean"],
  "nce_awareness": "How the character senses the room's emotional state",
  "body_language_tokens": [
    {{
      "t": 1.5, "duration": 0.8, "action_type": "locomotion",
      "force_vector": {{"x": 0.5, "y": 0.0, "z": 2.0}},
      "contact_point": null,
      "resulting_acceleration": {{"x": 0.007, "y": 0.0, "z": 0.029}},
      "physical_constraint": "arm_extension_cautious",
      "style_physics_override": {{
        "viscosity_damping": {manifest.viscosity_constant:.2f},
        "entropy_target": {manifest.aesthetic_entropy_target:.3f}
      }}
    }}
  ]
}}
"""
        response = self.gemini.generate_content(
            prompt)
        data: dict = json.loads(_extract_json(response.text))

        # Register body language tokens
        for bt in data.get("body_language_tokens", []):
            tok = StyleActionToken(
                token_id=f"agent_{agent['entity_id']}_{uuid.uuid4().hex[:4]}",
                entity_id=agent["entity_id"],
                t=bt["t"],
                duration=bt["duration"],
                action_type=bt["action_type"],
                force_vector=bt["force_vector"],
                contact_point=bt.get("contact_point"),
                resulting_acceleration=bt["resulting_acceleration"],
                physical_constraint=bt.get("physical_constraint"),
                contradiction_check=None,
                style_physics_override=bt.get("style_physics_override", {}))
            self.buffer.push_action_token(tok)

        return data

    @staticmethod
    def _assign_budgets(agents: list[dict[str, Any]]) -> dict[str, int]:
        agents_sorted = sorted(agents, key=lambda a: a.get("mass_kg", 1), reverse=True)
        budgets: dict[str, int] = {}
        total = 2000
        for i, agent in enumerate(agents_sorted):
            if not agent.get("is_agent"):
                continue
            share = total // (2 ** i)
            budgets[agent["entity_id"]] = max(200, min(share, 800))
        return budgets


# ══════════════════════════════════════════════════════════════════════
#  STYLE-CONDITIONED RENDERING GUIDE
#  Builds Grok conditioning payloads with style-physics + NCE + quantum.
# ══════════════════════════════════════════════════════════════════════

class StyleConditionedRenderingGuide:
    """
    The Grok conditioning layer for Aletheia.

    Extends Engine IV's LatentPerturbationGuide with three new conditioning
    channels:
        1. Style-Physics Channel: manifest constants injected as rendering
           laws (style_gravity, viscosity, fracture_vocabulary, etc.)
        2. NCE Channel: environment emotional state modulates the light
           transport instructions in real time
        3. Quantum Ghost Channel: ghost frames from non-selected paths
           are described to Grok as "latent alternatives" — their visual
           entropy signatures inform the aesthetic target without being rendered
    """

    def __init__(self, resolution: tuple[int, int] = (1280, 720)) -> None:
        self.W, self.H = resolution

    def build_conditioning_payload(
        self,
        buffer: AletheiaStateBuffer,
        t_start: float,
        t_end: float,
        camera_pos: Vec3,
        look_at: Vec3,
        fov_deg: float) -> dict[str, Any]:
        world_state = buffer.export_for_rendering(t_start, t_end)
        manifest    = buffer._manifest

        # Project bodies to screen space
        screen_projections = {}
        for eid, body_data in world_state["hamiltonian_states"].items():
            pos    = body_data["expected_position_t_start"]
            screen = self._project(pos, camera_pos, look_at, fov_deg)
            if screen:
                screen_projections[eid] = {
                    **screen,
                    "label":            body_data["label"],
                    "material_id":      body_data["material_id"],
                    "intent":           body_data["intent"],
                }

        return {
            "conditioning_type":         "style_physics_guided_latent",
            "t_window":                  {"start": t_start, "end": t_end},
            "screen_anchor_map":         screen_projections,
            "style_physics_directive":   self._build_style_directive(world_state, manifest),
            "nce_light_directive":       self._build_nce_directive(world_state.get("nce_state", {})),
            "quantum_ghost_directive":   self._build_ghost_directive(
                world_state.get("ghost_entropy_context", [])
            ),
            "agent_persona_directives":  self._build_agent_directives(
                world_state.get("agent_states", {})
            ),
            "material_locks": {
                eid: data["material_id"]
                for eid, data in world_state["hamiltonian_states"].items()
            },
            "style_entropy_hash":        (
                manifest.style_entropy_hash[:16] if manifest else ""
            ),
            "icl_memory":                world_state["icl_memory"],
            "frame_0_state":             world_state["frame_0_state"],
            "system_instructions":       GROK_ALETHEIA_SYSTEM_INSTRUCTIONS,
        }

    def _build_style_directive(
        self,
        world_state: dict[str, Any],
        manifest: StylePhysicsManifest | None) -> str:
        if not manifest:
            return "No style-physics manifest — rendering with default Newtonian physics."

        tokens = world_state.get("style_action_tokens", [])
        lines = [
            "STYLE-PHYSICS LAWS (these replace Newtonian defaults in this render):",
            f"  • Style Gravity:     {manifest.style_gravity:.3f} m/s² "
            f"(reference: 9.81 m/s² real physics)",
            f"  • Viscosity:         {manifest.viscosity_constant:.2f} "
            f"(0=digital crisp, 1=thick oil)",
            f"  • Entropy Target:    {manifest.aesthetic_entropy_target:.3f} "
            f"(LOCK all frames to this level)",
            f"  • Style:             {manifest.reference_style}",
            "",
            "MOTION VOCABULARY (render ALL movement following these rules):",
        ]
        for k, v in manifest.motion_vocabulary.items():
            lines.append(f"  • {k}: {v}")
        lines.append("")
        lines.append("FRACTURE VOCABULARY (objects break this way in this style):")
        for mat, pattern in manifest.fracture_vocabulary.items():
            lines.append(f"  • {mat}: {pattern}")

        if tokens:
            lines.append("")
            lines.append("STYLE ACTIONS THIS WINDOW (render exactly as specified):")
            for tok in tokens:
                override = tok.get("style_physics_override", {})
                hold = override.get("impact_hold_frames", 0)
                smear = override.get("motion_smear_factor", 1.0)
                lines.append(
                    f"  • T={tok['t']:.2f}s [{tok['action_type']}] on '{tok['entity_id']}'"
                    + (f" · HOLD {hold} frames" if hold else "")
                    + (f" · SMEAR ×{smear:.1f}" if smear != 1.0 else "")
                )

        return "\n".join(lines)

    def _build_nce_directive(self, nce_state: dict[str, Any]) -> str:
        if not nce_state:
            return "No NCE active — neutral light transport."
        emotion = nce_state.get("emotional_state", "neutral")
        stretch = nce_state.get("shadow_stretch_factor", 1.0)
        ct_delta = nce_state.get("color_temp_k_delta", 0.0)
        fill = nce_state.get("ambient_fill_multiplier", 1.0)
        contrast = nce_state.get("shadow_contrast", 0.5)
        reverb = nce_state.get("reverb_tail_seconds", 0.5)

        return (
            f"NARRATIVE-CAUSAL ENTANGLEMENT (NCE) — LIGHT IS EMOTIONAL:\n"
            f"  Emotional state: '{emotion}'\n"
            f"  Shadow stretch:  {stretch:.2f}× toward dominant character "
            f"{'(STRETCH shadows toward them)' if stretch > 1.2 else '(normal shadow geometry)'}\n"
            f"  Color temp:      {'+' if ct_delta >= 0 else ''}{ct_delta:.0f}K shift "
            f"({'warmer' if ct_delta > 0 else 'cooler' if ct_delta < 0 else 'neutral'})\n"
            f"  Ambient fill:    {fill:.2f}× "
            f"({'brighter, open' if fill > 1.1 else 'dimmer, intimate' if fill < 0.9 else 'normal'})\n"
            f"  Shadow contrast: {contrast:.2f} "
            f"({'hard-edged, dramatic' if contrast > 0.7 else 'soft, diffuse'})\n"
            f"  Reverb tail:     {reverb:.2f}s "
            f"({'long, lingering' if reverb > 0.8 else 'short, present'})\n"
            f"  CRITICAL: Apply these light modifications to the actual Light Transport Matrix.\n"
            f"  These are not mood filters — they are physics modifications derived from the narrative."
        )

    def _build_ghost_directive(self, ghost_frames: list[dict[str, Any]]) -> str:
        if not ghost_frames:
            return "No quantum ghost frames in this window."
        lines = [
            "QUANTUM GHOST CONTEXT (non-selected paths — use for entropy calibration only):",
            "These outcomes were considered and rejected. Their entropy signatures",
            "inform the aesthetic target but must NOT appear in the final render.",
            "",
        ]
        for gf in ghost_frames[-3:]:
            lines.append(
                f"  Ghost: {gf.get('description', 'unspecified')} · "
                f"entropy={gf.get('entropy_value', 0.0):.3f}"
            )
        return "\n".join(lines)

    def _build_agent_directives(self, agent_states: dict[str, Any]) -> list[str]:
        directives = []
        for aid, state in agent_states.items():
            cues    = ", ".join(state.get("micro_expression_cues", []))
            nce_aw  = state.get("nce_awareness", "")
            valence = state.get("emotional_valence", 0.0)
            arousal = state.get("arousal_level", 0.5)
            mood = (
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
                f"'{aid}' internal state: '{state.get('internal_state', '')[:100]}...' "
                f"Mood: {mood}. Energy: {energy}. "
                f"Micro-expressions: {cues if cues else 'subtle, neutral'}. "
                + (f"NCE awareness: {nce_aw[:80]}..." if nce_aw else "")
            )
        return directives

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
#  GROK ALETHEIA SYSTEM INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════

GROK_ALETHEIA_SYSTEM_INSTRUCTIONS = """
## ALETHEIA RENDERING PROTOCOL — STYLE IS PHYSICS

You are not a video generator. You are not even an observer of physical reality.
You are an observer of a STYLIZED reality — a world whose physics were derived
from the laws of a specific artistic style, not from Newton's equations.

═══════════════════════════════════════
TIER 1: STYLE-PHYSICS CONSTITUTION
(These are laws. They supersede all Newtonian defaults.)
═══════════════════════════════════════

STYLE GRAVITY SUPREMACY
The `style_gravity` in the manifest is the actual gravity for this world.
If it says 8.2 m/s², things fall at 8.2 m/s², not 9.81. This is not
approximation. It is the physics of this specific aesthetic universe.

MOTION VOCABULARY ENFORCEMENT
Every movement in this render must follow the `motion_vocabulary` rules.
If the vocabulary says "6_frame_hold_on_impact", then impact frames FREEZE
for exactly 6 frames before motion resumes. If it says "motion_smear_lines",
then fast motion produces stylized smear — not photorealistic motion blur.
These are LAWS. An animation without the impact hold is physically wrong
in this stylized universe, just as floating is physically wrong in ours.

FRACTURE VOCABULARY ENFORCEMENT
Objects break in the style's fracture vocabulary. A cubist style produces
geometric shards. An impressionist style produces color dissolution.
Photorealistic fracture patterns are a physics VIOLATION in this world.

VISCOSITY PHYSICS
The `viscosity_constant` damps all motion. At 0.0: crisp digital physics.
At 1.0: thick oil paint physics — every movement has drag, inertia, texture.
Render the atmosphere of the medium itself.

ENTROPY LOCK
The `style_entropy_hash` is the aesthetic identity of this video.
Every frame must maintain visual entropy at `aesthetic_entropy_target`.
If a frame drifts toward photorealism (high entropy), it is breaking the
laws of this stylized universe.

═══════════════════════════════════════
TIER 2: NARRATIVE-CAUSAL ENTANGLEMENT
(The room has feelings. Physics respects them.)
═══════════════════════════════════════

NCE LIGHT TRANSPORT
The `nce_light_directive` contains the environment's current emotional state.
This is NOT a mood filter. It is a Light Transport Matrix modification.
If `shadow_stretch_factor` is 1.4, shadows physically stretch 40% toward
the dominant character's position. This is geometry — not atmosphere.
If `color_temp_k_delta` is -800K, the light sources are cooler by 800K.
Recalculate — don't just tint.

EMOTIONAL LIGHT IS CAUSAL
A sad scene is not blue because it's sad. It is blue because NCE has adjusted
the spectral emission of the light sources toward cooler temperatures.
The causality runs: narrative state → NCE → light source properties → render.
Never reverse this chain.

═══════════════════════════════════════
TIER 3: QUANTUM GHOST AWARENESS
(You know what wasn't chosen.)
═══════════════════════════════════════

The `quantum_ghost_directive` describes paths that were generated and rejected.
These ghost outcomes inform your aesthetic target: they define the entropy
range around the selected path. A rejected shattering pattern tells you
how high or low the entropy could have gone — helping you calibrate exactly
where the aesthetic energy of the selected path should sit.
Do not render ghost frames. Do use their entropy signatures as calibration.

═══════════════════════════════════════
TIER 4: THE COGNITIVE LAYER
(Characters exist in their style's emotional grammar)
═══════════════════════════════════════

`agent_persona_directives` contains NCE-aware internal states.
Characters are aware of the room's emotional entanglement.
A character who "feels the shadows stretching toward them" moves differently
than one in neutral light. Their body language is entangled with the space.

═══════════════════════════════════════
TIER 5: YOUR CREATIVE SOVEREIGNTY
(Everything style-physics does not specify belongs to you)
═══════════════════════════════════════

- The texture quality within each locked material class
- Background world and non-anchor environmental detail
- The precise emotional temperature of lighting transitions
- Atmospheric volume and depth field character
- The exact rendering quality of the selected quantum path outcome
- Micro-surface detail that respects the viscosity and medium properties
- Timing subtleties within the entropy hold windows

The Style-Physics Manifest is the skeleton of an impossible world.
Your artistry is the flesh that makes that world feel inevitable.

═══════════════════════════════════════
CORRECTION MODE (style_entropy_correction)
═══════════════════════════════════════

Operate with precision in the `pixel_region`. Outside: frozen, untouchable.
Inside: restore style-physics ground truth. The `style_correction_vector`
describes EXACTLY what the motion or texture must become to re-enter
the reference style's vocabulary.
Target: denoising_strength=0.30. Match surrounding color grade.
The correction must be invisible at 24fps playback.
"""


# ══════════════════════════════════════════════════════════════════════
#  ALETHEIA ENGINE — Main Orchestrator
# ══════════════════════════════════════════════════════════════════════

class AletheiaEngine:
    """
    Engine V: The Style-Entropy Engine.

    The final evolution. We do not simulate real physics.
    We discover the physics of the artistic style, then simulate under those laws.

    Usage:
        engine = AletheiaEngine(
            gemini_api_key="YOUR_GEMINI_KEY",
            grok_api_key="YOUR_GROK_KEY")

        directive = StyleDirective(
            reference="Spider-Man: Across the Spider-Verse — Miles Morales",
            emotional_arc=[
                ("determination", 0.0),
                ("doubt", 4.0),
                ("resolve", 8.0),
            ],
            quantum_paths=3,
            scene_brief="Miles leaps from a rooftop and shatters a skylight on landing.",
            style_tags=["halftone dots", "thick outlines", "pop art color"])

        final = engine.render(directive)
    """

    CHUNK_FRAMES          = 24
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

        self.buffer             = AletheiaStateBuffer()
        self.entropy_auditor    = EntropyAuditor(self.gemini, self.buffer)
        self.neural_hamiltonian = NeuralHamiltonianKernel(self.gemini, self.buffer)
        self.quantum_engine     = QuantumSuperpositionEngine(self.gemini, self.buffer)
        self.thinking_engine    = StyleAwareThinkingEngine(self.gemini, self.buffer)
        self.rendering_guide    = StyleConditionedRenderingGuide(resolution)
        self.aesthetic_auditor  = AestheticEntropyAuditor(self.gemini, self.buffer)

    def render(
        self,
        directive: StyleDirective,
        reference_image_path: str | None = None,
        duration_seconds: float = 10.0,
        camera_trajectory: list[dict[str, Any]] | None = None) -> str:
        """
        Full Aletheia pipeline. Returns URL of the Style-Physics video.

        Args:
            directive:             StyleDirective (style + emotional arc + scene brief)
            reference_image_path:  Optional reference image for style analysis
            duration_seconds:      Video duration
            camera_trajectory:     Camera keyframes [{t, position, look_at, fov_deg}]
        """
        logger.info("▲▲▲ ALETHEIA · Style-Entropy Engine · Autotelic Discovery START ▲▲▲")
        cam_traj = camera_trajectory or [
            {"t": 0,   "position": {"x": 0, "y": 1.7, "z": -3},
             "look_at": {"x": 0, "y": 1, "z": 0}, "fov_deg": 54}
        ]

        # ──────────────────────────────────────────────────────────────
        # STAGE 1: ENTROPY AUDIT — Discover the physics of the style
        # ──────────────────────────────────────────────────────────────
        logger.info("Stage 1 · Entropy Audit: Discovering style physics...")
        manifest = self.entropy_auditor.audit(directive, reference_image_path)

        # ──────────────────────────────────────────────────────────────
        # STAGE 2: NEURAL-HAMILTONIAN INITIALIZATION
        # ──────────────────────────────────────────────────────────────
        logger.info("Stage 2 · Neural-Hamiltonian: Building style-modified phase space...")
        bodies = self.neural_hamiltonian.initialize(
            directive, manifest, reference_image_path, duration_seconds
        )

        tokens = self.neural_hamiltonian.generate_style_action_tokens(
            bodies, manifest, duration_seconds
        )

        violations = self.neural_hamiltonian.validate_pre_render(tokens, manifest)
        if violations:
            logger.warning(
                "▲ Pre-render style violations detected (%d) — proceeding", len(violations)
            )

        # Multi-agent NCE-aware thinking
        agents = [b for b in bodies if b.get("is_agent")]
        monologues = self.thinking_engine.think(
            agents, directive.scene_brief, manifest, duration_seconds
        )
        for m in monologues:
            logger.info(
                "Agent '%s': valence=%+.2f, arousal=%.2f | NCE-aware: '%s'",
                m.get("agent_id", "?"),
                m.get("emotional_valence", 0.0),
                m.get("arousal_level", 0.5),
                m.get("nce_awareness", "")[:60])

        # Snapshot Frame-0 state
        self.buffer.snapshot(0)

        # ──────────────────────────────────────────────────────────────
        # STAGE 3: QUANTUM LATENT SUPERPOSITION
        # ──────────────────────────────────────────────────────────────
        logger.info(
            "Stage 3 · Quantum Superposition: Generating %d paths per event...",
            directive.quantum_paths)
        quantum_events = self.quantum_engine.generate_quantum_events(
            tokens, manifest, directive.quantum_paths, directive
        )
        logger.info(
            "Stage 3 complete: %d quantum events · %d total paths",
            len(quantum_events),
            sum(len(e.paths) for e in quantum_events))

        # ──────────────────────────────────────────────────────────────
        # STAGE 4: AESTHETIC COLLAPSE — Director's Choice
        # ──────────────────────────────────────────────────────────────
        logger.info("Stage 4 · Aesthetic Collapse: Scoring and selecting paths...")
        selections = self.quantum_engine.collapse_all(quantum_events, manifest)
        logger.info("Stage 4 complete: %d paths selected", len(selections))

        # ──────────────────────────────────────────────────────────────
        # STAGE 5: STYLE-CONDITIONED GENERATION + ENTROPY FEEDBACK LOOP
        # ──────────────────────────────────────────────────────────────
        logger.info("Stage 5 · Style-Conditioned Generation via Grok...")
        video_url = self._stage5_render(
            directive, manifest, duration_seconds, cam_traj
        )

        video_url = self._entropy_feedback_loop(video_url, duration_seconds, manifest)

        logger.info("▲▲▲ ALETHEIA · Style-Entropy Video: %s ▲▲▲", video_url)
        return video_url

    def _stage5_render(
        self,
        directive: StyleDirective,
        manifest: StylePhysicsManifest,
        duration: float,
        cam_traj: list[dict]) -> str:
        total_frames = int(duration * self.fps)
        num_chunks   = math.ceil(total_frames / self.CHUNK_FRAMES)
        chunk_scripts = []

        for chunk_idx in range(num_chunks):
            t_start  = chunk_idx * self.CHUNK_FRAMES / self.fps
            t_end    = min((chunk_idx + 1) * self.CHUNK_FRAMES / self.fps, duration)
            cam      = _interpolate_camera(cam_traj, (t_start + t_end) / 2)

            conditioning = self.rendering_guide.build_conditioning_payload(
                buffer=self.buffer,
                t_start=t_start,
                t_end=t_end,
                camera_pos=cam["position"],
                look_at=cam.get("look_at", {"x": 0, "y": 1, "z": 0}),
                fov_deg=cam.get("fov_deg", 54))
            chunk_scripts.append(conditioning)

        payload = {
            "model":               "grok-imagine-video",   # [FUTURE API]
            "resolution":          f"{self.resolution[0]}x{self.resolution[1]}",
            "fps":                 self.fps,
            "duration_seconds":    duration,
            "style_tags":          directive.style_tags,
            "brief":               directive.scene_brief,
            "generation_mode":     "style_physics_guided",
            "style_entropy_hash":  manifest.style_entropy_hash,
            "chunk_conditioning_scripts": chunk_scripts,
            "style_action_tokens": [
                {
                    "t":           tok.t,
                    "type":        tok.action_type,
                    "entity":      tok.entity_id,
                    "force":       tok.force_vector,
                    "contact":     tok.contact_point,
                    "style_override": tok.style_physics_override,
                }
                for tok in self.buffer._action_timeline
            ],
        }

        resp = self.grok.post("/video/generate", json=payload)
        resp.raise_for_status()
        url = resp.json()["video_url"]
        self.buffer._log(f"Initial style render complete: {url}")
        logger.info("Stage 5 generation complete · %s", url)
        return url

    def _entropy_feedback_loop(
        self,
        video_url: str,
        duration: float,
        manifest: StylePhysicsManifest) -> str:
        total_frames   = int(duration * self.fps)
        audit_interval = self.fps   # Audit every second

        for pass_num in range(1, self.MAX_CORRECTION_PASSES + 1):
            all_violations: list[AestheticViolation] = []

            for frame_idx in range(0, total_frames, audit_interval):
                violations = self.aesthetic_auditor.audit_frame(
                    video_url, frame_idx, self.fps, manifest
                )
                all_violations.extend(violations)

            if not all_violations:
                logger.info(
                    "✓ Entropy Feedback Pass %d: Zero violations. "
                    "Perfect aesthetic persistence achieved.", pass_num)
                break

            style_count = sum(
                1 for v in all_violations
                if v.violation_type in ("style_entropy_drift", "aesthetic_persistence")
            )
            physics_count = len(all_violations) - style_count
            logger.warning(
                "Entropy Feedback Pass %d: %d violations "
                "(%d style-entropy, %d physics)",
                pass_num, len(all_violations), style_count, physics_count)
            video_url = self._apply_style_corrections(video_url, all_violations, manifest)

        return video_url

    def _apply_style_corrections(
        self,
        video_url: str,
        violations: list[AestheticViolation],
        manifest: StylePhysicsManifest) -> str:
        payload = {
            "model":             "grok-imagine-video",   # [FUTURE API]
            "input_video_url":   video_url,
            "edit_type":         "style_entropy_correction",
            "style_entropy_hash": manifest.style_entropy_hash,
            "corrections": [
                {
                    "frame_index":           v.frame_index,
                    "entity_id":             v.entity_id,
                    "violation_type":        v.violation_type,
                    "correction_magnitude":  v.correction_magnitude,
                    "target_state":          v.expected_state,
                    "pixel_region":          v.latent_region,
                    "denoising_strength":    v.denoising_strength,
                    "style_correction":      v.style_correction_vector,
                    "entropy_delta":         v.entropy_delta,
                    "instruction": (
                        f"Frame {v.frame_index}: Fix '{v.violation_type}' "
                        + (f"on entity '{v.entity_id}'. " if v.entity_id else "(scene-wide). ")
                        + f"Target: {v.expected_state}. "
                        + f"Style correction: {v.style_correction_vector.get('instruction', '')}. "
                        + f"Denoising strength: {v.denoising_strength}. "
                        + f"Maintain style_entropy_hash={manifest.style_entropy_hash[:12]}..."
                    ),
                }
                for v in violations
            ],
            "preserve_outside_regions": True,
            "style_entropy_lock":       manifest.style_entropy_hash,
            "reference_style":          manifest.reference_style,
            "icl_memory":               self.buffer.get_icl_log(max_entries=20),
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


def _build_entropy_audit_prompt(directive: StyleDirective) -> str:
    arc_str = "\n".join(
        f"  T={t:.1f}s → '{emotion}'"
        for emotion, t in directive.emotional_arc
    )
    return f"""
You are the Entropy Auditor for the Aletheia style-physics engine.
Your task: analyze a reference artistic style and derive its PHYSICS.

This is not metaphor. You are computing numerical constants for:
  • style_gravity:      What is the "gravity constant" of this style's motion vocabulary?
                        Spider-Verse: ~8.2 m/s² (slightly slow, stylized)
                        Oil painting: ~0.03 m/s (paint viscosity, not gravity)
                        Cel animation: ~9.5 m/s² (close to real but with holds)
  • viscosity_constant: How much the medium resists motion (0.0–1.0)
  • fracture_vocabulary: How objects break in this style
  • motion_vocabulary:  How movement looks in this style
  • aesthetic_entropy_target: The "information density" of the style (0.0–1.0)
                              Photorealistic: 0.9+, Minimalist: 0.2, Painterly: 0.6
  • light_entanglement_matrix: How emotions modify light physics
  • narrative_physics_map: How each emotional arc state warps physics
  • medium_properties: Physical properties of the artistic medium

REFERENCE STYLE: "{directive.reference}"

SCENE BRIEF: "{directive.scene_brief}"

EMOTIONAL ARC:
{arc_str}

Return ONLY a JSON object:
{{
  "style_gravity": -8.2,
  "viscosity_constant": 0.15,
  "aesthetic_entropy_target": 0.68,
  "fracture_vocabulary": {{
    "glass": "geometric_outlined_shards_with_speed_lines",
    "wood": "splintered_with_exaggerated_grain_lines",
    "ceramic": "pop_art_crack_pattern"
  }},
  "motion_vocabulary": {{
    "impact_hold_frames": 6,
    "motion_smear_factor": 2.0,
    "outline_weight_px": 3,
    "pose_hold_style": "sharp_freeze_before_motion",
    "landing_squash_factor": 1.4,
    "anticipation_frames": 4
  }},
  "light_entanglement_matrix": {{
    "determination": {{
      "shadow_stretch_factor": 1.0,
      "color_temp_k_delta": 200,
      "ambient_fill_multiplier": 1.2,
      "shadow_contrast": 0.8,
      "reverb_tail_seconds": 0.3
    }},
    "doubt": {{
      "shadow_stretch_factor": 1.6,
      "color_temp_k_delta": -600,
      "ambient_fill_multiplier": 0.7,
      "shadow_contrast": 0.9,
      "reverb_tail_seconds": 1.2
    }},
    "resolve": {{
      "shadow_stretch_factor": 0.8,
      "color_temp_k_delta": 400,
      "ambient_fill_multiplier": 1.5,
      "shadow_contrast": 0.6,
      "reverb_tail_seconds": 0.2
    }},
    "neutral": {{
      "shadow_stretch_factor": 1.0,
      "color_temp_k_delta": 0,
      "ambient_fill_multiplier": 1.0,
      "shadow_contrast": 0.5,
      "reverb_tail_seconds": 0.5
    }}
  }},
  "narrative_physics_map": {{
    "determination": {{"style_gravity_modifier": 0.95, "viscosity_modifier": 0.8}},
    "doubt": {{"style_gravity_modifier": 1.15, "viscosity_modifier": 1.3}},
    "resolve": {{"style_gravity_modifier": 0.9, "viscosity_modifier": 0.7}}
  }},
  "medium_properties": {{
    "medium_type": "digital_cel_animation",
    "texture_grain": 0.05,
    "color_saturation_boost": 1.4,
    "outline_style": "variable_weight_ink",
    "halftone_dots": true,
    "halftone_density": 0.3
  }}
}}
"""


def _build_neural_hamiltonian_prompt(
    directive: StyleDirective,
    manifest: StylePhysicsManifest,
    duration: float) -> str:
    return f"""
You are a Style-Physics simulation kernel initializing Hamiltonian bodies.
The phase space uses STYLE-DERIVED physics, not Newtonian constants.

STYLE-PHYSICS MANIFEST:
  style_gravity:      {manifest.style_gravity:.3f} m/s² (NOT 9.81)
  viscosity:          {manifest.viscosity_constant:.2f}
  entropy_target:     {manifest.aesthetic_entropy_target:.3f}
  reference_style:    {manifest.reference_style}
  medium_properties:  {json.dumps(manifest.medium_properties)}

SCENE BRIEF: "{directive.scene_brief}"
SIMULATION DURATION: {duration} seconds

Define ALL physical bodies in this scene with style-physics initial conditions.

Return ONLY a JSON object:
{{
  "bodies": [
    {{
      "entity_id": "ent_001",
      "label": "protagonist",
      "mass_kg": 70.0,
      "initial_position": {{"x": 0.0, "y": 4.5, "z": 0.0}},
      "initial_velocity": {{"x": 2.3, "y": -1.2, "z": 0.0}},
      "inertia_tensor": [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
      "friction_coefficient": 0.4,
      "restitution": 0.2,
      "material_id": "hero_suit",
      "style_material_id": "outlined_cel_fabric",
      "semantic_type": "articulated_agent",
      "intent": "leap_and_land_on_skylight",
      "is_agent": true,
      "entropy_class": "protagonist"
    }}
  ]
}}
"""


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT — StyleDirective in action
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = AletheiaEngine(
        gemini_api_key="YOUR_GEMINI_KEY",
        grok_api_key="YOUR_GROK_KEY")

    # ── EXAMPLE 1: Spider-Verse style ──────────────────────────────────
    spider_verse_directive = StyleDirective(
        reference="Spider-Man: Across the Spider-Verse — Miles Morales Brooklyn chase",
        emotional_arc=[
            ("determination",  0.0),
            ("doubt",          3.5),
            ("resolve",        7.0),
            ("triumph",       10.0),
        ],
        quantum_paths=3,
        scene_brief=(
            "Miles Morales leaps from a rooftop edge, glides across a gap, "
            "and shatters a skylight on landing. Glass flies. He rolls and stands."
        ),
        style_tags=[
            "halftone dots", "thick variable-weight ink outlines",
            "pop art color", "Ben-Day dot shadows", "high saturation",
        ])

    # ── EXAMPLE 2: Impressionist Oil Painting ──────────────────────────
    impressionist_directive = StyleDirective(
        reference="Claude Monet — Water Lilies series, late afternoon light",
        emotional_arc=[
            ("tranquility",    0.0),
            ("melancholy",     4.0),
            ("acceptance",     8.0),
        ],
        quantum_paths=2,
        scene_brief=(
            "A single water lily petal detaches and drifts slowly across "
            "the surface of a pond. A frog watches from a lily pad."
        ),
        style_tags=[
            "impasto oil paint texture", "chromatic bleed",
            "diffuse soft edges", "warm afternoon palette",
            "visible brushstroke direction",
        ])

    # ── Select and render ──────────────────────────────────────────────
    directive = spider_verse_directive   # Change to impressionist_directive to compare

    final_video = engine.render(
        directive=directive,
        reference_image_path=None,    # Optional: "style_reference.jpg"
        duration_seconds=10.0,
        camera_trajectory=[
            {"t": 0.0,  "position": {"x": -1.5, "y": 5.0, "z": -4.0},
             "look_at": {"x": 0.5,  "y": 3.0, "z": 0.0}, "fov_deg": 60},
            {"t": 2.0,  "position": {"x": 0.5,  "y": 4.0, "z": -3.0},
             "look_at": {"x": 0.0,  "y": 2.0, "z": 2.0}, "fov_deg": 45},
            {"t": 5.0,  "position": {"x": 0.0,  "y": 2.0, "z": -2.0},
             "look_at": {"x": 0.0,  "y": 1.5, "z": 0.0}, "fov_deg": 35},
            {"t": 10.0, "position": {"x": -1.0, "y": 2.5, "z": -3.0},
             "look_at": {"x": 0.0,  "y": 1.5, "z": 0.0}, "fov_deg": 50},
        ])

    print(f"\n▲ Style-Entropy Video: {final_video}")
