"""
Engine VI — Prometheus: Neural-Symbolic 3D Orchestration

The paradigm shift: from fighting hallucination to making it mathematically impossible.
Blender's 3D mesh IS the ground truth. Bullet physics IS the simulator.
Gemini is no longer the Physical Supervisor — it is the Creative Director.

Exports:
    PrometheusEngine       — Main orchestrator
    SceneDirective         — Primary input primitive
    AssetSpec              — 3D asset specification
    StoryBeat              — Narrative event at a timestamp
    BpyScript              — Generated Blender Python script
    KeyframeSpec           — Blender f-curve keyframe
    CameraOperation        — DP-style camera directive
    RenderPass             — Blender Eevee-Next render output
    NeuralSkinResult       — ControlNet + Diffusion output
    RenderAuditResult      — Autonomous audit finding
    PrometheusState        — .blend-file-backed world state
"""

from .prometheus_engine import (
    PrometheusEngine,
    SceneDirective,
    AssetSpec,
    StoryBeat,
    BpyScript,
    KeyframeSpec,
    CameraOperation,
    RenderPass,
    NeuralSkinResult,
    RenderAuditResult,
    PrometheusState,
    BpyTranslator,
    KeyframeOrchestrator,
    PhysicsHandoffManager,
    CinematographyModule,
    BlenderRenderer,
    NeuralSkinningPipeline,
    AutonomousAuditLoop,
    PROMETHEUS_DIFFUSION_INSTRUCTIONS,
)

__all__ = [
    "PrometheusEngine",
    "SceneDirective",
    "AssetSpec",
    "StoryBeat",
    "BpyScript",
    "KeyframeSpec",
    "CameraOperation",
    "RenderPass",
    "NeuralSkinResult",
    "RenderAuditResult",
    "PrometheusState",
    "BpyTranslator",
    "KeyframeOrchestrator",
    "PhysicsHandoffManager",
    "CinematographyModule",
    "BlenderRenderer",
    "NeuralSkinningPipeline",
    "AutonomousAuditLoop",
    "PROMETHEUS_DIFFUSION_INSTRUCTIONS",
]
