"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         GEMINI ER CLIENT  —  Shared SDK Adapter for All Engines             ║
║         google-genai SDK v1.x  •  gemini-robotics-er-1.5-preview            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  WHAT THIS SOLVES:                                                           ║
║    All engines previously used the deprecated google-generativeai package.   ║
║    The official Gemini Robotics-ER 1.5 docs exclusively use the new          ║
║    google-genai package with genai.Client(), types.Part.from_bytes(),        ║
║    and types.GenerateContentConfig(thinking_config=...).                     ║
║                                                                              ║
║    This adapter provides a single migration point so every engine gets:      ║
║      1. Correct SDK (google-genai, not google-generativeai)                  ║
║      2. Task-specific ThinkingConfig budgets (0 for fast, 4096 for complex)  ║
║      3. Native box_2d format matching ER 1.5 fine-tuning                     ║
║      4. Consensus querying (multi-pass averaging for high-precision tasks)   ║
║      5. Trajectory generation (waypoint sequences for path planning)         ║
║      6. Code execution tool for resolving ambiguous small objects            ║
║                                                                              ║
║  USAGE:                                                                      ║
║    from gemini_er_client import GeminiERClient, ThinkingPreset               ║
║    client = GeminiERClient(api_key="...")                                    ║
║    result = client.point(image_bytes, "point to the red cup")               ║
║    result = client.detect(image_bytes, "cup, bottle, hand")                  ║
║    result = client.reason(image_bytes, prompt, thinking=ThinkingPreset.DEEP) ║
║    result = client.trajectory(image_bytes, "move pen to organizer")          ║
║    result = client.consensus(image_bytes, prompt, passes=3)                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from statistics import mean
from typing import Any

from google import genai                   # pip install google-genai
from google.genai import types

logger = logging.getLogger("gemini_er_client")

# ── Official model string ─────────────────────────────────────────────────────
ER15_MODEL = "gemini-robotics-er-1.5-preview"
FALLBACK_MODEL = "gemini-2.0-flash"        # used only if ER 1.5 unavailable


# ── Thinking budgets ──────────────────────────────────────────────────────────

class ThinkingPreset(IntEnum):
    """
    Task-appropriate thinking budgets per official Gemini ER 1.5 documentation.

    NONE    (0)     — Object detection / pointing.  ER 1.5 is fine-tuned for
                      fast pointing and the docs recommend thinking_budget=0
                      for these tasks.
    LIGHT   (512)   — Quick spatial alignment checks.
    MEDIUM  (2048)  — Oracle audit / collision reasoning / causal inference.
    DEEP    (4096)  — Multi-hop graph reasoning / complex scene understanding.
    MAX     (8192)  — Full causal chain + physics + character cognition.
    """
    NONE   = 0
    LIGHT  = 512
    MEDIUM = 2048
    DEEP   = 4096
    MAX    = 8192


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class SpatialPoint:
    """A 2D point returned by the ER 1.5 Pointing API, normalized 0–1000."""
    y: float
    x: float
    label: str = ""
    confidence: float = 1.0

    @property
    def normalized(self) -> tuple[float, float]:
        """Returns (y_norm, x_norm) in 0.0–1.0 range."""
        return (self.y / 1000.0, self.x / 1000.0)


@dataclass
class BoundingBox2D:
    """
    Bounding box in official ER 1.5 box_2d format.
    Field name matches model fine-tuning: box_2d not bbox.
    Coordinates normalized 0–1000, [ymin, xmin, ymax, xmax].
    """
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    label: str = ""
    confidence: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.ymin + self.ymax) / 2.0, (self.xmin + self.xmax) / 2.0)

    @property
    def height(self) -> float:
        return abs(self.ymax - self.ymin)

    @property
    def width(self) -> float:
        return abs(self.xmax - self.xmin)

    @property
    def area(self) -> float:
        return self.height * self.width

    def to_dict(self) -> dict:
        return {
            "box_2d": [self.ymin, self.xmin, self.ymax, self.xmax],
            "label": self.label,
        }


@dataclass
class TrajectoryPoint:
    """A single waypoint in an ER 1.5 trajectory response."""
    y: float
    x: float
    order: int = 0
    label: str = ""


@dataclass
class GeminiERResponse:
    """Unified response object from any GeminiERClient call."""
    raw_text: str
    points: list[SpatialPoint] = field(default_factory=list)
    boxes: list[BoundingBox2D] = field(default_factory=list)
    trajectory: list[TrajectoryPoint] = field(default_factory=list)
    structured: dict = field(default_factory=dict)
    thinking_tokens_used: int = 0
    model_used: str = ER15_MODEL
    passes: int = 1                        # >1 if consensus mode


# ── Main client ───────────────────────────────────────────────────────────────

class GeminiERClient:
    """
    Drop-in replacement for the old google-generativeai genai.GenerativeModel
    pattern used across Engines I–IX.

    Uses the official google-genai SDK (genai.Client + types.Part.from_bytes)
    and the gemini-robotics-er-1.5-preview model string.

    Key capabilities exposed:
      .point()       — 2D spatial pointing [y, x]
      .detect()      — object detection with box_2d bounding boxes
      .reason()      — free-form reasoning with configurable thinking budget
      .trajectory()  — multi-waypoint path generation
      .consensus()   — multi-pass averaged results for high-precision tasks
      .with_code()   — call with code_execution tool for small/occluded objects
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = ER15_MODEL,
        default_thinking: ThinkingPreset = ThinkingPreset.NONE,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            logger.warning(
                "No Gemini API key provided. Set GEMINI_API_KEY env var or pass api_key=."
            )
        self._client = genai.Client(api_key=resolved_key)
        self.model = model
        self.default_thinking = default_thinking
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_image_part(self, image: bytes | str | Path) -> types.Part:
        """Convert various image sources to types.Part for the new SDK."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            raw = path.read_bytes()
            # Detect MIME type by magic bytes
            mime = "image/jpeg" if raw[:2] == b"\xff\xd8" else "image/png"
            return types.Part.from_bytes(data=raw, mime_type=mime)
        # Already bytes — probe MIME
        mime = "image/jpeg" if image[:2] == b"\xff\xd8" else "image/png"
        return types.Part.from_bytes(data=image, mime_type=mime)

    def _make_config(
        self,
        thinking: ThinkingPreset | int | None = None,
        use_code_execution: bool = False,
    ) -> types.GenerateContentConfig:
        budget = int(thinking) if thinking is not None else int(self.default_thinking)
        cfg = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=budget),
        )
        if use_code_execution:
            cfg.tools = [types.Tool(code_execution=types.ToolCodeExecution)]
        return cfg

    def _call(
        self,
        contents: list,
        thinking: ThinkingPreset | int | None = None,
        use_code_execution: bool = False,
    ) -> str:
        """Raw call to client.models.generate_content. Returns response text."""
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._make_config(thinking, use_code_execution),
        )
        # Collect text from all content parts (handles code_execution responses)
        parts_text = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                parts_text.append(part.text)
            if hasattr(part, "code_execution_result") and part.code_execution_result:
                parts_text.append(part.code_execution_result.output or "")
        return "\n".join(parts_text)

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Extract JSON from a model response that may include prose."""
        # Try code blocks first
        m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if m:
            return json.loads(m.group(1).strip())
        # Try raw JSON array or object
        m = re.search(r"(\[[\s\S]+\]|\{[\s\S]+\})", text)
        if m:
            return json.loads(m.group(1))
        return {}

    @staticmethod
    def _parse_points(data: Any) -> list[SpatialPoint]:
        """Parse [{"point": [y, x], "label": "..."}, ...] responses."""
        if not isinstance(data, list):
            return []
        pts: list[SpatialPoint] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            pt = item.get("point", [])
            if len(pt) >= 2:
                pts.append(SpatialPoint(y=float(pt[0]), x=float(pt[1]),
                                        label=item.get("label", "")))
        return pts

    @staticmethod
    def _parse_boxes(data: Any) -> list[BoundingBox2D]:
        """
        Parse [{"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}, ...].
        Also accepts legacy "bbox" key for backwards compatibility.
        """
        if not isinstance(data, list):
            return []
        boxes: list[BoundingBox2D] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            # Official key first, then legacy fallback
            bb = item.get("box_2d") or item.get("bbox", [])
            if len(bb) >= 4:
                boxes.append(BoundingBox2D(
                    ymin=float(bb[0]), xmin=float(bb[1]),
                    ymax=float(bb[2]), xmax=float(bb[3]),
                    label=item.get("label", ""),
                ))
        return boxes

    @staticmethod
    def _parse_trajectory(data: Any) -> list[TrajectoryPoint]:
        """Parse trajectory waypoints from ER 1.5 response."""
        if not isinstance(data, list):
            return []
        waypoints: list[TrajectoryPoint] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            pt = item.get("point", [])
            if len(pt) >= 2:
                waypoints.append(TrajectoryPoint(
                    y=float(pt[0]), x=float(pt[1]),
                    order=int(item.get("order", len(waypoints))),
                    label=item.get("label", str(len(waypoints))),
                ))
        waypoints.sort(key=lambda w: w.order)
        return waypoints

    # ── Public API ────────────────────────────────────────────────────────────

    def point(
        self,
        image: bytes | str | Path,
        prompt: str,
        thinking: ThinkingPreset | int = ThinkingPreset.NONE,
    ) -> GeminiERResponse:
        """
        2D Spatial Pointing — returns [y, x] coordinates for objects.

        Per official docs, thinking_budget=0 is recommended for pointing tasks
        as the model is specifically fine-tuned for fast spatial pointing.

        Coordinates are normalized 0–1000.
        """
        full_prompt = (
            f"{prompt}\n\n"
            "Return ONLY valid JSON in this exact format:\n"
            '[{"point": [<y>, <x>], "label": "<label>"}, ...]\n'
            "Coordinates must be integers normalized to 0–1000. No prose."
        )
        contents = [self._make_image_part(image), full_prompt]
        raw = self._call(contents, thinking=thinking)
        data = self._extract_json(raw)
        return GeminiERResponse(
            raw_text=raw,
            points=self._parse_points(data),
            model_used=self.model,
        )

    def detect(
        self,
        image: bytes | str | Path,
        prompt: str,
        thinking: ThinkingPreset | int = ThinkingPreset.NONE,
    ) -> GeminiERResponse:
        """
        Object Detection — returns box_2d bounding boxes.

        Uses the official box_2d key format that matches ER 1.5 fine-tuning.
        Format: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}]
        """
        full_prompt = (
            f"{prompt}\n\n"
            "Return ONLY valid JSON. Use this exact format:\n"
            '[{"box_2d": [<ymin>, <xmin>, <ymax>, <xmax>], "label": "<label>"}, ...]\n'
            "box_2d values must be integers normalized to 0–1000, [ymin, xmin, ymax, xmax].\n"
            "No prose, no markdown."
        )
        contents = [self._make_image_part(image), full_prompt]
        raw = self._call(contents, thinking=thinking)
        data = self._extract_json(raw)
        return GeminiERResponse(
            raw_text=raw,
            boxes=self._parse_boxes(data),
            model_used=self.model,
        )

    def reason(
        self,
        image: bytes | str | Path | None,
        prompt: str,
        thinking: ThinkingPreset | int = ThinkingPreset.MEDIUM,
        schema: dict | None = None,
    ) -> GeminiERResponse:
        """
        Free-form reasoning with configurable thinking budget.

        Use ThinkingPreset.MEDIUM (2048) for oracle audits and collision checks.
        Use ThinkingPreset.DEEP (4096) for causal chain inference.
        Use ThinkingPreset.MAX (8192) for full multi-hop graph reasoning.

        If schema is provided, appends a JSON output constraint to the prompt.
        """
        if schema:
            prompt = (
                f"{prompt}\n\n"
                f"Return ONLY valid JSON matching this schema:\n{json.dumps(schema, indent=2)}\n"
                "No prose, no markdown fences."
            )
        contents = []
        if image is not None:
            contents.append(self._make_image_part(image))
        contents.append(prompt)
        raw = self._call(contents, thinking=thinking)
        structured = {}
        try:
            structured = self._extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            pass
        return GeminiERResponse(
            raw_text=raw,
            structured=structured,
            model_used=self.model,
        )

    def trajectory(
        self,
        image: bytes | str | Path,
        prompt: str,
        num_waypoints: int = 15,
        thinking: ThinkingPreset | int = ThinkingPreset.LIGHT,
    ) -> GeminiERResponse:
        """
        Trajectory Generation — returns a sequence of [y, x] waypoints.

        Per official docs, ER 1.5 is fine-tuned to return ordered waypoints
        describing a path between objects or to a destination. Waypoints are
        labeled by order from '0' to str(num_waypoints - 1).

        Useful for camera path planning and robot movement.
        """
        full_prompt = (
            f"{prompt}\n\n"
            f"Return ONLY valid JSON with exactly {num_waypoints} trajectory waypoints:\n"
            '[{"point": [<y>, <x>], "label": "<order_number>", "order": <int>}, ...]\n'
            "Coordinates are integers normalized to 0–1000. "
            "Label each point by its order in the trajectory, starting at '0'.\n"
            "No prose, no markdown."
        )
        contents = [self._make_image_part(image), full_prompt]
        raw = self._call(contents, thinking=thinking)
        data = self._extract_json(raw)
        return GeminiERResponse(
            raw_text=raw,
            trajectory=self._parse_trajectory(data),
            points=self._parse_points(data),   # also expose as points for compat
            model_used=self.model,
        )

    def consensus(
        self,
        image: bytes | str | Path,
        prompt: str,
        passes: int = 3,
        mode: str = "point",
        thinking: ThinkingPreset | int = ThinkingPreset.NONE,
    ) -> GeminiERResponse:
        """
        Consensus Querying — queries the same frame multiple times and averages.

        Per official docs and the Gemini instruction reference, this is the
        recommended approach for high-precision tasks where a single-pass result
        may be noisy.  The averaged coordinates reduce false positives in the
        correction loop.

        mode: "point" averages SpatialPoint [y, x] across passes.
              "detect" averages BoundingBox2D coordinates across passes.
        """
        all_results: list[GeminiERResponse] = []
        for _ in range(passes):
            if mode == "detect":
                r = self.detect(image, prompt, thinking=thinking)
            else:
                r = self.point(image, prompt, thinking=thinking)
            all_results.append(r)

        if mode == "detect":
            avg_boxes = _average_boxes([r.boxes for r in all_results])
            return GeminiERResponse(
                raw_text=all_results[-1].raw_text,
                boxes=avg_boxes,
                model_used=self.model,
                passes=passes,
            )
        else:
            avg_pts = _average_points([r.points for r in all_results])
            return GeminiERResponse(
                raw_text=all_results[-1].raw_text,
                points=avg_pts,
                model_used=self.model,
                passes=passes,
            )

    def with_code(
        self,
        image: bytes | str | Path,
        prompt: str,
        thinking: ThinkingPreset | int = ThinkingPreset.LIGHT,
    ) -> GeminiERResponse:
        """
        Code Execution Tool — ER 1.5 writes and runs Python on the image itself.

        Per official docs, ER 1.5 supports tools=[types.Tool(code_execution=...)]
        which allows the model to crop, zoom, or transform the image before
        returning coordinates. Improves accuracy on small or occluded objects.
        """
        contents = [self._make_image_part(image), prompt]
        raw = self._call(contents, thinking=thinking, use_code_execution=True)
        structured = {}
        try:
            structured = self._extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            pass
        boxes = self._parse_boxes(structured) if isinstance(structured, list) else []
        points = self._parse_points(structured) if isinstance(structured, list) else []
        return GeminiERResponse(
            raw_text=raw,
            boxes=boxes,
            points=points,
            structured=structured,
            model_used=self.model,
        )

    def generate_content(
        self,
        contents: list,
        thinking: ThinkingPreset | int | None = None,
    ) -> str:
        """
        Low-level passthrough — accepts arbitrary contents list.
        For engines that build their own prompt structures.
        Returns raw text.
        """
        return self._call(contents, thinking=thinking)

    def make_image_part(self, image: bytes | str | Path) -> types.Part:
        """Expose _make_image_part publicly for engines that assemble contents themselves."""
        return self._make_image_part(image)

    def make_text_part(self, text: str) -> types.Part:
        return types.Part.from_text(text=text)


# ── Consensus averaging helpers ───────────────────────────────────────────────

def _average_points(
    all_passes: list[list[SpatialPoint]],
) -> list[SpatialPoint]:
    """
    Average point coordinates across multiple passes.
    Matches by label if present, otherwise by index.
    """
    if not all_passes:
        return []
    reference = all_passes[0]
    averaged: list[SpatialPoint] = []
    for i, ref_pt in enumerate(reference):
        ys, xs = [ref_pt.y], [ref_pt.x]
        for other in all_passes[1:]:
            # Try to find by label, fall back to index
            match = next(
                (p for p in other if p.label == ref_pt.label), None
            ) if ref_pt.label else (other[i] if i < len(other) else None)
            if match:
                ys.append(match.y)
                xs.append(match.x)
        averaged.append(SpatialPoint(
            y=mean(ys), x=mean(xs),
            label=ref_pt.label,
            confidence=len(ys) / len(all_passes),
        ))
    return averaged


def _average_boxes(
    all_passes: list[list[BoundingBox2D]],
) -> list[BoundingBox2D]:
    """
    Average bounding box coordinates across multiple passes.
    Matches by label, falls back to index.
    """
    if not all_passes:
        return []
    reference = all_passes[0]
    averaged: list[BoundingBox2D] = []
    for i, ref_box in enumerate(reference):
        ymins, xmins, ymaxs, xmaxs = (
            [ref_box.ymin], [ref_box.xmin], [ref_box.ymax], [ref_box.xmax]
        )
        for other in all_passes[1:]:
            match = next(
                (b for b in other if b.label == ref_box.label), None
            ) if ref_box.label else (other[i] if i < len(other) else None)
            if match:
                ymins.append(match.ymin)
                xmins.append(match.xmin)
                ymaxs.append(match.ymax)
                xmaxs.append(match.xmax)
        averaged.append(BoundingBox2D(
            ymin=mean(ymins), xmin=mean(xmins),
            ymax=mean(ymaxs), xmax=mean(xmaxs),
            label=ref_box.label,
            confidence=len(ymins) / len(all_passes),
        ))
    return averaged


# ── Coordinate transform utilities ────────────────────────────────────────────
#
# These were already correct in all engines. Keeping them here as a canonical
# reference. Gemini ER 1.5 outputs [y, x] normalized 0–1000. To map into
# Blender's meter-based [X, Y, Z] right-hand Y-up coordinate system:

def er15_to_blender(
    y_norm: float,
    x_norm: float,
    scene_width_m: float = 10.0,
    scene_height_m: float = 7.0,
    depth_m: float = 0.0,
) -> tuple[float, float, float]:
    """
    Project ER 1.5 [y, x] normalized 0–1000 into Blender [X, Y, Z] meters.

    Args:
        y_norm:        ER 1.5 y coordinate, 0–1000 (top = 0)
        x_norm:        ER 1.5 x coordinate, 0–1000 (left = 0)
        scene_width_m: Width of the scene in Blender meters (X axis)
        scene_height_m: Height of the scene in Blender meters (Y axis)
        depth_m:       Z depth — estimated separately (focal length / bbox height)

    Returns:
        (blender_X, blender_Y, blender_Z) in meters
    """
    blender_x = (x_norm / 1000.0) * scene_width_m - (scene_width_m / 2.0)
    blender_y = -(y_norm / 1000.0) * scene_height_m + (scene_height_m / 2.0)
    return (blender_x, blender_y, depth_m)
