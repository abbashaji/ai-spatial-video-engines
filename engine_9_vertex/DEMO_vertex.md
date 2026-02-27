# Vertex Engine — Real Video Demo

Runs the full seven-stage Vertex pipeline on a real input video.

## Usage

```bash
python vertex_engine.py \
  --video input_video.mp4 \
  --scene "A kitchen with objects on a countertop" \
  --api-key YOUR_GEMINI_KEY \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

| Flag | Description |
|------|-------------|
| `--video` | Path to your input `.mp4` (or any cv2-readable format) |
| `--scene` | Natural-language description of the scene for Gemini spatial prompting |
| `--api-key` | Your `GEMINI_API_KEY` |
| `--blender` | Full path to Blender 4.3+ executable |

## Pipeline stages

| Stage | Component | What it does |
|-------|-----------|--------------|
| 1 | `VideoFrameExtractor` | cv2 keyframe + motion extraction |
| 2 | `SpatialKernel` | Gemini ER 1.5 as hardware sensor → Structural JSON |
| 3 | `SpatialGraph` | NetworkX spatial-relation graph (distance, angle) |
| 4 | `VoxelMap` | Pinhole Camera 2D→3D numpy occupancy grid |
| 5 | `ProxyCubeBuilder` | SpatialGraph → Blender BPY proxy cubes |
| 6 | `PixelAuditor` | cv2 pixel diff, 5 px threshold, recursive Delta-Correction |
| 7 | `NeuralRefinement` | ControlNet + Grok skin pass |
