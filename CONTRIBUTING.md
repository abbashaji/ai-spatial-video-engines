# Contributing

Thank you for your interest in the AI Spatial Video Engine Series.

---

## How to add a new engine

Each engine is a self-contained folder following this structure:

```
engine_N_name/
├── __init__.py          # Clean exports (see any existing engine)
└── name_engine.py       # Main engine implementation
```

### Step-by-step

1. **Create the folder** `engine_N_name/` where `N` is the next number.

2. **Write `name_engine.py`**. Your engine must expose at minimum:
   - A `Controller` class (e.g. `MyEngineController`)
   - A `Config` dataclass
   - A `Directive` dataclass (user input)
   - Optionally a `run_demo()` function

3. **Write `__init__.py`** — export all public classes:
   ```python
   from .name_engine import MyEngineController, MyEngineConfig, MyEngineDirective
   __all__ = ["MyEngineController", "MyEngineConfig", "MyEngineDirective"]
   ```

4. **Register in root `__init__.py`**:
   - Add a `try/except ImportError` block (follow the existing pattern)
   - Add the class names to `__all__`
   - Add a `get_version_info()` entry
   - Add a `select_engine()` branch if the engine covers a new use case

5. **Update `requirements.txt`**:
   - Add any new dependencies with a comment explaining their role
   - Add a minimum-install line in the summary section

6. **Update `CHANGELOG.md`** — add a new `## vN.0.0` section at the top.

7. **Add a demo file** `engine_N_name/DEMO_name.md` with a runnable example.

---

## Code style

- Follow the docstring/banner style of existing engines (the `╔══╗` header is optional but appreciated).
- Type-annotate all public methods.
- Keep the `GeminiERClient` in `gemini_er_client.py` as the single SDK adapter — do not inline SDK calls in engine files.
- All Blender interactions should go through headless CLI mode: `blender --background --python script.py`.

---

## Running tests

```bash
pip install pytest pytest-asyncio
pytest
```

Tests live alongside the engines. Name test files `test_name_engine.py`.

---

## Questions?

Open an issue on GitHub describing your use case.
