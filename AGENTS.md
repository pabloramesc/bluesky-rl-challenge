# Project conventions

- Write clean, concise, readable Python. Prefer small functions and explicit names.
- Keep abstractions minimal; extract shared logic when it removes duplication.
- Put reusable code in `src/bluesky_pabloramesc/`; keep `experiments/` entry points thin.
- Keep model settings in `models.py`, environment setup in `single_agent.py` and
  `multi_agent.py`, training/checkpoints in `training.py`, and argument parsing in `cli.py`.
- Treat `vendor/bluesky-gym/` as an upstream submodule; keep custom implementations in `src/`.
- Use one BlueSky environment per process; the simulator has global state.
- Save normalization statistics alongside policies; freeze them during evaluation.
- Add focused pytest tests for behavior changes. Prefer small synthetic environments.
- Keep generated models, results, caches, and virtual environments out of Git.
- Update documentation when commands or architecture change.

Checks after code changes:

```bash
python -m pytest
ruff check src experiments tests
ruff format --check src experiments tests
```
