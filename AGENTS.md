# AGENTS.md

Guidance for coding agents working in this repository. Read `README.md` first: it explains
what the tool measures, the method, and the current results.

## What this is

`cc-compact-tuning` is a Python CLI that reads Claude Code transcripts
(`~/.claude/projects`) and the status line's rate-limit log, fits how the subscription's usage
limits weigh each kind of token, and replays sessions to find the cheapest auto-compact trigger.
It is unofficial: the transcript format and limit accounting are undocumented and may change.

## Layout

| Path | Role |
|---|---|
| `src/cc_compact_tuning/cli.py` | argparse entry point (`cc-compact-tuning`): `collect`, `windows`, `fit`, `replay`, `fit-log` |
| `settings.py` | `CC_TUNING_*` paths via pydantic-settings; read once in `cli.main` and passed down |
| `store.py` | gzip JSON store (`store.json.gz`) with fixed `SECTIONS`; atomic save |
| `collect.py` | parses transcripts and the rate-limit log, merges into the store (never drops old records) |
| `pricing.py` | API list prices, `MODELS`, conversion of any model's tokens to Opus-equivalent |
| `windows.py` | rebuilds 5-hour windows that ended in a limit hit; flags unclean / Fable-heavy ones |
| `fit.py` | least-squares fit of the weights on clean windows, with bootstrap saved to the data dir |
| `fit_log.py` | alternative fit from rate-limit log percentage steps; `--synthetic` self-check |
| `replay.py` | compaction cost measurement and trigger replay; holds `FITTED_W` and `FITTED_MODEL_W` |
| `tests/` | pytest; `conftest.py` has builders (`request`, `assistant`, `write_jsonl`, `T0`) |
| `systemd/` | user timer that runs `mise run collect` daily |

## Commands

```sh
mise install && mise run setup   # pinned tools, then uv sync --locked
mise run check                   # ruff lint + format check, pytest, actionlint, gitleaks
mise run format                  # ruff format
uv run --locked pytest tests/test_fit.py -q   # a single test file
uv run --locked cc-compact-tuning --help
```

Run `mise run check` before committing; CI runs the same checks.

## Conventions

- Python ≥ 3.13 (pinned 3.15 via mise), ruff with line length 110 and rules `E F W I UP B`.
- Dependencies only via `uv add` / `uv remove`; tool versions in `mise.toml` then `mise lock`.
  Never edit `uv.lock` or `mise.lock` by hand.
- Modules take paths and the store as arguments; only `cli.py` builds `Settings`. Keep it so
  tests can use `tmp_path` and `monkeypatch.setenv`.
- Each module starts with a docstring explaining its method. Update it when the method changes.
- Token categories are always ordered `CATS = ["inp", "cw5", "cw1", "cr", "out"]` (input,
  5m cache write, 1h cache write, cache read, output). Weights are per Opus token, relative
  to uncached input = 1.
- Model ids and prices live only in `pricing.py`; `MODELS` maps short names (`opus`, `fable`,
  `sonnet`, `haiku`) to ids. Unknown ids fall back by prefix, then to Opus.
- `collect` must stay merge-only and idempotent: Claude Code deletes old transcripts, and the
  store is the only remaining copy. Don't drop or rename store sections without a migration in
  `store.load`.
- Tests use synthetic transcripts built with the `conftest.py` helpers. Never read the real
  `~/.claude` or data directory from tests.

## Data and privacy

- The store and bootstraps live in `~/.local/share/cc-compact-tuning`, outside the repo. They
  hold the user's own usage; never copy them into the repository, fixtures, or commits.
- `collect` keeps token counts and metadata only, no message text or tool output. Keep it so.
- `gitleaks` runs in `mise run check`; don't commit secrets or real transcript excerpts.

## Results and constants

- The numbers in `README.md` ("Results") and `replay.FITTED_W` / `replay.FITTED_MODEL_W` come
  from running the tool on real data. Don't change them unless you have rerun
  `fit`/`fit-log`/`replay` on that data; when you do, update both and the date in the heading.
- Dated constants (`replay.SWITCH`, the 200K window switch) and thresholds
  (`windows.FABLE_SHARE`, `MIN_FABLE_WINDOWS`, `replay.MIN_COMPACTIONS`) are part of the method
  described in the README; change them together with its "Method" section.
