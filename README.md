# cc-compact-tuning

Fits how a Claude subscription's usage limits weigh each kind of token, using Claude Code
transcripts, then replays real sessions to find the auto-compact window that uses the least
of the limit.

> **Unofficial, and fragile by nature.** This reads Claude Code's local transcript files and
> the status line's rate-limit fields, neither of which Anthropic documents or promises to keep
> stable. How the usage limits weigh tokens is not published either: the weights here are
> estimates fitted to one account's data, and Anthropic can change them without notice (they
> have changed before). A Claude Code update can change the transcript format and break
> `collect`, and a change in limit accounting makes old fits stale. Treat the results as a
> measurement of your own usage at the time, not as a specification. Not affiliated with or
> endorsed by Anthropic.

## Usage

```sh
mise install                  # pinned Python, uv and actionlint from mise.toml / mise.lock
mise run setup                # uv sync --locked
mise run collect              # merge ~/.claude/projects and the rate-limit log into the store
uv run --locked cc-compact-tuning windows             # 5-hour windows that ended in a limit hit
uv run --locked cc-compact-tuning fit --bootstrap 1000   # weights from those windows
uv run --locked cc-compact-tuning replay              # compaction cost, trigger replay, optimum
uv run --locked cc-compact-tuning replay --weights 1 1.25 2 0.05 5   # with other weights
uv run --locked cc-compact-tuning replay --model fable   # priced as fable, sonnet or haiku
uv run --locked cc-compact-tuning fit-log             # weights from the status line's rate-limit log
uv run --locked cc-compact-tuning fit-log --synthetic # check that fit on a log made from known weights
```

`fit` saves its bootstrap to the data directory, and `replay` reads it from there.

Paths come from the environment and default to:

| Variable | Default |
|---|---|
| `CC_TUNING_PROJECTS_DIR` | `~/.claude/projects` |
| `CC_TUNING_DATA_DIR` | `~/.local/share/cc-compact-tuning` (store and bootstraps) |
| `CC_TUNING_RATE_LOG` | `~/.local/state/claude-statusline/rate-limits.tsv` |

## Development

```sh
mise install && mise run check   # ruff lint and format check, pytest, actionlint
mise run format                  # ruff format
mise tasks ls                    # every task
```

Without mise, the same checks with uv:

```sh
uv sync --locked
uv run --locked ruff check . && uv run --locked ruff format --check .
uv run --locked pytest
```

Change dependencies with `uv add` / `uv remove`, and tool versions in `mise.toml`
followed by `mise lock`. Never edit `uv.lock` or `mise.lock` by hand.

## Keeping the data

Claude Code deletes transcripts older than `cleanupPeriodDays`, which is 30 by default and
set to 365 here. `collect` saves what the analysis needs into
`~/.local/share/cc-compact-tuning/store.json.gz`, so
nothing is lost when a transcript goes. That is per-request token counts, compactions,
limit hits, logins, cost-state records and the rate-limit log. It comes to about 1.6 MB
for a month of transcripts that take 2.2 GB. Tool output and message text are not kept.

A systemd user timer runs `mise run collect` daily:

```sh
ln -sf ~/src/cc-compact-tuning/systemd/cc-compact-tuning-collect.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now cc-compact-tuning-collect.timer
```

The data directory is outside the repository: it holds your own usage.

## Method

- **Requests.** Deduplicated by message id across all transcripts, subagents included. A
  request on another model is converted to Opus-equivalent tokens per category, using API
  list prices.
- **Windows.** A 5-hour window that hit the limit used one whole budget, from reset − 5h to
  the hit. Accounts are switched with `/login`, so only requests after the last login count.
  A window is flagged when its account was in use more than 10 minutes before the first
  request the transcripts show (claude.ai, or another account's segment).
- **Fit.** Free weights: 1h cache write, cache read, output, and factors for Sonnet and Fable.
  Their tokens are converted to Opus at API prices first, so a factor multiplies the API
  ratio: Sonnet ×1 is 0.5 Opus tokens, Fable ×1 is 2 (1 for cache reads). Uncached input and
  the 5m write are tied to the 1h write at API ratios; there is too little of them to fit.
  The loss is log(window total), each window counting as 100%.
- **Fable.** A window that is 10% Fable or more is flagged until there are three of them.
  With fewer, a free factor would absorb each window's error rather than measure Fable.
- **Replay.** Main-thread Opus sessions since the switch to a 200K window (2026-10-27) are
  re-run with compaction at other triggers. The model is the old one: context grows by the
  measured amount per request, and a compaction resets it to the measured post-compaction
  size. The summary request is not in transcripts, so its cost comes from cost-state minus
  transcript totals.
- **Other models.** `replay --model` prices each request at that model's API price ratios to
  Opus, Haiku's 5× rate above 100K included, then weighs them per Opus token. It replays that
  model's own sessions once it has 10 measured auto compactions, else the Opus sessions: how a
  session grows and compacts depends on the work more than on the model. A Sonnet or Fable
  factor scales every trigger alike, so it changes neither the optimum nor the percentages.

## Results (2026-11-12)

> **Fable is not measured yet.** The one window with much Fable in it (Nov 12, a third of its
> usage) adds up to 68% of a budget when Fable is counted at its API price. If nothing else
> was using the account, Fable counts about ×2.35 its API ratio, ≈4.7 Opus tokens per input
> or output token. That window is flagged, since a factor fitted to one window says nothing.
> Left in the fit with Fable at its API price, it drove the read weight to zero and the replay
> to "never compact".

### Weights

Fitted on 14 clean windows out of 21. Every clean window lands at 100% ± 2%.

| weight (relative to input) | API | Oct fit (16 hits, output fixed) | now |
|---|---|---|---|
| 1h cache write | 2 | 1.5 | 1.9 (80%: 1.2–2.7) |
| cache read | 0.05 | 0.026 | 0.025 (80%: 0.011–0.037) |
| output | 5 | 5 (assumed) | 5.3 (80%: 3.7–8.1) |
| Sonnet token vs Opus token | 0.5 | – | ≈1.0 (80%: 0.5–1.6) |
| Fable token vs Opus token (input, output) | 2 | – | not yet fitted; ≈4.7 from one window |

- The 1h write and cache read values are with output fixed at 5. Output is with the 1h write
  fixed at 2.
- The output ratio is only loosely identified. Fixed ratios from 2× to 8× fit almost as well
  (rms 2.0–2.4%), so the windows confirm 5× rather than pin it.
- The Sonnet factor rests on the few Sonnet-heavy windows. With Sonnet at the Opus price,
  the clean windows' API-weighted totals vary by 3.0%; at its API price, by 7.4%.

### Compaction

- **Summary request:** 3.5K input, 0.9K cache write, reads 97% of the context, 6.6K output.
- **First request after a compaction:** 46.9K context, of which 33.7K is written.
- **Growth:** 2.15K per request.
- **Cost of one compaction:** 107K input units plus reading the context.

### Optimum

| | optimum trigger | current 168K (200K window) | 270K (300K window) |
|---|---|---|---|
| fitted weights | 180K (213K window) | +0.3% | +2.3% |
| API weights | 140–150K | +1.0% | +7.7% |
| 1000 bootstrap fits | 170K / 180K / 220K | +0.2 / 0.3 / 1.1% | +0.2 / 2.1 / 3.6% |

Bootstrap rows are 10/50/90%. By the week each session started, the best trigger is 180K,
190K and 180K. The cost curve is flat between 150K and 200K.

### Fable

Fable's cache read is half Opus's relative to its own input. With the same structure, the
optimum moves to a trigger of about 235K (window ≈270K). That assumes the subscription prices
Fable's tokens relative to each other as the API does, which is not measured yet. If Fable's
input and output count more while its cache reads do not (the one window above hints at it),
reads get relatively cheaper and the optimal window gets larger.

## Rate-limit log

`~/.claude/statusline.sh` appends to `~/.local/state/claude-statusline/rate-limits.tsv`:
time, session, model, 5h % and reset, 7d % and reset.

- **When it writes:** only when a session's own numbers change, so an idle session showing
  another account's numbers stays quiet.
- **Pruning:** past 150 000 lines the log is cut to its last 100 000. `collect` has saved
  the older lines by then. Per-session state older than a week is deleted; it only holds
  the session's last line, which is already in the log.
- **Fitting:** `fit-log` turns each window into hundreds of points instead of one hit, which is
  the quickest way to measure the Fable factor. The weekly reset time tells the accounts apart.
