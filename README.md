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

`fit` and `fit-log` save their bootstraps to the data directory, and `replay` prefers
`bootstrap_log.npz` when present (falling back to `bootstrap.npy`).

Paths come from the environment and default to:

| Variable | Default |
|---|---|
| `CC_TUNING_PROJECTS_DIR` | `~/.claude/projects` |
| `CC_TUNING_DATA_DIR` | `~/.local/share/cc-compact-tuning` (store and bootstraps) |
| `CC_TUNING_RATE_LOG` | `~/.local/state/claude-statusline/rate-limits.tsv` |

## Development

```sh
mise install && mise run check   # ruff lint and format check, pytest, actionlint, gitleaks
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
limit hits, logins, cost-state records and the rate-limit log. It comes to about 2.3 MB
for 58K requests. Tool output and message text are not kept.

A systemd user timer runs `mise run collect` daily:

```sh
ln -sf ~/src/cc-compact-tuning/systemd/cc-compact-tuning-collect.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now cc-compact-tuning-collect.timer
```

The data directory is outside the repository: it holds your own usage.

## Method

- **Requests.** Deduplicated by message id across all transcripts, subagents included. Both
  first-chunk (`t`) and last-chunk (`t_end`) timestamps are stored. A request on another model
  is converted to Opus-equivalent tokens per category using API list prices in `windows`/`fit`,
  while `fit-log` fits Fable directly on raw tokens.
- **Windows.** A 5-hour window that hit the limit spans reset − 5h to the hit. Accounts are
  switched with `/login` or outside transcripts; only requests after the last `/login` count.
  When the rate-limit log shows multiple accounts active during a window, requests and
  compactions are attributed by `(session, timestamp)` to the hitting `(5h reset, 7d reset)`
  account (flagging if `>5%` of tokens are unattributed); single-account windows keep all
  requests in the time span so unlogged headless (`claude -p`) runs are preserved. A window
  is also flagged if its account started `>10m` before the first logged request, if two
  unattributed windows overlap, or if the rate-limit log shows an unexplained `≥8%` jump
  (`<25K` API tokens per percentage point).
- **Fit (`fit`).** Summary requests for compactions inside each window are added using
  `replay.measure_all` (when enough compactions are measured). Free weights: 1h cache write,
  cache read, output, and factors for Sonnet and Fable (relative to API price ratios). Uncached
  input and 5m writes are tied to 1h writes at API ratios (`0.5` and `0.625`). Loss is
  `log(window total)`. Reports degrees-of-freedom-adjusted RMS, leave-one-out RMS, per-window
  leverage, a date-step confound check at the first Fable window, and an output/input ratio
  profile (while `out=5` is a rescaling of the free fit, not a constrained fit).
- **Fable guard (`fit`).** A hit window that is `≥10%` Fable is flagged until there are at
  least `3` clean ones (`MIN_FABLE_WINDOWS`), so a free Fable factor does not absorb single-window
  error.
- **Rate-limit log fit (`fit-log`).** Groups rows by `(5h reset, 7d reset)`, keeps the
  monotonic progression strictly `<100%` (skipping uncollected future lines, stale dips, `100%`
  censoring, and unexplained `≥8%` jumps), and extracts one whole-window observation (`dp ≥ 15%`)
  per window, attributed by `(session, timestamp)` when accounts overlap (preserving single-account
  headless runs). Summary requests are included. With enough variance in 5m writes and Fable
  windows, `fit-log` frees `cw5` (relative to `cw1`) and Fable's own `(write, read, output)`
  weights alongside `read`, `output`, and Sonnet; bootstrap resamples over windows exclude
  resamples missing a model instead of filling `1.0`.
- **Replay.** Main-thread sessions since the switch to a 200K window (`2026-10-27`) are re-run
  at other triggers. The summary request is measured from cost-state minus transcript totals;
  growth `g` excludes compaction boundaries (allowing non-compaction negative deltas such as
  microcompactions). The simulator clamps counterfactual context `cf ≥ min(c, SYSTEM)` and
  caps/shrinks cache writes to `cf` (or `cf - SYSTEM` on cold caches). The analytical optimum
  is `T* = R + sqrt(2 g (rf R + K0 / r))`. Bootstrap intervals over `replay` reflect weight
  uncertainty only (using discrete nearest-rank percentiles so `"never"` is not interpolated),
  not session sampling or extrapolation beyond observed compaction triggers.
- **Other models.** `replay --model` replays a model's own sessions once it has `10` measured
  auto compactions (`MIN_COMPACTIONS`), else Opus's sessions. For models with per-model weights
  fitted directly by `fit-log` (`FITTED_MODEL_W`, currently Fable), `replay` uses those weights
  directly on raw tokens rather than scaling by API price ratios.

## Results (2026-11-23)

### Weights

Fitted on **57 clean whole windows** (`dp ≥ 15%`) in `fit-log` (Nov 12–23; 1 window skipped for an
unexplained `+9%` external-usage jump; `0.06%` unattributed tokens in overlaps; `rms = 1.95`
percentage points, median `|error| = 0.74pp`), and cross-checked on **29 clean hit windows** out
of 36 in `fit` (Oct 15–Nov 23; in-sample `rms = 2.82%`, dof-adjusted `3.10%`, leave-one-out
`4.34%`):

| weight (relative to input = 1, 1h write = 2) | API | `fit` (29 hit windows, with summaries) | `fit-log` (57 log windows, free Fable) |
|---|---|---|---|
| 5m cache write (`cw5`) | 1.25 | 1.25 (tied) | 2.05 (80%: 1.87–3.17) |
| cache read | 0.05 | 0.0324 (80%: 0.022–0.045) | 0.0368 (80%: 0.0351–0.0395) |
| output | 5 | 6.34 (80%: 3.9–10.7) | 7.63 (80%: 6.34–8.49) |
| Sonnet token vs Opus token | 0.5 | 0.97 (80%: 0.89–1.34) | 0.74 (80%: 0.57–0.89) |
| Fable write / output vs Opus | 2.0 | 4.02 (80%: 3.75–4.31) | 4.32 write / 3.24 output |
| Fable cache read vs Opus read | 1.0 | 2.01 (tied to API ratio) | 2.65 (80%: 1.41–3.89) |

- **5m cache writes (`cw5`)** count about the same as 1h writes (`2.05` vs `2.0`, 80%: `1.87–3.17`),
  rather than `1.25` at API ratios.
- **Opus cache reads** are consistently cheaper relative to writes than on the API (`~0.037` vs
  `0.05` when `write = 2`), while `output` is loosely identified (`~5–8×`). `fit-log`'s Opus
  budget is `10.21M` input units per 5h (`9.3M` in `fit`).
- **Date confound resolved.** Adding Nov 16 Fable and Nov 22 Opus hit windows rejects a Nov 12
  budget drop (`rms 10.12%` for `budget ×0.70` with `Fable = 1`, vs `2.82%` with Fable free;
  freeing both gives `Fable ×1.89`, `budget ×0.95`).
- **Sonnet caveat.** In `fit`, Sonnet still leans on one clean window (`Tue 11-10 20:58`, leverage
  `0.98`, LOO `113.7%`); `fit-log` measures Sonnet across multiple windows at `×1.47` its API
  ratio (`0.74` Opus, 80%: `0.57–0.89` Opus).
- **Fable cache reads.** In Fable's own token units (`1h write = 2`), `fit-log` gives Fable
  `read = 0.0226` (80%: `0.0116–0.0414`) and `out = 5.74` (80%: `5.12–7.71`). The ratio of
  Fable's `(read / write)` to Opus's `(read / write)` is `0.61` (bootstrap median `0.67`, 80% CI:
  `0.28–1.10`), spanning both the API half-read ratio (`0.50`) and flat scaling (`1.00`, which
  also fits at `rms = 2.20pp`, `med = 0.68pp`, `Fable = 3.54× Opus`).

### Compaction

- **Opus (`107` sessions / `213` compactions):** summary request averages `4.2K` input, `1.0K`
  cache write, `96%` context read, `6.7K` output (median `6.6K`); post-compaction request
  (`n = 281`) has `47.3K` context (`34.7K` written); growth mean `2.15K` (median `1.26K`) per
  request; one compaction costs `126K` input units plus reading the context.
- **Fable (`10` sessions / `15` compactions):** summary request averages `5.3K` input, `1.2K`
  cache write, `96%` context read, `10.8K` output; post-compaction request (`n = 15`) has
  `42.9K` context (`30.3K` written); growth mean `3.22K` (median `2.17K`) per request; one
  compaction costs `129K` Fable input units plus reading the context.

### Optimum

| Model (sessions replayed) | flat range (`≤1%` of best) | grid best (window) | current `168K` | `270K` (`303K` window) |
|---|---|---|---|---|
| **Opus** (`729` sessions, `294` comps) | **150K–200K** (`+0.2–0.5%`) | 180K (`213K`, analytic 169K) | +0.3% | +3.2% |
| **Fable** (`52` own sessions, `16` comps) | **160K–200K** (`≤1.2%`) | 168K (`201K`, analytic 236K) | +0.0% | +2.9% (API) / +1.9% (fit-log) |

- **Opus:** Under `500` `fit-log` bootstrap weight samples (weight uncertainty only), the grid best
  is `180K` across the `10/50/90%` range (`168K` penalty `0.3 / 0.3 / 0.4%`, `270K` penalty
  `3.1 / 3.3 / 3.5%`). Across the 4 weekly cohorts (`10-27` to `11-17`), the best trigger stays
  in `150K–180K`. Under API weights the grid best is `150K` (`168K` is `+1.0%`, `270K` is `+7.3%`).
- **Fable:** Now has `16` auto compactions of its own (`≥10`) and replays its own sessions using
  `fit-log`'s Fable weights (`[1, 2.05, 2, 0.0226, 5.74]`). The best trigger is `168K` across
  all `500` bootstrap weight samples (`10/50/90%` = `168K / 168K / 168K`, `270K` penalty
  `2.1 / 2.5 / 2.9%`). (The earlier `250K` recommendation came from replaying Opus sessions
  assuming Fable's half-price API read ratio; on Fable's own higher-growth sessions and fitted
  weights, `250K` costs `+1.5%`.)
- **Sonnet & Haiku:** `<10` measured auto compactions, so both replay the Opus sessions priced at
  their API ratios (`Sonnet` optimum `150K–200K`, `Haiku` `90K` due to its `5×` rate above `100K`).

## Rate-limit log

`~/.claude/statusline.sh` appends to `~/.local/state/claude-statusline/rate-limits.tsv`:
time, session, model, 5h % and reset, 7d % and reset.

- **When it writes:** only when a session's own numbers change, so an idle session showing
  another account's numbers stays quiet.
- **Pruning:** past 150 000 lines the log is cut to its last 100 000. `collect` has saved
  the older lines by then. Per-session state older than a week is deleted; it only holds
  the session's last line, which is already in the log.
- **Fitting:** `fit-log` extracts whole-window observations per `(5h reset, 7d reset)` account,
  attributing overlapping accounts by `(session, timestamp)` and preserving single-account
  headless (`claude -p`) usage, which pins down `cw5`, Opus weights, Sonnet, and Fable's own
  weights without single-window leverage issues.
