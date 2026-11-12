# cc-compact-tuning

Fits how a Claude subscription's usage limits weigh each kind of token, using Claude Code
transcripts, then replays real sessions to find the auto-compact window that uses the least
of the limit.

```sh
uv run python collect.py      # merge ~/.claude/projects and the rate-limit log into data/store.json.gz
uv run python windows.py      # 5-hour windows that ended in a usage-limit hit
uv run python fit.py 1000     # weights from those windows, with 1000 bootstrap fits
uv run python replay.py       # compaction cost, replay of trigger points, optimum with uncertainty
uv run python fit_log.py      # weights from the status line's rate-limit log, once it has data
uv run python fit_log.py --synthetic   # check that fit on a log made from known weights
```

## Keeping the data

Claude Code deletes transcripts older than `cleanupPeriodDays`, which is 30 by default and
set to 365 here. `collect.py` saves what the analysis needs into `data/store.json.gz`, so
nothing is lost when a transcript goes. That is per-request token counts, compactions,
limit hits, logins, cost-state records and the rate-limit log. It comes to about 1.6 MB
for a month of transcripts that take 2.2 GB. Tool output and message text are not kept.

A systemd user timer runs it daily:

```sh
ln -sf ~/src/cc-compact-tuning/systemd/cc-compact-tuning-collect.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now cc-compact-tuning-collect.timer
```

`data/` is git-ignored: it holds your own usage.

## Method

- **Requests.** Deduplicated by message id across all transcripts, subagents included. A
  request on another model is converted to Opus-equivalent tokens per category, using API
  list prices.
- **Windows.** A 5-hour window that hit the limit used one whole budget, from reset − 5h to
  the hit. Accounts are switched with `/login`, so only requests after the last login count.
  A window is flagged when its account was in use more than 10 minutes before the first
  request the transcripts show (claude.ai, or another account's segment).
- **Fit.** Free weights: 1h cache write, cache read, output, and a Sonnet-vs-Opus factor.
  Uncached input and the 5m write are tied to the 1h write at API ratios; there is too
  little of them to fit. The loss is log(window total), each window counting as 100%.
- **Replay.** Main-thread Opus sessions since the switch to a 200K window (2026-10-27) are
  re-run with compaction at other triggers. The model is the old one: context grows by the
  measured amount per request, and a compaction resets it to the measured post-compaction
  size. The summary request is not in transcripts, so its cost comes from cost-state minus
  transcript totals.

## Results (2026-11-12)

### Weights

Fitted on 14 clean windows out of 20. Every clean window lands at 100% ± 2%.

| weight (relative to input) | API | Oct fit (16 hits, output fixed) | now |
|---|---|---|---|
| 1h cache write | 2 | 1.5 | 1.9 (80%: 1.2–2.7) |
| cache read | 0.05 | 0.026 | 0.025 (80%: 0.011–0.037) |
| output | 5 | 5 (assumed) | 5.3 (80%: 3.7–8.1) |
| Sonnet token vs Opus token | 0.5 | – | ≈1.0 (80%: 0.5–1.6) |

- The 1h write and cache read values are with output fixed at 5. Output is with the 1h write
  fixed at 2.
- The output ratio is only loosely identified. Fixed ratios from 2× to 8× fit almost as well
  (rms 2.0–2.4%), so the windows confirm 5× rather than pin it.
- The Sonnet factor rests on only two Sonnet-heavy windows.

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
| 1000 bootstrap fits | 170K / 180K / 230K | +0.2 / 0.3 / 1.1% | +0.2 / 2.1 / 3.6% |

Bootstrap rows are 10/50/90%. By the week each session started, the best trigger is 180K,
190K and 180K. The cost curve is flat between 150K and 200K.

### Fable

Fable's cache read is half Opus's relative to its own input. With the same structure, the
optimum moves to a trigger of about 235K (window ≈270K). That is not measured: there are
fewer than 100 Fable requests.

## Rate-limit log

`~/.claude/statusline.sh` appends to `~/.local/state/claude-statusline/rate-limits.tsv`:
time, session, model, 5h % and reset, 7d % and reset.

- **When it writes:** only when a session's own numbers change, so an idle session showing
  another account's numbers stays quiet.
- **Pruning:** past 150 000 lines the log is cut to its last 100 000. `collect.py` has saved
  the older lines by then. Per-session state older than a week is deleted; it only holds
  the session's last line, which is already in the log.
- **Fitting:** `fit_log.py` turns each window into hundreds of points instead of one hit. The
  weekly reset time tells the accounts apart.
