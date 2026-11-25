#!/usr/bin/env bash
# Rate-limit logger for `cc-compact-tuning fit-log`, run as the Claude Code status line.
#
# Claude Code pipes the status-line JSON to this script on each refresh. For subscription
# accounts that JSON carries `rate_limits` (5-hour and 7-day used percentage and reset time).
# Each time a session's numbers change, one tab-separated line is appended to the log:
#
#   time  session (first 8 chars)  model  five_hour_pct  five_hour_resets_at  seven_day_pct  seven_day_resets_at
#
# Comparing with the session's own last line, not the log's, keeps an idle session that still
# shows another account's numbers from writing on every refresh. Past CC_TUNING_RATE_LOG_MAX
# lines (150000) the log is cut to its last CC_TUNING_RATE_LOG_KEEP (100000); run
# `cc-compact-tuning collect` often enough to save lines before they go. Per-session state
# older than a week is dropped when a new session starts.
#
# Use it alone, which prints the two percentages:
#   "statusLine": {"type": "command", "command": "~/src/cc-compact-tuning/statusline/rate-limit-log.sh"}
# or in front of an existing status line, which then gets the same JSON on stdin:
#   "statusLine": {"type": "command",
#                  "command": "~/src/cc-compact-tuning/statusline/rate-limit-log.sh ~/.claude/statusline.sh"}
#
# The log goes to CC_TUNING_RATE_LOG, default ~/.local/state/claude-statusline/rate-limits.tsv,
# the same default `cc-compact-tuning` reads. Needs bash and jq; uses flock when present.

input=$(cat)

{
  IFS= read -r limits
  IFS= read -r session
  IFS= read -r short
} < <(jq -r '
  (if (.rate_limits.five_hour.used_percentage | type) == "number" then
     [.model.id // "", .rate_limits.five_hour.used_percentage, .rate_limits.five_hour.resets_at,
      .rate_limits.seven_day.used_percentage, .rate_limits.seven_day.resets_at]
     | map(. // "" | tostring) | join("\t")
   else "" end),
  (.session_id // ""),
  ([(.rate_limits.five_hour.used_percentage // empty | "5h \(round)%"),
    (.rate_limits.seven_day.used_percentage // empty | "7d \(round)%")] | join(" · "))
' <<<"$input" 2>/dev/null)

log=${CC_TUNING_RATE_LOG:-$HOME/.local/state/claude-statusline/rate-limits.tsv}
log_dir=$(dirname "$log")
if [[ -n $limits && -n $session ]]; then
  state=$log_dir/sessions/$session
  last=$(cat "$state" 2>/dev/null)
  if [[ $last != "$limits" ]]; then
    if [[ -z $last ]]; then
      mkdir -p "$log_dir/sessions"
      find "$log_dir/sessions" -type f -mtime +7 -delete 2>/dev/null
    fi
    printf '%s\n' "$limits" >"$state"
    [[ -s $log ]] || printf 'time\tsession\tmodel\tfive_hour_pct\tfive_hour_resets_at\tseven_day_pct\tseven_day_resets_at\n' >>"$log"
    printf '%s\t%s\t%s\n' "$(date +%s)" "${session:0:8}" "$limits" >>"$log"
    if (($(wc -l <"$log") > ${CC_TUNING_RATE_LOG_MAX:-150000})); then
      prune() { { head -n 1 "$log"; tail -n "${CC_TUNING_RATE_LOG_KEEP:-100000}" "$log"; } >"$log.tmp" && mv "$log.tmp" "$log"; }
      if command -v flock >/dev/null; then
        (flock -n 9 || exit; prune) 9>"$log_dir/prune.lock"
      else
        prune
      fi
    fi
  fi
fi

if (($# > 0)); then
  exec "$@" <<<"$input"
fi
printf '%s\n' "$short"
