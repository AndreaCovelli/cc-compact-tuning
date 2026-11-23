"""Extract per-request usage, compactions, usage-limit hits and logins from Claude Code transcripts.

Claude Code deletes transcripts after cleanupPeriodDays, so every run merges what it finds into the
store instead of starting over: requests are keyed by message id, and records whose transcripts
have since been deleted are kept. Both first-chunk (`t`) and last-chunk (`t_end`) timestamps are
preserved across streaming updates. The status line's rate-limit log, which it cuts down past
150 000 lines, is merged the same way.
"""

import datetime as dt
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

from .store import Store

HIT = re.compile(r"You've hit your (session|5-hour|weekly) limit · resets (.+?) \(([^)]+)\)")
RESET = re.compile(r"(?:([A-Z][a-z]{2}) (\d{1,2}), )?(\d{1,2})(?::(\d\d))?([ap]m)")


def ts(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def parse_reset(text: str, tzname: str, hit: float) -> float | None:
    """Turn '9:20pm', '2pm' or 'Nov 14, 2pm' into the first such moment after the hit."""
    m = RESET.fullmatch(text)
    if not m:
        return None
    tz = ZoneInfo(tzname)
    mon, day, h, mi, ap = m.groups()
    hour = int(h) % 12 + (12 if ap == "pm" else 0)
    local = dt.datetime.fromtimestamp(hit, tz)
    if mon:
        month = dt.datetime.strptime(mon, "%b").month
        year = local.year + (1 if month < local.month else 0)
        t = dt.datetime(year, month, int(day), hour, int(mi or 0), tzinfo=tz)
    else:
        t = local.replace(hour=hour, minute=int(mi or 0), second=0, microsecond=0)
        if t.timestamp() <= hit:
            t += dt.timedelta(days=1)
    return t.timestamp()


def _request(d: dict, session: str, agent: str | None) -> dict:
    u = d["message"]["usage"]
    cc = u.get("cache_creation") or {}
    cw = u.get("cache_creation_input_tokens") or 0
    cw1 = cc.get("ephemeral_1h_input_tokens") or 0
    t = ts(d["timestamp"])
    return dict(
        t=t,
        t_end=t,
        model=d["message"].get("model"),
        session=session,
        agent=agent,
        side=bool(d.get("isSidechain")) or agent is not None,
        inp=u.get("input_tokens") or 0,
        cw1=cw1,
        cw5=max(cw - cw1, 0),
        cr=u.get("cache_read_input_tokens") or 0,
        out=u.get("output_tokens") or 0,
        speed=u.get("speed"),
    )


def scan(path: Path, root: Path, store: Store) -> None:
    """Merge one transcript, <project>/<session>.jsonl or <project>/<session>/subagents/agent-*.jsonl."""
    parts = path.relative_to(root).parts
    session = parts[1].removesuffix(".jsonl")
    agent = parts[-1].removesuffix(".jsonl") if "subagents" in parts else None
    R, C, H = store["requests"], store["compactions"], store["hits"]
    with open(path, errors="ignore") as f:
        for line in f:
            if not any(k in line for k in ('"usage"', "compact_boundary", "cost-state", "Login successful")):
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict):
                continue
            if d.get("type") == "cost-state":
                store["cost_states"][session if agent is None else f"{session}/{agent}"] = d
                continue
            # A successful /login may switch to another account, which has its own limits.
            if d.get("type") == "user" and not agent and "timestamp" in d:
                c = (d.get("message") or {}).get("content")
                if isinstance(c, str) and c.startswith("<local-command-stdout>Login successful"):
                    store["logins"][d["uuid"]] = dict(t=ts(d["timestamp"]), session=session)
                continue
            if d.get("subtype") == "compact_boundary":
                md = d.get("compactMetadata") or {}
                C[d["uuid"]] = dict(
                    t=ts(d["timestamp"]),
                    session=session,
                    agent=agent,
                    trigger=md.get("trigger"),
                    pre=md.get("preTokens"),
                    post=md.get("postTokens"),
                    version=d.get("version"),
                )
                continue
            m = d.get("message")
            if d.get("type") != "assistant" or not isinstance(m, dict) or not m.get("usage"):
                continue
            if m.get("model") == "<synthetic>":
                if d.get("error") == "rate_limit":
                    for c in m.get("content") or []:
                        hm = HIT.search(c.get("text", "")) if isinstance(c, dict) else None
                        if hm:
                            t = ts(d["timestamp"])
                            H[d["uuid"]] = dict(
                                t=t,
                                kind="weekly" if hm[1] == "weekly" else "5h",
                                reset=parse_reset(hm[2], hm[3], t),
                                text=hm[0],
                                session=session,
                            )
                continue
            mid = m.get("id")
            if not mid:
                continue
            r = _request(d, session, agent)
            old = R.get(mid)
            if old:
                # Streaming writes a message several times: the last copy has the final output
                # count and completion time, the first the time of the request. A resumed session
                # copies earlier messages into its own file, so the first owner is kept.
                t = min(r["t"], old["t"])
                t_end = max(r["t_end"], old.get("t_end", old["t"]))
                if old["out"] > r["out"]:
                    r = dict(old)
                r |= dict(t=t, t_end=t_end, session=old["session"], agent=old["agent"], side=old["side"])
            R[mid] = r


def merge_rate_log(path: Path, store: Store) -> None:
    if not path.exists():
        return
    with open(path) as f:
        next(f, None)
        for line in f:
            line = line.rstrip("\n")
            if line.count("\t") == 6:
                store["rate_limits"][line] = 1  # the line is its own key


def collect(projects_dir: Path, rate_log: Path, store: Store) -> int:
    """Merge every transcript and the rate-limit log into the store; return the files scanned."""
    paths = sorted(projects_dir.rglob("*.jsonl"))
    for p in paths:
        scan(p, projects_dir, store)
    merge_rate_log(rate_log, store)
    return len(paths)
