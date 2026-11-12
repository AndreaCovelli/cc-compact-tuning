"""Extract per-request usage, compactions and usage-limit hits from Claude Code transcripts.

Claude Code deletes transcripts after cleanupPeriodDays (30 by default), so every run merges
what it finds into data/store.json.gz instead of starting over: requests are keyed by
message id, and records that have since vanished from ~/.claude/projects are kept. The status
line's rate-limit log, which it cuts down past 150 000 lines, is merged the same way.
"""
import datetime as dt
import gzip
import json
import os
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = os.path.expanduser("~/.claude/projects")
STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "store.json.gz")
ROME = ZoneInfo("Europe/Rome")
RATE_LOG = os.path.expanduser("~/.local/state/claude-statusline/rate-limits.tsv")
HIT = re.compile(r"You've hit your (session|5-hour|weekly) limit · resets (.+?) \(([^)]+)\)")


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def parse_reset(text, tzname, hit):
    """Turn '9:20pm', '2pm' or 'Nov 14, 2pm' into the first such moment after the hit."""
    tz = ZoneInfo(tzname)
    m = re.fullmatch(r"(?:([A-Z][a-z]{2}) (\d{1,2}), )?(\d{1,2})(?::(\d\d))?([ap]m)", text)
    if not m:
        return None
    mon, day, h, mi, ap = m.groups()
    h = int(h) % 12 + (12 if ap == "pm" else 0)
    local = dt.datetime.fromtimestamp(hit, tz)
    if mon:
        month = dt.datetime.strptime(mon, "%b").month
        year = local.year + (1 if month < local.month else 0)
        t = dt.datetime(year, month, int(day), h, int(mi or 0), tzinfo=tz)
    else:
        t = local.replace(hour=h, minute=int(mi or 0), second=0, microsecond=0)
        if t.timestamp() <= hit:
            t += dt.timedelta(days=1)
    return t.timestamp()


def load_store():
    if os.path.exists(STORE):
        with gzip.open(STORE, "rt") as f:
            return json.load(f)
    return {"requests": {}, "compactions": {}, "hits": {}, "cost_states": {}, "logins": {},
            "rate_limits": {}}


def files():
    out = subprocess.run(["rg", "--files", "-g", "*.jsonl", ROOT], capture_output=True, text=True).stdout
    return sorted(out.split())


def scan(path, store):
    rel = os.path.relpath(path, ROOT)
    parts = rel.split(os.sep)
    # <project>/<session>.jsonl or <project>/<session>/subagents/agent-<id>.jsonl
    session = parts[1].removesuffix(".jsonl")
    agent = parts[-1].removesuffix(".jsonl") if "subagents" in parts else None
    R, C, H = store["requests"], store["compactions"], store["hits"]
    with open(path, errors="ignore") as f:
        for line in f:
            if ('"usage"' not in line and "compact_boundary" not in line and "cost-state" not in line
                    and "Login successful" not in line):
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
                C[d["uuid"]] = dict(t=ts(d["timestamp"]), session=session, agent=agent,
                                    trigger=md.get("trigger"), pre=md.get("preTokens"),
                                    post=md.get("postTokens"), version=d.get("version"))
                continue
            if d.get("type") != "assistant":
                continue
            m = d.get("message") or {}
            u = m.get("usage")
            if not isinstance(m, dict) or not u:
                continue
            if m.get("model") == "<synthetic>":
                if d.get("error") == "rate_limit":
                    for c in m.get("content") or []:
                        hm = HIT.search(c.get("text", "")) if isinstance(c, dict) else None
                        if hm:
                            t = ts(d["timestamp"])
                            H[d["uuid"]] = dict(t=t, kind="weekly" if hm[1] == "weekly" else "5h",
                                                reset=parse_reset(hm[2], hm[3], t), text=hm[0],
                                                session=session)
                continue
            mid = m.get("id")
            if not mid:
                continue
            cc = u.get("cache_creation") or {}
            cw = u.get("cache_creation_input_tokens") or 0
            cw1 = cc.get("ephemeral_1h_input_tokens") or 0
            r = dict(t=ts(d["timestamp"]), model=m.get("model"), session=session, agent=agent,
                     side=bool(d.get("isSidechain")) or agent is not None,
                     inp=u.get("input_tokens") or 0, cw1=cw1, cw5=max(cw - cw1, 0),
                     cr=u.get("cache_read_input_tokens") or 0, out=u.get("output_tokens") or 0,
                     speed=u.get("speed"))
            old = R.get(mid)
            if old:
                # Streaming writes the same message several times; the last one has the final
                # output count, the first one the time the request was made.
                r["t"] = min(r["t"], old["t"])
                if old["out"] > r["out"]:
                    r = old | {"t": r["t"]}
                # A resumed session copies earlier messages into its own file: keep the original owner.
                r["session"], r["agent"], r["side"] = old["session"], old["agent"], old["side"]
            R[mid] = r


def main():
    store = load_store()
    for k in ("logins", "rate_limits"):
        store.setdefault(k, {})
    before = {k: len(v) for k, v in store.items()}
    paths = files()
    for p in paths:
        scan(p, store)
    if os.path.exists(RATE_LOG):
        with open(RATE_LOG) as f:
            next(f, None)
            for line in f:
                line = line.rstrip("\n")
                if line.count("\t") == 6:
                    store["rate_limits"][line] = 1     # the line is its own key
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    tmp = STORE + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(store, f)
    os.replace(tmp, STORE)
    print(f"{len(paths)} transcript files scanned")
    for k, v in store.items():
        print(f"  {k}: {len(v)} ({len(v) - before[k]:+d})")


if __name__ == "__main__":
    sys.exit(main())
