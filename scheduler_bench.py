#!/usr/bin/env python3
"""scheduler_bench.py - a small, verifiable eval environment for scheduling agents.

Built as a work sample for teams building agentic scheduling (like Vela).
Scheduling looks easy in a demo and fails in production on exactly three
things: timezone math, multi-party constraints, and reschedules. This bench
pins those down with hidden graders - an agent proposes a slot, the grader
checks it against every participant's real calendar. No self-reported wins.

Scenario families:
  - two-party across hard timezone pairs (PST/IST, CET/PST, London/IST)
  - panel interviews where every required attendee must be free
  - a reschedule, where the old slot is suddenly busy
  - hard constraints (no Friday) and soft preferences (candidate mornings)

Staged 0-1 reward per scenario:
  stage 1 (0.2)  well-formed answer, correct duration
  stage 2 (0.6)  every required participant inside an availability window,
                 no busy-conflict (partial credit). Hard constraints
                 (scheduling week, banned weekdays) are a gate: violation -> 0.0
  stage 3 (0.2)  soft preferences

No dependencies. Python 3.10+.

    python scheduler_bench.py run --agent greedy    # naive reference bar
    python scheduler_bench.py run --agent oracle    # grader consistency == 1.0
    python scheduler_bench.py test                  # unittest suite
"""
from __future__ import annotations

import argparse
import json
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc
GRID = timedelta(minutes=15)

# Week of Monday 2026-11-02. US DST ended 2026-11-01: PST = UTC-8, IST = UTC+5:30.
HORIZON_MONDAY = datetime(2026, 11, 2, tzinfo=UTC)
HORIZON_DAYS = 5  # Mon-Fri


# ---------------------------------------------------------------- model

@dataclass
class Participant:
    name: str
    tz: str
    # weekly availability: (weekday 0=Mon, start_min, end_min) in LOCAL time
    windows: list[tuple[int, int, int]]
    busy: list[tuple[datetime, datetime]] = field(default_factory=list)  # UTC
    required: bool = True


@dataclass
class Scenario:
    key: str
    title: str
    participants: list[Participant]
    duration_minutes: int
    no_weekdays: tuple[int, ...] = ()            # hard: banned weekdays
    prefer_morning_for: tuple[str, ...] = ()     # soft: local start < 12:00


def utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


def _window_utc(p: Participant, wday: int, start_min: int, end_min: int) -> tuple[datetime, datetime]:
    """Concrete UTC interval for one local availability window in horizon week."""
    date = (HORIZON_MONDAY + timedelta(days=wday)).date()
    local_tz = ZoneInfo(p.tz)
    s = datetime(date.year, date.month, date.day, tzinfo=local_tz) + timedelta(minutes=start_min)
    e = datetime(date.year, date.month, date.day, tzinfo=local_tz) + timedelta(minutes=end_min)
    return s.astimezone(UTC), e.astimezone(UTC)


def participant_free(p: Participant, start: datetime, end: datetime) -> bool:
    """Inside one full availability window (tz-correct) and clear of busy."""
    inside = False
    for wday, s_min, e_min in p.windows:
        ws, we = _window_utc(p, wday, s_min, e_min)
        if ws <= start and end <= we:
            inside = True
            break
    if not inside:
        return False
    return all(end <= bs or start >= be for bs, be in p.busy)


# ---------------------------------------------------------------- scenarios

def scenarios() -> list[Scenario]:
    nine_to_five = [(w, 9 * 60, 17 * 60) for w in range(5)]
    return [
        Scenario(
            key="pst-ist-intro",
            title="Intro call: candidate in PST, recruiter in IST, 30 min",
            participants=[
                Participant("candidate", "America/Los_Angeles", [(w, 7 * 60, 10 * 60) for w in range(5)]),
                Participant("recruiter", "Asia/Kolkata", [(w, 18 * 60, 23 * 60) for w in range(5)]),
            ],
            duration_minutes=30,
        ),
        Scenario(
            key="panel-3tz",
            title="Panel: PST + EST + IST, all required, 45 min, one conflict",
            participants=[
                Participant("hiring_manager", "America/Los_Angeles", [(w, 7 * 60, 11 * 60) for w in range(5)]),
                Participant("vp", "America/New_York", [(w, 9 * 60, 13 * 60) for w in range(5)],
                            busy=[(utc(2026, 11, 3, 15, 0), utc(2026, 11, 3, 16, 30))]),
                Participant("candidate", "Asia/Kolkata", [(w, 18 * 60, 23 * 60 + 30) for w in range(5)]),
            ],
            duration_minutes=45,
        ),
        Scenario(
            key="london-ist-nofri",
            title="London + IST, 30 min, no Friday (hard)",
            participants=[
                Participant("client", "Europe/London", nine_to_five),
                Participant("analyst", "Asia/Kolkata", [(w, 12 * 60, 21 * 60) for w in range(5)]),
            ],
            duration_minutes=30,
            no_weekdays=(4,),
        ),
        Scenario(
            key="reschedule-pst-est",
            title="Reschedule: confirmed slot just got busy (PST + EST, 30 min)",
            participants=[
                Participant("account_executive", "America/Los_Angeles", nine_to_five,
                            busy=[(utc(2026, 11, 4, 17, 0), utc(2026, 11, 4, 17, 30))]),
                Participant("champion", "America/New_York", nine_to_five),
            ],
            duration_minutes=30,
        ),
        Scenario(
            key="cet-pst-morning",
            title="CET recruiter + PST candidate, 60 min, candidate prefers mornings (soft)",
            participants=[
                Participant("recruiter", "Europe/Berlin", [(w, 15 * 60, 19 * 60) for w in range(5)]),
                Participant("candidate", "America/Los_Angeles", nine_to_five),
            ],
            duration_minutes=60,
            prefer_morning_for=("candidate",),
        ),
        Scenario(
            key="optional-shadow",
            title="IST + PST required, second PST shadow optional, 30 min",
            participants=[
                Participant("coordinator", "Asia/Kolkata", [(w, 19 * 60, 23 * 60) for w in range(5)]),
                Participant("interviewer", "America/Los_Angeles", [(w, 6 * 60, 9 * 60) for w in range(5)],
                            busy=[(utc(2026, 11, 2, 14, 0), utc(2026, 11, 2, 15, 0))]),
                Participant("shadow", "America/Los_Angeles", [(w, 6 * 60, 9 * 60) for w in range(5)],
                            required=False,
                            busy=[(utc(2026, 11, 3, 14, 0), utc(2026, 11, 3, 15, 0))]),
            ],
            duration_minutes=30,
        ),
    ]


# ---------------------------------------------------------------- grader

def grade(scenario: Scenario, answer: dict) -> tuple[float, str]:
    reasons = []
    score = 0.0

    # stage 1: shape + duration
    try:
        start = datetime.fromisoformat(str(answer["start_utc"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(answer["end_utc"]).replace("Z", "+00:00"))
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("naive datetimes")
        start, end = start.astimezone(UTC), end.astimezone(UTC)
    except (KeyError, ValueError, TypeError):
        return 0.0, "unparseable start_utc/end_utc (need tz-aware ISO)"
    if end - start != timedelta(minutes=scenario.duration_minutes):
        reasons.append(f"duration {end - start} != {scenario.duration_minutes}m")
    else:
        score += 0.2

    # stage 2: hard feasibility - hard-constraint violations zero the reward
    horizon_end = HORIZON_MONDAY + timedelta(days=HORIZON_DAYS)
    hard_fail = False
    if not (HORIZON_MONDAY <= start and end <= horizon_end):
        reasons.append("outside the scheduling week")
        hard_fail = True
    if start.weekday() in scenario.no_weekdays:
        reasons.append(f"lands on banned weekday {start.weekday()}")
        hard_fail = True
    if hard_fail:
        return 0.0, "; ".join(reasons)
    checks = []
    for p in scenario.participants:
        if not p.required:
            continue
        ok = participant_free(p, start, end)
        checks.append(ok)
        if not ok:
            reasons.append(f"{p.name} not free in that slot ({p.tz})")
    if checks:
        score += 0.6 * sum(checks) / len(checks)

    # stage 3: soft preferences
    prefs = []
    for name in scenario.prefer_morning_for:
        p = next((x for x in scenario.participants if x.name == name), None)
        if p is None:
            continue
        local_start = start.astimezone(ZoneInfo(p.tz))
        prefs.append(local_start.hour < 12)
        if local_start.hour >= 12:
            reasons.append(f"soft: {name} starts at {local_start.hour}:00 local (prefers morning)")
    if prefs:
        score += 0.2 * sum(prefs) / len(prefs)
    else:
        score += 0.2

    return round(min(score, 1.0), 4), "; ".join(reasons) or "valid slot"


# ---------------------------------------------------------------- agents

def greedy_agent(scenario: Scenario) -> dict:
    """Naive bar: first participant's first window, start read as UTC.

    This is the classic scheduling-agent bug - local window times treated as
    UTC - and it should lose exactly where timezone math matters.
    """
    p = scenario.participants[0]
    wday, s_min, _ = p.windows[0]
    date = (HORIZON_MONDAY + timedelta(days=wday)).date()
    start = datetime(date.year, date.month, date.day, tzinfo=UTC) + timedelta(minutes=s_min)
    return {
        "start_utc": start.isoformat(),
        "end_utc": (start + timedelta(minutes=scenario.duration_minutes)).isoformat(),
    }


def oracle_agent(scenario: Scenario) -> dict:
    """Grid-search solver: proves every shipped scenario is fully solvable."""
    day = HORIZON_MONDAY
    horizon_end = HORIZON_MONDAY + timedelta(days=HORIZON_DAYS)
    while day < horizon_end:
        t = day
        while t < day + timedelta(hours=24):
            end = t + timedelta(minutes=scenario.duration_minutes)
            score, _ = grade(scenario, {"start_utc": t.isoformat(), "end_utc": end.isoformat()})
            if score == 1.0:
                return {"start_utc": t.isoformat(), "end_utc": end.isoformat()}
            t += GRID
        day += timedelta(days=1)
    raise RuntimeError(f"scenario {scenario.key} has no perfect slot - fix the fixture")


AGENTS = {"greedy": greedy_agent, "oracle": oracle_agent}


# ---------------------------------------------------------------- runner

def run(agent_fn, agent_name: str) -> dict:
    trials = []
    for sc in scenarios():
        score, reason = grade(sc, agent_fn(sc))
        trials.append({"scenario": sc.key, "title": sc.title, "score": score, "reason": reason})
    reward = sum(t["score"] for t in trials) / len(trials)
    return {"agent": agent_name, "reward": round(reward, 4), "trials": trials}


def render(report: dict) -> str:
    lines = [f"agent: {report['agent']}", f"overall reward: {report['reward']:.3f}", ""]
    for t in report["trials"]:
        flag = "ok  " if t["score"] == 1.0 else ("part" if t["score"] > 0 else "miss")
        lines.append(f"[{flag}] {t['scenario']:<22} {t['score']:.2f}  {t['reason']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- tests

class TestWindows(unittest.TestCase):
    def test_ist_window_converts_to_utc_correctly(self):
        p = Participant("x", "Asia/Kolkata", [(0, 18 * 60, 23 * 60)])
        s, e = _window_utc(p, 0, 18 * 60, 23 * 60)
        self.assertEqual((s.hour, s.minute), (12, 30))  # 18:00 IST = 12:30 UTC
        self.assertEqual((e.hour, e.minute), (17, 30))

    def test_pst_window_after_dst_end(self):
        p = Participant("x", "America/Los_Angeles", [(0, 7 * 60, 10 * 60)])
        s, _ = _window_utc(p, 0, 7 * 60, 10 * 60)
        self.assertEqual(s.hour, 15)  # 07:00 PST = 15:00 UTC (no DST)


class TestGrader(unittest.TestCase):
    def test_naive_datetime_rejected(self):
        sc = scenarios()[0]
        score, reason = grade(sc, {"start_utc": "2026-11-02T15:00:00", "end_utc": "2026-11-02T15:30:00"})
        self.assertEqual(score, 0.0)
        self.assertIn("tz-aware", reason)

    def test_wrong_duration_loses_stage1(self):
        sc = scenarios()[0]
        ans = oracle_agent(sc)
        start = datetime.fromisoformat(ans["start_utc"])
        bad = {"start_utc": ans["start_utc"], "end_utc": (start + timedelta(minutes=45)).isoformat()}
        score, reason = grade(sc, bad)
        self.assertLess(score, 1.0)
        self.assertIn("duration", reason)

    def test_no_friday_enforced(self):
        # adversarial: this Friday slot fits every participant's real window,
        # so only the hard weekday ban can stop it - it must score zero
        sc = next(s for s in scenarios() if s.key == "london-ist-nofri")
        friday = utc(2026, 11, 6, 13, 0)  # 13:00 UTC Fri = 13:00 London, 18:30 IST
        score, reason = grade(sc, {"start_utc": friday.isoformat(),
                                   "end_utc": (friday + timedelta(minutes=30)).isoformat()})
        self.assertEqual(score, 0.0)
        self.assertIn("banned weekday", reason)

    def test_outside_scheduling_week_scores_zero(self):
        # adversarial: correct local times next week - every participant would
        # be free at these clock times, but the slot is outside the horizon
        sc = next(s for s in scenarios() if s.key == "pst-ist-intro")
        start = utc(2026, 11, 9, 15, 0)  # next Mon: 07:00 PST, 20:30 IST
        score, reason = grade(sc, {"start_utc": start.isoformat(),
                                   "end_utc": (start + timedelta(minutes=30)).isoformat()})
        self.assertEqual(score, 0.0)
        self.assertIn("outside the scheduling week", reason)

    def test_busy_conflict_caught(self):
        sc = next(s for s in scenarios() if s.key == "reschedule-pst-est")
        busy_slot = {"start_utc": utc(2026, 11, 4, 17, 0).isoformat(),
                     "end_utc": utc(2026, 11, 4, 17, 30).isoformat()}
        _, reason = grade(sc, busy_slot)
        self.assertIn("account_executive", reason)


class TestBenchmark(unittest.TestCase):
    def test_oracle_solves_everything(self):
        report = run(oracle_agent, "oracle")
        self.assertEqual(report["reward"], 1.0)

    def test_greedy_loses_on_timezone_math(self):
        report = run(greedy_agent, "greedy")
        self.assertLess(report["reward"], 0.85)

    def test_garbage_answer_scores_zero(self):
        sc = scenarios()[0]
        self.assertEqual(grade(sc, {})[0], 0.0)


# ---------------------------------------------------------------- cli

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--agent", default="greedy", choices=list(AGENTS))
    run_p.add_argument("--json", action="store_true")
    sub.add_parser("test")
    args = parser.parse_args()

    if args.cmd == "test":
        unittest.main(argv=[parser.prog], exit=True, verbosity=2)
        return
    report = run(AGENTS[args.agent], args.agent)
    print(json.dumps(report, indent=2) if args.json else render(report))


if __name__ == "__main__":
    main()
