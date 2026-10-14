# vela-scheduler-bench

A small, verifiable eval environment for scheduling agents - built as a work
sample for teams building agentic scheduling (like Vela).

Scheduling looks easy in a demo and breaks in production on exactly three
things: timezone math, multi-party constraints, and reschedules. This bench
pins all three down with hidden graders. The agent proposes a slot; the
grader checks it against every participant's real calendar. No self-reported
wins.

## Scenarios

Six pinned scenarios over the week of Mon 2026-11-02 (US DST just ended, so
PST = UTC-8 - the week timezone bugs love):

| Scenario | What it tests |
| --- | --- |
| `pst-ist-intro` | 30-min intro across the classic PST/IST overlap problem |
| `panel-3tz` | 45-min panel, PST + EST + IST all required, one busy conflict |
| `london-ist-nofri` | hard constraint: never land on a Friday |
| `reschedule-pst-est` | the confirmed slot just got busy - find a new one |
| `cet-pst-morning` | 60 min with a soft preference (candidate mornings) |
| `optional-shadow` | optional attendee's conflict must not block the slot |

## Scoring

Staged 0-1 reward per scenario, written by the grader only:

- stage 1 (0.2): well-formed tz-aware ISO answer, correct duration
- stage 2 (0.6): every required participant inside a real availability
  window (converted through their IANA timezone), no busy conflicts, hard
  constraints respected - partial credit per participant
- stage 3 (0.2): soft preferences

## Run it

No dependencies, Python 3.10+:

```bash
python scheduler_bench.py run --agent greedy    # naive reference bar
python scheduler_bench.py run --agent oracle    # grid-search solver: must be 1.000
python scheduler_bench.py test                  # unittest suite
```

Reference numbers:

- oracle: **1.000** - proves every shipped scenario is fully solvable
- greedy-naive: **0.550** - it reads the first participant's local window as
  UTC and loses exactly where timezone math matters, which is the point

To plug in a real agent, pass any callable `scenario -> {start_utc, end_utc}`
to `run()`. The grader does the rest.
