# Stairs specialist (job 11849) training curve

Requested in `coordination/inbox/to-cluster.md` (2026-10-03 entry, item 3), to decide whether
a stairs v2 retrain is worth a GPU-day. Source: `go2-spec-11849.out`, which logs every
iteration (not just every 1000th) — the table below samples it at the requested grid.

| iteration | mean reward | mean ep length | terrain_levels | illegal_contact (per ep) |
|---|---|---|---|---|
| 1000 | 42.62 | 949.91 | 1.50 | 0.71 |
| 2000 | 43.91 | 972.56 | 1.71 | 0.42 |
| 3000 | 41.39 | 918.91 | 1.86 | 0.96 |
| 4000 | 43.63 | 962.94 | 1.87 | 0.50 |
| 5000 | 46.05 | 980.03 | 1.92 | 0.29 |
| 6000 | 42.15 | 983.75 | 0.79 | 0.25 |
| 7000 | 41.88 | 989.86 | 0.85 | 0.38 |
| 8000 | 41.86 | 985.94 | 0.91 | 0.21 |
| 9000 | 42.37 | 995.16 | 0.86 | 0.17 |
| 9999 | 42.33 | 979.87 | 0.97 | 0.21 |

**Mean reward and mean episode length are flat across the entire run**, 1000 to 9999 — no
climb at any point, not just "flat from ~6k". `illegal_contact` is noisy (0.17–0.96) with no
trend.

**`terrain_levels` is not flat — it collapses.** Sampling finer around the transition (every
100 iterations, 5100–5900):

| iteration | 5100 | 5200 | 5300 | 5400 | 5500 | 5600 | 5700 | 5800 | 5900 |
|---|---|---|---|---|---|---|---|---|---|
| terrain_levels | 2.63 | 1.65 | 0.81 | 0.65 | 0.67 | 0.69 | 0.74 | 0.80 | 0.81 |

It peaks at **2.63** at iteration 5100 (its highest point in the whole run) and falls to
**0.81** by 5300 — a 69% drop in 200 iterations — then stays flat at 0.65–0.97 for the
remaining 4700 iterations. That is a step change, not a gradual plateau; reward/episode
length don't move when it happens, so whatever caused it isn't visible in the other metrics.
Not diagnosed here (would need the curriculum/terrain-advance logic and probably the full
per-iteration series, not this 1000-wide grid) — flagging it because it directly bears on
"was terrain level still climbing at 10k": peak-and-collapse at iteration 5100, not "still
climbing at 10k", is a third answer the entry's two-way framing didn't anticipate.

**For the stairs-v2 go/no-go**: reward was never climbing, and the curriculum's own difficulty
signal (terrain_levels) was actively retreating from its iteration-5100 peak for the back
45% of the run. Nothing in this curve argues for "more iterations of the same config would
help." Whether that collapse itself is fixable (and would be worth chasing before any
retrain) is a separate, undiagnosed question.
