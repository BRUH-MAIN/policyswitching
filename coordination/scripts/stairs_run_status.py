#!/usr/bin/env python3
"""Early-stop check for a StairsV4a/V4b (or any run logging the V4 metrics) from its .out log.

    python3 coordination/scripts/stairs_run_status.py go2-spec-<jobid>.out

Prints reward, row histogram, achieved/commanded speed and stalled fraction at 250-iteration
steps (means over +-25 iterations), then the stop test agreed with the laptop (inbox
2026-10-06): at ~500 and ~1500 iterations, report if achieved speed is under 30% of commanded,
or if the mean row has not risen by 1500. Reward, episode length and falls are printed for
reference only: a robot that stands still scores well on all three (stairs v3).
"""
import re
import sys

K = {
  "rew": "Mean reward", "len": "Mean episode length", "row": "Curriculum/terrain_row_mean",
  "r01": "Curriculum/terrain_rows_0_1", "r23": "Curriculum/terrain_rows_2_3", "r45": "Curriculum/terrain_rows_4_5",
  "r67": "Curriculum/terrain_rows_6_7", "r89": "Curriculum/terrain_rows_8_9",
  "cmd": "Episode_Metrics/cmd_speed", "act": "Episode_Metrics/actual_speed",
  "mov": "Episode_Metrics/cmd_moving", "stl": "Episode_Metrics/stalled",
}


def load(path):
  it, rows = None, {}
  for line in open(path, errors="ignore"):
    m = re.search(r"Learning iteration (\d+)/", line)
    if m:
      it = int(m.group(1)); rows[it] = {}; continue
    if it is None:
      continue
    t = line.strip()
    for k, name in K.items():
      if t.startswith(name + ":"):
        try:
          rows[it][k] = float(t.split(":")[-1])
        except ValueError:
          pass
  return rows


def win(rows, i, k, w=25):
  v = [rows[j][k] for j in range(max(0, i - w), i + w + 1) if j in rows and k in rows[j]]
  return sum(v) / len(v) if v else float("nan")


def main():
  rows = load(sys.argv[1])
  last = max(rows)
  print(f"{sys.argv[1]}: latest iteration {last}")
  print("iter | speed/cmd  stalled(of moving) | mean row | rows 0-1 2-3 4-5 6-7 8-9 | reward   ep_len")
  pts = [i for i in range(25, last + 1, 250)] + [last - 25 if last > 50 else last]
  for i in sorted(set(pts)):
    ratio = win(rows, i, "act") / win(rows, i, "cmd")
    stall = win(rows, i, "stl") / win(rows, i, "mov")
    print(f"{i:5d} | {100 * ratio:7.0f}%  {100 * stall:9.0f}%     | {win(rows, i, 'row'):7.2f}  | "
          f"{win(rows, i, 'r01'):.2f} {win(rows, i, 'r23'):.2f} {win(rows, i, 'r45'):.2f} {win(rows, i, 'r67'):.2f} {win(rows, i, 'r89'):.2f} | "
          f"{win(rows, i, 'rew'):6.2f} {win(rows, i, 'len'):7.1f}")
  print()
  # NOT a diagnostic. Iteration 0-5 mean episode length (~78) and reward (~1) are identical with
  # the warm start's normalizer reset and with it kept (12581 vs 12586, same seed: 17.26 / 0.39 at
  # iteration 0 in both), so they are not a failure signature. An earlier version of this
  # script "REPORTed" on them and mis-flagged a healthy start. What did differ, early: achieved
  # speed at iteration ~36 was 43% of commanded kept vs 28% reset (see the table above).
  early = [rows[i]["len"] for i in range(0, 6) if i in rows and "len" in rows[i]]
  if early:
    print(f"@start (iterations 0-5): mean episode length {sum(early) / len(early):.0f} (informational only; see comment)")
  for at in (500, 1500):
    if last < at + 25:
      print(f"@{at}: not reached yet"); continue
    ratio = win(rows, at, "act") / win(rows, at, "cmd")
    verdict = []
    if ratio != ratio:  # NaN: the metrics are not in this log. Never read that as "ok".
      verdict.append("speed metrics MISSING from the log (not a V4 run?) -- cannot judge")
    elif ratio < 0.30:
      verdict.append(f"SPEED {100 * ratio:.0f}% of commanded < 30%")
    if at == 1500:
      start, now = win(rows, 100, "row", 75), win(rows, 1500, "row")
      if not now > start + 0.2:
        verdict.append(f"mean row not rising ({start:.2f} @100 -> {now:.2f} @1500)")
    print(f"@{at}: " + ("REPORT: " + "; ".join(verdict) if verdict else "ok (speed >= 30% of commanded" + (", mean row rising)" if at == 1500 else ")")))


if __name__ == "__main__":
  main()
