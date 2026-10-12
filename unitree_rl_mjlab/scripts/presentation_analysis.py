"""Analyse the presentation evaluation (switch-follow-preregistration.md, Addendum 8).

Three switching controllers on the bank flat + rough + stairs v8b, against GeneralistV3 (the
matched generalist), stairs v8b alone and generalist v2, on randomised layouts with real
risers. The layout is the unit: mean success over layouts with a bootstrap interval over
layouts, and paired per-layout differences (scripts/switch_follow_layouts.py does the same for
Addendum 7). Writes every figure plus a markdown table file.

Usage (from unitree_rl_mjlab/):
  python scripts/presentation_analysis.py eval_results/switch_follow/present \
      --out ../report_content/figures/presentation
  # regression check of the bootstrap against Addendum 7 (final_report.md, Section V.H):
  python scripts/presentation_analysis.py --regress eval_results/switch_follow/random
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Display order = categorical palette slot order (validated as adjacent pairs, light mode).
ARMS = (
  ("hard:0.8:label", "A  oracle switch"),
  ("clf", "B  onboard-scan switch"),
  ("soft:0.8:0.0:label", "C  soft blend"),
  ("fixed:genv3", "Matched generalist (V3)"),
  ("fixed:stairs", "Stairs v8b alone"),
  ("fixed:gen2", "Generalist v2 (old)"),
)
SHORT = ("A", "B", "C", "G3", "v8b", "G2")
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CONTROLLERS = ARMS[:3]
CONDITIONS = {"saro": "tipping only (primary)", "training": "strict: any non-foot contact", "saro_d20": "tipping only, 20 ms delay"}
# Round 2 (Addendum 9): bank R, compute-matched generalist, strict falls primary.
ROUNDS = {
  "r1": dict(arms=ARMS, short=SHORT, conditions=CONDITIONS, primary="saro", clf_cond="saro", label_lead=0.3, clf_title="It switches late — too late for 17 cm risers", delay=("saro", "saro_d20"),
             delay_label="tipping-only definition"),
  "r2": dict(
    arms=(("hard:0.8:label", "A  oracle switch"), ("clf", "B  onboard-scan switch"), ("soft:0.8:0.0:label", "C  soft blend"),
          ("fixed:genv3", "Matched generalist (V3, 6k it)"), ("fixed:stairs", "Stairs v8b alone"),
          ("oldbank:hard:0.8:label", "A with the old bank")),
    short=("A", "B", "C", "G3", "v8b", "A-old"),
    conditions={"training": "strict: any non-foot contact (primary)", "saro": "tipping only", "training_d20": "strict, 20 ms delay"},
    primary="training", clf_cond="saro", label_lead=0.8, clf_title="Labels now lead by 0.8 m: it switches in time", delay=("training", "training_d20"), delay_label="strict definition"),
}
ROUND = ROUNDS["r1"]
HEIGHTS = (0.09, 0.12, 0.15, 0.17)
SEGMENT_NAMES = {"flat": "flat", "rough": "rough", "stairs_up": "up-flight", "stairs_down": "down-flight"}


def boot(values: np.ndarray, rng: np.random.Generator, n: int = 10000) -> tuple[float, float]:
  idx = rng.integers(0, len(values), size=(n, len(values)))
  means = values[idx].mean(axis=1)
  return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def arm_key(arms: dict, key: str) -> str | None:
  """The stored spec for a display key ('clf' matches whichever classifier spec was run)."""
  if key == "clf":
    return next((k for k in arms if k.startswith("clf:")), None)
  return key if key in arms else None


def load_condition(directory: Path) -> dict[int, dict]:
  """seed -> {'layout', 'course', 'arms'} with main and genv3 files joined."""
  out: dict[int, dict] = {}
  for f in sorted(directory.glob("L*_*.json")):
    seed = int(f.stem.split("_")[0][1:])
    d = json.loads(f.read_text())
    rec = out.setdefault(seed, dict(layout=d["layout"], course=d["course"], conditions=d["conditions"], arms={}))
    tag = f.stem.split("_", 1)[1]
    prefix = "" if tag in ("main", "genv3", "clf") else f"{tag}:"
    rec["arms"].update({prefix + k: v for k, v in d["arms"].items()})
  return out


def complete_seeds(runs: dict[int, dict], keys: list[str]) -> list[int]:
  return sorted(s for s, r in runs.items() if all(arm_key(r["arms"], k) for k in keys))


def success(runs, seeds, key) -> np.ndarray:
  return np.array([runs[s]["arms"][arm_key(runs[s]["arms"], key)]["summary"]["success_pct"] for s in seeds])


def trials(runs, seeds, key) -> list[tuple[int, dict]]:
  return [(s, t) for s in seeds for t in runs[s]["arms"][arm_key(runs[s]["arms"], key)]["trials"]]


def riser(runs, s) -> float:
  return round(float(runs[s]["conditions"]["step_height"]), 2)


def failure_segment(course: dict, end_x: float) -> str:
  x = 0.0
  for seg in course["segments"]:
    if end_x < x + seg["length"]:
      return seg["kind"]
    x += seg["length"]
  return course["segments"][-1]["kind"]


# ---------------------------------------------------------------- plotting helpers
def _style(ax, grid_axis="x"):
  ax.set_facecolor(SURFACE)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)
  ax.tick_params(colors=INK2, labelsize=10)
  ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
  ax.set_axisbelow(True)


def _fig(plt, w=9.0, h=4.8):
  fig, ax = plt.subplots(figsize=(w, h), dpi=200)
  fig.patch.set_facecolor(SURFACE)
  return fig, ax


def _save(fig, out: Path, name: str):
  fig.tight_layout()
  for ext in ("png", "svg"):
    fig.savefig(out / f"{name}.{ext}", facecolor=fig.get_facecolor())


# ---------------------------------------------------------------- figures
def fig_hero(plt, runs, seeds, rng, out, cond, title_suffix=""):
  fig, ax = _fig(plt, 9.5, 4.6)
  ys = np.arange(len(ARMS))[::-1]
  rows = []
  for y, (key, name), c in zip(ys, ARMS, COLORS):
    v = success(runs, seeds, key)
    lo, hi = boot(v, rng)
    ax.barh(y, v.mean(), height=0.62, color=c, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(v.mean(), y, xerr=[[v.mean() - lo], [hi - v.mean()]], fmt="none", ecolor=INK, elinewidth=1.4, capsize=4)
    ax.text(min(hi, 100) + 1.5, y, f"{v.mean():.1f}%", va="center", color=INK, fontsize=11, fontweight="bold")
    rows.append((name, v.mean(), lo, hi, v.min(), v.max()))
  ax.set_yticks(ys, [n for _, n in ARMS], color=INK, fontsize=11)
  ax.set_xlim(0, 112)
  ax.set_xlabel("course success, mean over layouts (%) — bars: 95% bootstrap interval over layouts", color=INK2)
  ax.set_title(f"Course success by arm — {CONDITIONS[cond]}{title_suffix}  ({len(seeds)} layouts)", loc="left", color=INK, fontsize=13)
  _style(ax)
  _save(fig, out, f"fig01_success_{cond}")
  plt.close(fig)
  return rows


def fig_paired(plt, runs, seeds, rng, out, cond):
  comps = [(k, n, "fixed:genv3", "matched generalist") for k, n in CONTROLLERS] + \
          [(k, n, "fixed:stairs", "stairs v8b alone") for k, n in CONTROLLERS]
  fig, ax = _fig(plt, 9.5, 5.2)
  rows = []
  for i, (a, an, b, bn) in enumerate(comps):
    d = success(runs, seeds, a) - success(runs, seeds, b)
    lo, hi = boot(d, rng)
    y = len(comps) - 1 - i
    jitter = (rng.random(len(d)) - 0.5) * 0.36
    c = COLORS[[k for k, _ in ARMS].index(a)]
    ax.scatter(d, y + jitter, s=18, color=c, alpha=0.55, edgecolors=SURFACE, linewidths=0.6, zorder=2)
    ax.errorbar(d.mean(), y, xerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o", color=INK, markersize=7, elinewidth=1.8, capsize=5, zorder=3)
    rows.append((f"{an.split('  ')[1]} − {bn}", d.mean(), lo, hi, int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())))
  ax.axvline(0, color=AXIS, linewidth=1.2, zorder=1)
  ax.set_yticks(range(len(comps))[::-1],
                [f"{an.split('  ')[0]} − {bn}\n{m:+.1f}  [{lo:+.1f}, {hi:+.1f}]   ahead {a} / behind {b}"
                 for (_, an, _, bn), (_, m, lo, hi, a, b, _) in zip(comps, rows)], color=INK, fontsize=9.5)
  ax.set_xlabel("per-layout success difference (points) · dots = layouts · black = mean, 95% bootstrap CI", color=INK2)
  ax.set_title(f"Paired per-layout differences — {CONDITIONS[cond]}", loc="left", color=INK, fontsize=12)
  _style(ax)
  _save(fig, out, f"fig02_paired_{cond}")
  plt.close(fig)
  return rows


def fig_by_height(plt, runs, seeds, out, cond):
  fig, ax = _fig(plt, 10.5, 4.8)
  table = {}
  for (key, name), c in zip(ARMS, COLORS):
    ys = []
    for h in HEIGHTS:
      hs = [s for s in seeds if riser(runs, s) == h]
      ys.append(success(runs, hs, key).mean() if hs else np.nan)
    table[name] = ys
    ax.plot([h * 100 for h in HEIGHTS], ys, color=c, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label=name)
  ax.set_xticks([h * 100 for h in HEIGHTS], [f"{round(h * 100)} cm" for h in HEIGHTS])
  ax.set_xlim(8.0, 18.0)
  ax.set_ylim(-3, 103)
  ax.set_ylabel("course success (%)", color=INK2)
  ax.set_xlabel("riser height of both staircases on the course (6 layouts each)", color=INK2)
  ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5), labelcolor=INK)
  ax.set_title(f"Success by riser height — {CONDITIONS[cond]}", loc="left", color=INK, fontsize=12)
  _style(ax, "y")
  _save(fig, out, f"fig03_by_height_{cond}")
  plt.close(fig)
  return table


def fig_outcomes(plt, runs, seeds, out, cond):
  fig, ax = _fig(plt, 9.5, 4.4)
  parts = (("success", "#2a78d6", "success"), ("fall", "#d03b3b", "fall"), ("lost", "#898781", "lost / timeout"))
  ys = np.arange(len(ARMS))[::-1]
  table = {}
  for y, (key, name) in zip(ys, ARMS):
    t = [tr for _, tr in trials(runs, seeds, key)]
    n = len(t)
    pct = {p: 100 * sum(x["outcome"] == p for x in t) / n for p, _, _ in parts}
    pct["lost"] = 100 - pct["success"] - pct["fall"]
    left = 0.0
    for p, c, _ in parts:
      ax.barh(y, pct[p], left=left, height=0.62, color=c, edgecolor=SURFACE, linewidth=2)
      if pct[p] >= 6:
        ax.text(left + pct[p] / 2, y, f"{pct[p]:.0f}%", va="center", ha="center", color="white", fontsize=9, fontweight="bold")
      left += pct[p]
    table[name] = pct
  ax.set_yticks(ys, [n for _, n in ARMS], color=INK, fontsize=10)
  ax.set_xlim(0, 100)
  ax.set_xlabel("share of all trials, pooled over layouts (%)", color=INK2)
  handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in parts]
  ax.legend(handles, [l for _, _, l in parts], frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), labelcolor=INK)
  ax.set_title(f"How trials end — {CONDITIONS[cond]}", loc="left", color=INK, fontsize=12, pad=26)
  _style(ax)
  _save(fig, out, f"fig04_outcomes_{cond}")
  plt.close(fig)
  return table


def fig_failure_location(plt, runs, seeds, out, cond):
  kinds = ("flat", "rough", "stairs_up", "stairs_down")
  fig, ax = _fig(plt, 9.5, 4.6)
  width = 0.13
  table = {}
  for i, ((key, name), c) in enumerate(zip(ARMS, COLORS)):
    pairs = trials(runs, seeds, key)
    n = len(pairs)
    counts = {k: 0 for k in kinds}
    for s, t in pairs:
      if t["outcome"] != "success":
        counts[failure_segment(runs[s]["course"], t["end_x"])] += 1
    vals = [100 * counts[k] / n for k in kinds]
    table[name] = dict(zip(kinds, vals))
    xs = np.arange(len(kinds)) + (i - 2.5) * width
    ax.bar(xs, vals, width=width * 0.92, color=c, edgecolor=SURFACE, linewidth=1.5, label=name)
  ax.set_xticks(range(len(kinds)), [SEGMENT_NAMES[k] for k in kinds], color=INK, fontsize=11)
  ax.set_ylabel("failed trials ending on this segment\n(% of all trials)", color=INK2)
  ax.legend(frameon=False, fontsize=9, ncol=2, labelcolor=INK)
  ax.set_title(f"Where failures happen — {CONDITIONS[cond]}", loc="left", color=INK, fontsize=12)
  _style(ax, "y")
  _save(fig, out, f"fig05_failure_location_{cond}")
  plt.close(fig)
  return table


def _trial_metric(runs, seeds, key, metric, successes_only=True):
  v = [t.get(metric) for _, t in trials(runs, seeds, key) if (t["outcome"] == "success" or not successes_only)]
  return np.array([x for x in v if x is not None], dtype=float)


def fig_quality(plt, runs, seeds, out, cond):
  metrics = (("ar_entry", "action rate entering\nnon-flat terrain"), ("ar_exit", "action rate leaving\nnon-flat terrain"),
             ("gap_rms", "following-gap error\nRMS (m)"), ("time_s", "time to finish\n(s)"))
  fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4.4), dpi=200)
  fig.patch.set_facecolor(SURFACE)
  table = {}
  for ax, (m, label) in zip(axes, metrics):
    data = [_trial_metric(runs, seeds, key, m) for key, _ in ARMS]
    parts = ax.boxplot(data, vert=True, widths=0.6, patch_artist=True, showfliers=False,
                       medianprops=dict(color=INK, linewidth=1.6), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    for patch, c in zip(parts["boxes"], COLORS):
      patch.set_facecolor(c)
      patch.set_edgecolor(SURFACE)
    ax.set_xticks(range(1, len(ARMS) + 1), SHORT, fontsize=9)
    ax.set_title(label, color=INK, fontsize=10)
    _style(ax, "y")
    table[m] = {n: float(np.median(d)) if len(d) else float("nan") for (_, n), d in zip(ARMS, data)}
  fig.suptitle(f"Motion and following quality, successful trials only — {CONDITIONS[cond]}   "
               "(A/B/C = switching controllers; G3 = matched generalist; v8b = stairs v8b alone; G2 = generalist v2)",
               x=0.01, ha="left", color=INK, fontsize=11)
  _save(fig, out, f"fig06_quality_{cond}")
  plt.close(fig)
  return table


def fig_delay(plt, base, delay, seeds, rng, out):
  fig, ax = _fig(plt, 9.5, 4.6)
  ys = np.arange(len(ARMS))[::-1]
  rows = []
  for y, (key, name), c in zip(ys, ARMS, COLORS):
    v0, v1 = success(base, seeds, key).mean(), success(delay, seeds, key).mean()
    ax.plot([v1, v0], [y, y], color=AXIS, linewidth=2, zorder=1)
    ax.scatter([v0], [y], s=70, color=c, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.scatter([v1], [y], s=70, facecolors=SURFACE, edgecolors=c, linewidths=2.2, zorder=3)
    ax.text(max(v0, v1) + 2, y, f"{v0:.0f}% → {v1:.0f}%", va="center", color=INK, fontsize=10)
    rows.append((name, v0, v1))
  ax.set_yticks(ys, [n for _, n in ARMS], color=INK, fontsize=10)
  ax.set_xlim(0, 118)
  ax.set_xlabel("course success (%): filled = no delay, hollow = every motor command 20 ms late", color=INK2)
  ax.set_title(f"Robustness to control latency ({ROUND['delay_label']})", loc="left", color=INK, fontsize=12)
  _style(ax)
  _save(fig, out, "fig07_delay")
  plt.close(fig)
  return rows


def fig_classifier(plt, runs, seeds, out, cond, meta_path: Path | None = None):
  key = arm_key(runs[seeds[0]]["arms"], "clf")
  fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=200, gridspec_kw=dict(width_ratios=[0.8, 1, 1.25]))
  fig.patch.set_facecolor(SURFACE)
  info: dict = dict(spec=key)
  # (1) Validation confusion matrix from training (held-out trials, layouts 2000-2011).
  ax = axes[0]
  if meta_path and meta_path.exists():
    meta = json.loads(meta_path.read_text())
    cm = np.array(meta["val_confusion_rows_true_cols_pred"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
      for j in range(3):
        ax.text(j, i, f"{100 * cmn[i, j]:.0f}%", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else INK, fontsize=11)
    ax.set_xticks(range(3), ["flat", "rough", "stairs"])
    ax.set_yticks(range(3), ["flat", "rough", "stairs"])
    ax.set_xlabel("predicted", color=INK2)
    ax.set_ylabel(f"true (label, {ROUND['label_lead']} m lead)", color=INK2)
    ax.set_title(f"Held-out accuracy {100 * meta['val_accuracy']:.1f}%", loc="left", color=INK, fontsize=11)
    info.update(val_accuracy=meta["val_accuracy"], val_recall=meta["val_recall"])
  # (2) Per-trial agreement with the label during the evaluation.
  agree = np.array([t.get("clf_agree") for _, t in trials(runs, seeds, "clf") if t.get("clf_agree") is not None], dtype=float)
  axes[1].hist(100 * agree, bins=30, color=COLORS[1], edgecolor=SURFACE)
  axes[1].set_xlabel("% of steps matching the terrain label, per trial", color=INK2)
  axes[1].set_title("Agreement on the test layouts", loc="left", color=INK, fontsize=11)
  info["agree_median"] = float(np.median(agree))
  # (3) Lead on up-flights: when stairs is first selected before the first riser.
  ax = axes[2]
  never = {}
  for h, c in zip(HEIGHTS, ("#86b6ef", "#5598e7", "#256abf", "#0d366b")):
    lv, nn, tot = [], 0, 0
    for s in [s for s in seeds if riser(runs, s) == h]:
      kinds = [g["kind"] for g in runs[s]["course"]["segments"] if g["kind"] != "flat"]
      iu = kinds.index("stairs_up")
      for t in runs[s]["arms"][key]["trials"]:
        l = (t.get("clf_leads") or [None] * len(kinds))[iu]
        tot += 1
        if l is None:
          nn += 1
        elif l < 1.49:  # 1.5 = already selected when the window opened
          lv.append(l)
    never[h] = 100 * nn / max(tot, 1)
    ax.hist(lv, bins=np.arange(-0.2, 1.0, 0.025), histtype="step", linewidth=2, color=c,
            label=f"{round(h * 100)} cm risers: never switched before stopping {never[h]:.0f}%")
  lead = ROUND["label_lead"]
  ax.axvline(lead, color=INK, linestyle="--", linewidth=1.2)
  ax.text(lead + 0.01, ax.get_ylim()[1] * 0.92, f"label switches\n{lead} m ahead", color=INK2, fontsize=8)
  if abs(lead - 0.8) > 1e-6:
    ax.axvline(0.8, color=COLORS[0], linestyle="--", linewidth=1.2)
    ax.text(0.81, ax.get_ylim()[1] * 0.92, "oracle A\n0.8 m ahead", color=INK2, fontsize=8)
  else:
    ax.text(lead + 0.01, ax.get_ylim()[1] * 0.78, "(= oracle A's lead)", color=INK2, fontsize=8)
  ax.set_xlabel("distance before the first up-step when 'stairs' is first chosen (m)", color=INK2)
  ax.set_title(ROUND["clf_title"], loc="left", color=INK, fontsize=11)
  ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
  info["never_switched_pct_by_height"] = {round(h * 100): v for h, v in never.items()}
  for ax in axes:
    _style(ax, "y")
  axes[0].grid(False)
  fig.suptitle("Arm B: the onboard height-scan classifier", x=0.01, ha="left", color=INK, fontsize=13)
  _save(fig, out, f"fig08_classifier_{cond}")
  plt.close(fig)
  return info


def fig_definitions(plt, base, strict, seeds, out):
  fig, ax = _fig(plt, 9.5, 4.4)
  ys = np.arange(len(ARMS))[::-1]
  rows = []
  for y, (key, name), c in zip(ys, ARMS, COLORS):
    v0, v1 = success(base, seeds, key).mean(), success(strict, seeds, key).mean()
    ax.barh(y + 0.17, v0, height=0.32, color=c, edgecolor=SURFACE, linewidth=1.5)
    ax.barh(y - 0.17, v1, height=0.32, color=c, alpha=0.45, edgecolor=SURFACE, linewidth=1.5)
    ax.text(v0 + 1.5, y + 0.17, f"{v0:.0f}%", va="center", color=INK, fontsize=9)
    ax.text(v1 + 1.5, y - 0.17, f"{v1:.0f}%", va="center", color=INK2, fontsize=9)
    rows.append((name, v0, v1))
  ax.set_yticks(ys, [n for _, n in ARMS], color=INK, fontsize=10)
  ax.set_xlim(0, 110)
  ax.set_xlabel("course success (%): solid = tipping-only falls; faded = any shin, knee or body contact counts as a fall", color=INK2)
  ax.set_title("Effect of the fall definition", loc="left", color=INK, fontsize=12)
  _style(ax)
  _save(fig, out, "fig09_fall_definition")
  plt.close(fig)
  return rows


# ---------------------------------------------------------------- main
def regress(directory: Path) -> None:
  """Recompute Addendum 7's headline numbers (final_report.md, Section V.H) with `boot`."""
  rng = np.random.default_rng(0)
  runs = {"A": {}, "B": {}}
  for f in sorted(directory.glob("[AB]_s*.json")):
    runs[f.name[0]][int(f.stem.split("_s")[1])] = json.loads(f.read_text())
  seeds = sorted(set(runs["A"]) & set(runs["B"]))
  succ = lambda r, a: np.array([runs[r][s]["arms"][a]["summary"]["success_pct"] for s in seeds])
  for r, a, expect in (("A", "fixed:stairs", 92.7), ("A", "fixed:gen2", 37.5), ("A", "hard:0.3:label", 92.1)):
    v = succ(r, a)
    print(f"{r} {a}: {v.mean():.1f} {boot(v, rng)}  (final_report: {expect})")
  d = succ("A", "hard:0.3:label") - succ("A", "fixed:stairs")
  print(f"switch 0.3 - stairs v2 alone: {d.mean():+.1f} {boot(d, rng)}  (final_report: -0.5, -2.0 to +0.9)")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("directory")
  ap.add_argument("--out", default="../report_content/figures/presentation")
  ap.add_argument("--regress", action="store_true")
  ap.add_argument("--round", default="r1", choices=tuple(ROUNDS))
  args = ap.parse_args()
  if args.regress:
    regress(Path(args.directory))
    return
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  plt.rcParams["font.family"] = "DejaVu Sans"

  global ARMS, SHORT, CONTROLLERS, CONDITIONS, ROUND
  ROUND = ROUNDS[args.round]
  ARMS, SHORT, CONDITIONS = ROUND["arms"], ROUND["short"], ROUND["conditions"]
  CONTROLLERS = ARMS[:3]
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  root = Path(args.directory)
  keys = [k for k, _ in ARMS]
  data = {c: load_condition(root / c) for c in CONDITIONS if (root / c).exists()}
  md = ["# Presentation evaluation: results tables", "",
        "Generated by `unitree_rl_mjlab/scripts/presentation_analysis.py` from "
        f"`unitree_rl_mjlab/eval_results/switch_follow/{root.name}/`. Pre-registration: "
        f"`coordination/results/switch-follow-preregistration.md`, Addendum {'9' if args.round == 'r2' else '8'}.", ""]
  summary = {}
  for cond, runs in data.items():
    seeds = complete_seeds(runs, keys)
    if not seeds:
      md += [f"## {CONDITIONS[cond]}", "", "No layout has every arm yet.", ""]
      continue
    rng = np.random.default_rng(0)
    hero = fig_hero(plt, runs, seeds, rng, out, cond)
    paired = fig_paired(plt, runs, seeds, rng, out, cond)
    md += [f"## {CONDITIONS[cond]} — {len(seeds)} layouts (seeds {seeds[0]}-{seeds[-1]})", "",
           "| arm | mean success (95% CI over layouts) | worst layout | best layout |", "|---|---|---|---|"]
    md += [f"| {n} | **{m:.1f}** ({lo:.1f}-{hi:.1f}) | {mn:.1f} | {mx:.1f} |" for n, m, lo, hi, mn, mx in hero]
    md += ["", "| comparison | mean per-layout difference (95% CI) | ahead / behind / tied |", "|---|---|---|"]
    md += [f"| {n} | **{m:+.1f}** ({lo:+.1f} to {hi:+.1f}) | {a} / {b} / {t} |" for n, m, lo, hi, a, b, t in paired]
    summary[cond] = dict(seeds=seeds, success=[dict(arm=n, mean=m, lo=lo, hi=hi) for n, m, lo, hi, _, _ in hero],
                         paired=[dict(comparison=n, mean=m, lo=lo, hi=hi, ahead=a, behind=b) for n, m, lo, hi, a, b, _ in paired])
    if cond in ("saro", "training"):
      byh = fig_by_height(plt, runs, seeds, out, cond)
      md += ["", "| arm | " + " | ".join(f"{round(h * 100)} cm" for h in HEIGHTS) + " |", "|---|" + "---|" * len(HEIGHTS)]
      md += [f"| {n} | " + " | ".join(f"{v:.1f}" for v in vals) + " |" for n, vals in byh.items()]
      oc = fig_outcomes(plt, runs, seeds, out, cond)
      md += ["", "| arm | success % | fall % | lost / timeout % |", "|---|---|---|---|"]
      md += [f"| {n} | {p['success']:.1f} | {p['fall']:.1f} | {p['lost']:.1f} |" for n, p in oc.items()]
      fl = fig_failure_location(plt, runs, seeds, out, cond)
      md += ["", "| arm (failed trials as % of all, by segment) | flat | rough | up-flight | down-flight |", "|---|---|---|---|---|"]
      md += [f"| {n} | " + " | ".join(f"{v:.1f}" for v in d.values()) + " |" for n, d in fl.items()]
      q = fig_quality(plt, runs, seeds, out, cond)
      md += ["", "| median over successful trials | " + " | ".join(n for _, n in ARMS) + " |", "|---|" + "---|" * len(ARMS)]
      md += [f"| `{m}` | " + " | ".join(f"{v:.3f}" for v in vals.values()) + " |" for m, vals in q.items()]
      summary[cond].update(by_height=byh, outcomes=oc, failure_location=fl, quality=q)
      # Classifier timing is read where few trials fail early (tipping-only): under the strict rule a
      # trial that falls before the up-flight never reaches it, which would count as "never switched".
      if cond == ROUND["clf_cond"]:
        summary[cond]["classifier"] = fig_classifier(plt, runs, seeds, out, cond, out / "classifier_meta.json")
        md += ["", f"Classifier: {summary[cond]['classifier']}"]
    md.append("")
  c0, c1 = ROUND["delay"]
  if c0 in data and c1 in data:
    seeds = sorted(set(complete_seeds(data[c0], keys)) & set(complete_seeds(data[c1], keys)))
    if seeds:
      rows = fig_delay(plt, data[c0], data[c1], seeds, np.random.default_rng(0), out)
      md += [f"## Latency: no delay vs 20 ms ({ROUND['delay_label']})", "", "| arm | 0 ms | 20 ms |", "|---|---|---|"]
      md += [f"| {n} | {a:.1f} | {b:.1f} |" for n, a, b in rows] + [""]
  if "saro" in data and "training" in data:
    seeds = sorted(set(complete_seeds(data["saro"], keys)) & set(complete_seeds(data["training"], keys)))
    if seeds:
      rows = fig_definitions(plt, data["saro"], data["training"], seeds, out)
      md += ["## Fall definition: tipping only vs strict", "", "| arm | tipping only | strict |", "|---|---|---|"]
      md += [f"| {n} | {a:.1f} | {b:.1f} |" for n, a, b in rows] + [""]
  (out / "results_tables.md").write_text("\n".join(md))
  (out / "results_summary.json").write_text(json.dumps(summary, indent=1, default=float))
  print("\n".join(md))


if __name__ == "__main__":
  main()
