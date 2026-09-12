# Inbox: laptop -> cluster

What to implement/change next, decided from eval results. Insert new entries
directly below "## Open" (top of that section, most recent first) with a
dated header. Cluster session: when you pick one up, note it in
`coordination/log/<date>-cluster.md` rather than deleting it here, then move
it under "## Done" (with a one-line pointer to what changed) once implemented
and merged -- don't delete an entry you're not sure was seen.

<!-- Example entry:

## 2026-09-11 -- bump Gaps specialist's flat/rough blend if attempt 3 plateaus again
If job 11914 (20% stepping_stones / 40% flat / 40% random_rough) plateaus like
attempts 1-2 did, try 10% stepping_stones / 45% flat / 45% random_rough next --
see findings.md "Terrain specialists" table for the plateau signature to check for
(reward flat at -6 to -8, episodes ending in ~10-14 steps).

-->

## Open

## 2026-09-12 -- user decisions on both blockers, plus the four changes they imply

User has now decided both of the things that were pending across the three sessions.
Recording them here so the decision is durable and not just in chat scrollback.

**Decision 1 -- cluster git auth: repo-scoped SSH deploy key.** Not `gh` on the
cluster (plaintext token in `~/.config/gh/hosts.yml` on a shared filesystem), not
a PAT in the remote URL, not "user pushes manually each time". Concretely, on the
cluster:
```
ssh-keygen -t ed25519 -f ~/.ssh/policyswitching_deploy -N ""
```
then hand the user the **public** half to add at BRUH-MAIN/policyswitching ->
Settings -> Deploy keys, with **write access** ticked (it is off by default -- a
read-only deploy key fetches fine and fails on push, which would look exactly like
the bug we just spent a round diagnosing). Then
`git remote set-url origin git@github.com:BRUH-MAIN/policyswitching.git`, plus a
`Host github.com / Hostname ssh.github.com / Port 443` block in `~/.ssh/config` if
outbound 22 is blocked from the node.

**Decision 2 -- specialists back up to a NEW PRIVATE HF repo, not `go2-pas-saro`.**
Verified anonymously (no token): `RohanRamesh/go2-pas-saro` returns HTTP 200 with
`private: false`, `gated: false`, and its `stage1/model_*.pt` files are listed --
it is genuinely world-readable, so syncing specialists there would publish them.
The user does not want unpublished experimental checkpoints public. Create a
separate **private** model repo for the specialists (`create_repo(..., private=True)`)
and point `HF_CHECKPOINT_REPO` at it. **Leave `go2-pas-saro` exactly as it is** --
the user was offered flipping it private and declined, so do not change its
visibility.

The laptop's SSH-key route was *not* chosen, so `laptop_pull_and_eval.sh`'s rsync
branch stays broken by design. HF is now the only path specialists reach this
machine by. That makes the four items below load-bearing rather than cleanup.

### 1. One-off backfill upload for the three finished specialists (unblocks everything, no GPU)
`go2_spec_flat` / `go2_spec_rough` / `go2_spec_stairs` were trained without
`HF_TOKEN`, so they exist only on cluster local disk. `HfSyncVelocityOnPolicyRunner`
only fires on `runner.save()` inside a live training call, so as you already noted
this needs a small standalone script over `push_checkpoint_to_hf` in
`rl/hf_upload.py` -- not a retrain, and not a re-submission.

Worth doing first: it needs no GPU, so it runs on the login node **while every node
is still drained**. It is the one thing on either side's list that the drain doesn't
block. Upload each `model_9999.pt` to `<private-repo>/<experiment_name>/model_9999.pt`,
matching the layout `train_specialist_slurm.sh` already uses
(`HF_CHECKPOINT_STAGE="$EXPERIMENT_NAME"`), so a backfilled checkpoint and a
future synced one land at the same path.

### 2. Future runs: set `HF_TOKEN` at submission, with `HF_CHECKPOINT_REPO` -> the private repo
Otherwise 11914/11918/11919's outputs repeat this whole problem when they finally run.

### 3. `a100/eval_matrix.py`: specialists need the same HF fallback PAS just got
This is the same bug you found and fixed for PAS, still live for every entry in
`LOCAL_POLICIES`. They resolve only through `latest_local_checkpoint(mjlab_dir,
experiment)`; on the laptop that finds nothing and takes the `[SKIP]` branch.

The failure mode is worth being precise about, because it is not a crash: the skip
prints one line to stdout, then `summary.md` is rendered from `order`, which is
built from the result rows that actually exist. So a laptop-run matrix produces a
clean, plausible-looking summary table containing only the PAS rows, with no marker
in the file itself that five policies were never evaluated. Anyone reading
`summary.md` later -- including us -- would have to notice an absence rather than
an error. Give each `LOCAL_POLICIES` entry an `hf_stage` (= its experiment name) and
the same local-then-HF fallback as PAS.

### 4. `cluster_update_status.sh` / `cluster.json`: specialist entries move to `source: "hf"`
`laptop_pull_and_eval.sh` branches on `run.source`; with `source: "local"` it takes
the rsync path, which now cannot work. Once #1 lands, the three specialist entries
need `"source": "hf"` plus `"hf_repo"` and `"hf_stage"` (both read at lines 58-59),
and the script should write those fields for future runs rather than only
`path_on_cluster`. Fold in the previously-noted gap while you are in there: the
script never writes `task`, and `laptop_pull_and_eval.sh` hard-errors on
`task == MISSING` (line 67), so any run_id it creates fresh breaks the laptop side.

I own the objective.md 2x2 pass and the stale `docs/CLAUDE.cluster.md`; those are
still coming and are not blocked on any of the above.

## Done
