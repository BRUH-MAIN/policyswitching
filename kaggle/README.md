# Kaggle training

`train_pas.ipynb` trains SARO's PAS policy (see
[`../docs/07-pas-implementation.md`](../docs/07-pas-implementation.md)) for
the Go2 on Kaggle's 2x T4 GPUs, with checkpoints synced to a Hugging Face
model repo for persistence across Kaggle's ephemeral sessions.

## Setup (one time)

1. Upload `train_pas.ipynb` to a new Kaggle notebook (File > Import Notebook).
2. Notebook settings (right sidebar): **Accelerator: GPU T4 x2**, **Internet: On**.
3. Add-ons > Secrets: add `HF_TOKEN` (a Hugging Face token with write access —
   huggingface.co/settings/tokens).
4. If `BRUH-MAIN/policyswitching` is private, also add a `GITHUB_TOKEN` secret
   (a GitHub PAT with `repo` scope).

## Running

Edit the CONFIG cell at the top (env count, GPU count, iteration budget) and
run all cells. Re-running the notebook after a session ends/gets killed
resumes automatically — both training stages check the Hugging Face repo for
an existing checkpoint before starting.

See the notebook's own markdown cells for what each step does, and
`docs/07-pas-implementation.md` for the two-stage training pipeline itself.
