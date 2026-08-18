# Installation

## System requirements

| | Requirement |
|---|---|
| OS (training) | Linux + NVIDIA GPU (this repo's own guide recommends Ubuntu 22.04). macOS/Windows(WSL) work for CPU-only evaluation. |
| OS (eval only) | Linux, macOS, or Windows (WSL) |
| GPU driver | NVIDIA driver ≥ 550 recommended |
| CUDA | 12.4+ recommended — not all CUDA versions work with MuJoCo Warp |
| Python | ≥ 3.10 (this repo's guide uses 3.11) |

Platform gotchas (from mjlab's FAQ):
- **macOS**: CPU-only, no GPU acceleration — fine for trying things out, not for real training.
- **Windows**: preliminary support, may be unstable; Linux is the primary target.
- Setting `device="cpu"` in code does **not** stop mjlab from initializing a GPU context. To truly avoid claiming GPU memory, run with `CUDA_VISIBLE_DEVICES="" python scripts/train.py ...`.

## This repo's install path (conda + pip, editable)

This is what [`doc/setup_en.md`](../unitree_rl_mjlab/doc/setup_en.md) in the
repo documents, and matches `unitree_rl_mjlab/setup.py`
(`install_requires = ["mjlab==1.2.0", "mujoco-warp==3.5.0"]`).

```bash
# 1. Miniconda (skip if you already have conda)
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init --all
source ~/.bashrc

# 2. Environment
conda create -n unitree_rl_mjlab python=3.11
conda activate unitree_rl_mjlab

# 3. System deps needed by the deploy/ C++ stack (cyclonedds, unitree_sdk2, etc.)
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

# 4. Python deps, editable install so `src/` and `scripts/` are importable
cd unitree_rl_mjlab
pip install -e .
```

`pip install -e .` pulls in `mjlab==1.2.0` and `mujoco-warp==3.5.0` as
declared in `setup.py`. Because it's `-e` (editable) on package `src`, edits
to `src/tasks/...` take effect immediately without reinstalling.

Note the currently-open project (`policyswitching`) has **not** run this
install yet — `pip show mjlab` and `import mjlab` both fail in the ambient
Python. Run the steps above (in whatever env you intend to use) before trying
to run any of the scripts in this repo.

## mjlab's own install paths (useful if working outside this repo, e.g. to prototype a standalone task)

mjlab is normally consumed as a library via [`uv`](https://astral.sh/uv/),
mjlab's preferred tool, but pip/conda work too.

**As a dependency (uv, recommended by mjlab upstream):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init --package my_mjlab_project
cd my_mjlab_project
uv add mjlab            # or: uv add git+https://github.com/mujocolab/mjlab --editable /path/to/local/checkout
uv run demo              # sanity check
```

**Contributing to mjlab itself:**
```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv sync
uv run demo
```

**Classic pip/venv/conda:**
```bash
pip install mjlab
demo   # console-script entry point
```

**Docker:**
```bash
docker run --rm --runtime=nvidia --gpus all ghcr.io/mujocolab/mjlab uv run demo
# or build locally:
./scripts/run_docker.sh uv run demo
```

## Finding mjlab's actual source once installed

Since mjlab is a pip dependency, not vendored, the real source for anything
this doc summarizes (managers, `mdp` term implementations, actuator/sensor
classes, terrain generators) lives in site-packages after install:

```bash
python -c "import mjlab, os; print(os.path.dirname(mjlab.__file__))"
```

That's the fastest way to check exact signatures/defaults against the
current pinned version (`1.2.0`) rather than trusting docs prose, since the
docs site tracks `main` and may drift from a pinned release.
