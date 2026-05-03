# IsaacLab Data Generation Workspace

This folder is a standalone data-generation workspace. It contains the copied
IsaacLab task registration/config code needed for door/drawer demo collection
and MimicGen HDF5 generation.

Only IsaacLab demo collection and MimicGen HDF5 generation live here.

## Folder layout

```text
isaaclab_data_generation/
  config.env
  data/
    source/<spec>/    # raw teleop HDF5 demos
    mimic/<spec>/     # merged, annotated, and MimicGen HDF5 demos
  isaac_lab_tasks/    # copied task/env/spec code needed by IsaacLab
  logs/
  scripts/
  tools/
```

## Supported specs

```text
open_door_normal
open_door_sphere
open_drawer_normal
open_drawer_sphere
close_door_normal
close_door_sphere
close_drawer_normal
close_drawer_sphere
```

## Environment

Inside the Docker image, the expected IsaacLab path is:

```text
ISAACLAB_ROOT=/opt/IsaacLab
```

If a Python environment is already active, the scripts use it automatically.

Outside Docker, the scripts fall back to:

```text
ISAACLAB_ROOT=/home/ntruong/Truong/IsaacLab
```

Edit `config.env` if your IsaacLab or Python paths differ. You do not need to set
any external project-repo path.

## Check status

```bash
cd /home/ntruong/Truong/isaaclab_data_generation
bash scripts/status.sh
```

## Collect raw demos

Collect one spec:

```bash
cd /home/ntruong/Truong/isaaclab_data_generation
bash scripts/collect_raw.sh open_drawer_normal 10
```

For sphere specs, the script automatically uses the registered bucket preset:

```bash
bash scripts/collect_raw.sh open_drawer_sphere 10
```

Raw demos are written to:

```text
data/source/<spec>/*.hdf5
```

To keep separate top/bottom drawer collections but create one mixed source
folder for MimicGen, copy them into a new interleaved folder:

```bash
# 25 top + 25 bottom -> 50 mixed raw demos:
bash scripts/interleave_raw_demos.sh open_drawer_sphere
```

The source folders stay unchanged. The output order is top, bottom, top,
bottom, and a `manifest.tsv` records where every copied demo came from. By
default it reads `data/source/<spec>_top` and `data/source/<spec>_bottom`,
then writes `data/source/<spec>_top_bottom`.

Show a viewport debug line from the end-effector to the nearest door/drawer
handle:

```bash
DRAW_EE_TARGET_LINE=1 bash scripts/collect_raw.sh open_drawer_sphere 10
```

Show X/Y/Z component lengths from the end-effector origin:

```bash
DRAW_EE_TARGET_LINE=1 EE_TARGET_VIS_MODE=axes bash scripts/collect_raw.sh open_drawer_sphere 10
```

## Run MimicGen

```bash
cd /home/ntruong/Truong/isaaclab_data_generation
HEADLESS=1 NUM_ENVS=1 bash scripts/mimic_generate.sh open_drawer_normal 1000
```

Mimic outputs are written to:

```text
data/mimic/<spec>/
```

For example:

```text
data/mimic/open_drawer_normal/open_drawer_demos_merged.hdf5
data/mimic/open_drawer_normal/open_drawer_demos_annotated.hdf5
data/mimic/open_drawer_normal/open_drawer_demos_mimic.hdf5
```

## Budget MimicGen variants

Use this when you have one source folder with at least 50 raw demos and want
to compare 10, 20, 30, 40, and 50 source-demo budgets. The `SETUP_NAME`
controls the output folder names under `data/mimic/`.

```bash
cd /home/ntruong/Truong/isaaclab_data_generation

HEADLESS=1 NUM_ENVS=1 bash scripts/mimic_budget_comparison.sh open_drawer_sphere
```

To select a specific IsaacLab CUDA device, prefer `ISAAC_DEVICE` instead of
`CUDA_VISIBLE_DEVICES`:

```bash
ISAAC_DEVICE=cuda:1 HEADLESS=1 NUM_ENVS=1 bash scripts/mimic_budget_comparison.sh open_drawer_sphere
```

For drawer MimicGen, the default grasp detector expects the Panda fingers to
cross below `-0.01` while the EE is within `0.10m` of the selected handle. These
values were chosen from the 50-demo top/bottom source set. Avoid setting
`OPEN_DRAWER_MIMIC_GRIPPER_THRESHOLD` to a positive value, because that can make
the `grasp` signal true from the first frame and MimicGen cannot find a subtask
transition.

You can re-check these thresholds on any raw drawer folder:

```bash
python scripts/analyze_drawer_grasp_thresholds.py data/source/open_drawer_sphere_top_bottom
```

A good threshold must produce `grasp=false` at the first frame, then a later
`false -> true` transition for every source demo. For the current 50-demo
top/bottom set, `finger < -0.01` gives 50/50 transitions and the largest EE
distance at that first crossing is about `0.0896m`, so the default `0.10m`
distance threshold leaves margin.

This creates:

```text
data/mimic/open_drawer_sphere_top_bottom_10/
data/mimic/open_drawer_sphere_top_bottom_20/
data/mimic/open_drawer_sphere_top_bottom_30/
data/mimic/open_drawer_sphere_top_bottom_40/
data/mimic/open_drawer_sphere_top_bottom_50/
```

Each run uses the first N demos from the raw folder. Temporary source subsets
are symlinked under `data/source_subsets/<SETUP_NAME>_<N>/`, so the original
raw HDF5 files are still untouched.

For normal drawer demos, use the same flow with `open_drawer_normal` and a
normal mixed source folder:

```bash
bash scripts/interleave_raw_demos.sh open_drawer_normal
HEADLESS=1 NUM_ENVS=1 bash scripts/mimic_budget_comparison.sh open_drawer_normal
```

## Run a group

Collect all configured specs:

```bash
bash scripts/run_all.sh collect
```

Run MimicGen for all configured specs:

```bash
HEADLESS=1 bash scripts/run_all.sh mimic
```

Run one subset:

```bash
bash scripts/run_all.sh collect open_door_normal open_drawer_normal
HEADLESS=1 bash scripts/run_all.sh mimic open_door_normal open_drawer_normal
```

## Notes

- Use the UI/NoMachine session for keyboard demo collection.
- Use `HEADLESS=1` for MimicGen after raw demos are collected.
- Set `OVERWRITE=1` to regenerate MimicGen outputs for a spec.
- This workspace writes only HDF5 data for IsaacLab/MimicGen.
