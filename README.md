# MLIII Group 3: Facial Age and Gender Estimation

Howdy! This is our DS 6050 deep learning project. We compare age-only, gender-only, and joint VGG16 models, then test whether rotation, random erasing, or both improve their predictions.

I put these instructions together so our professor, TAs, and teammates can review the results and run the code on their own machines without needing access to our personal HPC directories.

The notebook calls the classification task `sex_only`, and its columns use `sex`. These names refer to the binary gender label supplied by IMDB-WIKI. Age predictions estimate chronological age in years.

## Start here

There are three ways to use this repository:

| What you want to do | What you need |
| --- | --- |
| Review our saved results | The CSV files in `results/`. No dataset or GPU needed. |
| Check the notebook setup and load an example | A Python environment, the extracted IMDB images, and the committed split files. |
| Train and evaluate models | The same setup, enough storage for checkpoints, and preferably an NVIDIA GPU. |

**The notebook is still a working project file, so please do not start with Run All.** It contains paths from our HPC runs, a combined-experiment evaluation flag currently set to `True`, and cells that recover a particular saved session. The steps below explain what to change before running anything.

## 1. Clone the repository

```bash
git clone https://github.com/SavannaCabrera8796/ML3_Group3.git
cd ML3_Group3
```

Keep the notebook's working directory at this repository root. The code uses `Path.cwd()` to find the notebook and shared split files.

| Location | What is inside |
| --- | --- |
| `MLIII_Group3_Project_Code.ipynb` | Shared setup, model definitions, training, evaluation, and augmentation experiments |
| `requirements.txt` | Pinned dependencies recorded for the baseline environment, including CUDA-specific PyTorch builds |
| `data/cohort/` | Fixed train/validation/test manifests, cohort protocol, and source hashes |
| `results/baselines/` | No-augmentation test results, summaries, paired contrasts, and protocol |
| `results/rotation_only/20261029_113715_204868/` | Exported rotation-only results and protocol |

The repository does **not** include the dataset images, source archives, trained checkpoints, or full training histories and prediction files. Paths in saved run registries refer to the original machine; cloning the repository does not download those runs.

## 2. Create your own Python environment

Our baseline documentation records Python 3.12.14. The commands below use that version as a starting point. Please create your own environment rather than trying to use one of ours.

```bash
conda create -n mliii-group3 python=3.12.14 -y
conda activate mliii-group3
python -m pip install --upgrade pip
```

### NVIDIA GPU setup matching the baseline package pins

For a machine with a compatible NVIDIA driver:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

### CPU setup

For CPU review and small setup checks, install the CPU builds instead:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining pinned packages without the CUDA-specific PyTorch lines:

```bash
python -c "from pathlib import Path; import subprocess, sys; packages = [line.strip() for line in Path('requirements.txt').read_text().splitlines() if line.strip() and not line.lstrip().startswith('#') and not line.startswith(('torch==', 'torchvision=='))]; subprocess.check_call([sys.executable, '-m', 'pip', 'install', *packages])"
```

Do not run the unmodified `requirements.txt` afterward on the CPU environment; it requests the `+cu126` builds.

The CPU wheel command is intended for platforms supported by that wheel index. On Apple Silicon or another unsupported platform, use the [official PyTorch installer](https://pytorch.org/get-started/locally/) for a compatible torch/torchvision pair and record any version changes.

The notebook currently selects CUDA when available and otherwise CPU; it does not automatically select Apple MPS.

### Open the notebook

```bash
python -m pip install jupyterlab
python -m ipykernel install --user --name mliii-group3 --display-name "MLIII Group 3"
python -m jupyterlab
```

Open `MLIII_Group3_Project_Code.ipynb` and select **MLIII Group 3** as the kernel. VS Code users can select the same environment instead.

Check the kernel before continuing:

```python
import sys
import torch
import torchvision

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
```

`False` is expected for a CPU installation. Full VGG16 training on this dataset is much more practical on a GPU. We have not established a minimum GPU-memory requirement or a portable runtime estimate.

**Environment note:** The latest saved notebook output reports Python 3.13.12 and several package versions that differ from `requirements.txt`. Treat each original run's `environment.json` as the record of that run, rather than assuming the notebook's displayed environment applies to every experiment. The installation commands here have not been validated in a fresh environment on every platform.

## 3. Put the dataset on your machine

We use the **cropped IMDB component** of [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/), not the Wikipedia component. Obtain the images from the dataset source and follow its access and usage terms.

Your dataset directory should contain an `imdb_crop/` folder with the original image subfolders and files inside it.

Keep the image paths beneath `imdb_crop/` unchanged so they match the committed manifests. Loading the existing cohort needs the extracted images and `data/cohort/`; it does not require `imdb.mat` or the original archives.

The optional extraction/preparation workflow additionally expects `imdb_crop.tar` and `imdb_meta.tar` directly inside your dataset folder. Metadata extracts to `imdb/imdb.mat`. Archive extraction requires a Python version providing `tarfile.data_filter`.

### Fix the dataset path in the settings cell

The notebook currently contains both an environment-variable setting and a later hard-coded assignment:

```python
DATA_DIR = Path("/scratch/azd9vv/imdb_faces_data").resolve()
```

**Remove or comment out that hard-coded line.** Otherwise it overrides `MLIII_DATA_DIR`.

The beginning of the path configuration should look like this:

```python
PROJECT_DIR = Path.cwd().resolve()
DATA_DIR = Path(os.environ.get("MLIII_DATA_DIR", str(PROJECT_DIR))).expanduser().resolve()
WORK_DIR = DATA_DIR
IMAGE_ROOT = DATA_DIR / "imdb_crop"
```

Set `MLIII_DATA_DIR` before starting Jupyter, using your own path:

```bash
# Linux/macOS
export MLIII_DATA_DIR="/absolute/path/to/your/dataset"
python -m jupyterlab
```

```powershell
# Windows PowerShell
$env:MLIII_DATA_DIR = "C:\path\to\your\dataset"
python -m jupyterlab
```

Alternatively, replace the hard-coded assignment with your own absolute `Path(...)` in the notebook. In either case, `DATA_DIR` is the parent of `imdb_crop`, not `imdb_crop` itself.

The dataset directory must be writable because some export cells use `WORK_DIR / 'results'`. Full experiment runs are stored separately beneath the repository's `studies/` directory.

## 4. Check setup before training

Before running cells, set **all eight execution flags** to `False` at their definitions:

```python
RUN_EXPERIMENTS = False
EVALUATE_TEST = False
ROTATION_RUN_EXPERIMENTS = False
ROTATION_EVALUATE_TEST = False
RANDOM_ERASING_RUN_EXPERIMENTS = False
RANDOM_ERASING_EVALUATE_TEST = False
ROTATION_ERASING_RUN_EXPERIMENTS = False
ROTATION_ERASING_EVALUATE_TEST = False
```

Also leave these preparation flags off when reusing our cohort:

```python
RUN_EXTRACTION_RECOVERY = False
PREPARE_ARCHIVES = False
BUILD_COHORT_IF_MISSING = False
```

For a first local check, use `NUM_WORKERS = 0`. This helps avoid multiprocessing problems in Windows/macOS notebook kernels. Keep the declared batch size, epochs, seeds, and loss weights unchanged for comparisons intended to reproduce the study.

Restart the kernel, then run the cells from the top through the shared dataset, loader, model, and `epoch_pass` definitions. Stop before the training launch cells.

The cohort-loading cell automatically copies the approved files from `data/cohort/` into `studies/vgg16_common_cohort_v1/` when needed. You do not need to copy them manually.

Expected cohort output:

| Split | Images |
| --- | ---: |
| Train | 135,964 |
| Validation | 14,741 |
| Test | 14,095 |
| Total | 164,800 |

The training-only age mean should be about **36.2494**, with standard deviation **13.0466**. The loader verifies manifest checksums, person/hash disjointness, and age statistics. Do not regenerate splits or edit a manifest to get past an assertion.

After the shared definitions run, try one training image without starting an optimizer:

```python
image, age, sex = FaceDataset(frames["train"])[0]
print("Image shape:", tuple(image.shape))  # (3, 224, 224)
print("Standardized age:", age.item())
print("Recorded binary label:", sex.item())
```

This checks one image, not the entire dataset. A successful cohort load alone does not verify that every local image file is present or matches its recorded hash.

## 5. Review results without retraining

You can read the committed CSVs directly. From the repository root:

```python
from pathlib import Path
import pandas as pd

baseline = Path("results/baselines")
print(pd.read_csv(baseline / "test_summary.csv").to_string(index=False))
print(pd.read_csv(baseline / "test_results_per_seed.csv").to_string(index=False))
print(pd.read_csv(baseline / "test_paired_contrasts.csv").to_string(index=False))

rotation = Path("results/rotation_only/20261029_113715_204868")
print(pd.read_csv(rotation / "test_summary.csv").to_string(index=False))
```

- `test_results_per_seed.csv` contains individual model/seed results.
- `test_summary.csv` reports means and sample standard deviations across three seeds.
- Lower age MAE is better. Higher classification accuracy is better.
- The existing baseline contrast file expresses **improvement from joint learning**: age uses Single − Joint, while accuracy uses Joint − Single. Negative values mean joint performed worse. For the professor's requested **Joint − Single** table, negate the age-improvement column and keep the accuracy column unchanged.
- Rotation comparison files use **Rotation − None** within matching model/seed pairs. They do not directly compare joint versus single-task models.

Saved notebook outputs are a record of earlier executions. They are not evidence that your current kernel has run those cells successfully.

## 6. Run new experiments intentionally

Our fixed design uses three tasks (`age_only`, `sex_only`, `joint`), seeds 101/202/303, and four augmentation conditions: 36 training runs altogether. Each condition launches nine runs, sequentially.

The common recipe uses ImageNet VGG16, frozen blocks 1–4, trainable block 5 and classifier/heads, AdamW, batch size 16, and 10 epochs per network. Age uses standardized MAE, classification uses BCE with logits, and joint training adds them with weights of 1.

The classification threshold is fixed at 0.5. Checkpoints minimize validation loss, with one joint checkpoint for both tasks.

### No augmentation

1. Complete the shared setup above and save the notebook.
2. Set `RUN_EXPERIMENTS = True` in the settings cell. Leave other conditions disabled.
3. Run setup and definitions in order, initialize the baseline run registry, and run the baseline training launch cell.
4. After all nine runs complete, set `EVALUATE_TEST = True` and execute the baseline evaluation and summary cells in the same session.
5. Keep the resulting session folder. Return execution flags to `False` when finished.

Training initialization may download the ImageNet VGG16 weights if they are not already cached. Evaluation of a complete saved checkpoint uses `pretrained=False`.

### Rotation only

After shared setup, run the rotation transform, settings, tracking, snapshot/checking, and training definitions in order. Set `ROTATION_RUN_EXPERIMENTS = True` in the rotation settings cell, launch its nine runs, then enable `ROTATION_EVALUATE_TEST` for evaluation and summaries.

Rotation affects training inputs only, within ±10 degrees. The rotation-versus-baseline summary expects a baseline results file at:

```python
WORK_DIR / "results" / "baselines" / "test_results_per_seed.csv"
```

For comparison against our committed original baselines, copy that file from `PROJECT_DIR / 'results/baselines/'` to the expected location.

For a comparison against your own newly trained baselines, copy your new session's file instead and label that comparison accordingly. The notebook does not automatically stage the fresh baseline results there.

### Random erasing only and rotation plus erasing

The current implementation erases one randomly chosen quadrant, covering 25% of the input, with probability 0.5. Erasing happens after normalization and fills with zero, corresponding to the ImageNet channel means.

Combined augmentation applies rotation first. Validation and test preprocessing stay deterministic.

These sections need a little more handoff work before they support a full fresh-machine run:

- **Erasing only:** Training code is present, but the summary references `random_erasing_test_results` without defining it. This notebook version does not include the corresponding test-evaluation function. Michael needs to provide the completed evaluation code/results before we can reproduce that section end to end.
- **Combined:** The setup creates a new session, but later cells overwrite it with `20261111_011641_580896` and attempt to recover nine original checkpoints. For a fresh training run, skip the two historical-session override/recovery cells and retain the new session's directory, protocol paths, and registry through training and evaluation. For review of the original run, obtain its full matching artifacts and reconnect all those paths consistently. Do not mix a new protocol with an old session.
- Some copied comments and older markdown describe rotation or an earlier small-patch erasing proposal. The implemented quadrant transform and condition-specific run configuration describe the actual current setting.

Michael ran the erasing-only and combined experiments. Their exported result files are not yet included in this repository snapshot, although the notebook contains saved output from the combined run.

Please do not interpret missing exports as zero results or launch replacement experiments just to fill in a table.

## Where new outputs go

| Condition | Full local experiment location |
| --- | --- |
| None | `studies/vgg16_common_cohort_v1/baseline_reruns/<session>/` |
| Rotation | `studies/vgg16_common_cohort_v1/rotation_only/<session>/` |
| Erasing | `studies/vgg16_common_cohort_v1/random_erasing_only/<session>/` |
| Both | `studies/vgg16_common_cohort_v1/random_erasing_and_rotation/<session>/` |

Runs save checkpoints, histories, settings, code snapshots, and environment records. Successful evaluation saves per-image predictions and test metrics.

Augmentation export cells copy selected small files under `WORK_DIR / 'results'`; that may be outside the repository. Baseline evaluation writes to its session directory.

Rerunning a settings cell creates a new timestamped directory and can reset its registry. It does not resume a previous run. The training loop saves best-model weights but does not save optimizer state for interrupted-training resumption.

Keep the selected run folders and their matching protocols together.

## If something goes wrong

| Problem | What to check |
| --- | --- |
| Images not found | Remove the hard-coded HPC `DATA_DIR` override. Confirm `DATA_DIR / 'imdb_crop'` exists and preserves the original subfolders. |
| Approved splits missing | Start Jupyter in the repository root and confirm all three CSVs and `protocol.json` are in `data/cohort/`. |
| Partial or mismatched cohort | Restore the complete matching manifest set. Do not bypass checks or generate new splits for an augmentation comparison. |
| Wrong packages in the notebook | Check `sys.executable` and select the environment you installed. |
| CUDA unavailable | Confirm you installed the CUDA build and have a compatible NVIDIA driver/GPU. CPU review can still work. |
| DataLoader hangs or worker errors | Start with `NUM_WORKERS = 0` and restart the kernel. |
| GPU runs out of memory | Close other GPU work and release old model objects. A smaller batch size is a separate configuration, not an unchanged reproduction of our results. |
| Missing combined checkpoints | Skip the historical recovery cells for a fresh run, or obtain the original full session artifacts. They are not in the clone. |
| `random_erasing_test_results` undefined | The erasing-only evaluation handoff is incomplete in this notebook version. Obtain the missing evaluation/results from Michael. |
| Missing baseline CSV during augmentation comparison | Stage the intended baseline file under `WORK_DIR / 'results/baselines/'`. |
| A protocol/test-lock assertion fails | Keep the original records and investigate the mismatch. Use a new session for a revised experiment rather than overwriting locked evidence. |

Please select checkpoints using validation data and report all declared seeds. Test results should not guide tuning.

Three-seed SD describes training variability on our fixed split, not uncertainty across different datasets. Different hardware or software may also change the exact numbers.

## Team and AI assistance

Savanna Cabrera, Amanda Coffey, Sienna Lee, Manuel Sanchez, and Michael Walker.

I used OpenAI's ChatGPT (Codex) to help plan the code structure and experimental workflow, explain PyTorch concepts, and draft documentation. I review and adapt that assistance and take responsibility for the work I use. Each team member remains responsible for their own contributions and for checking the final results and interpretation.
