# Data Directory Guide

This directory contains the model-ready datasets, preserved raw test sources,
and several historical dataset-preparation scripts.

The model-ready datasets are already prepared. Normal training and testing do
not require running any Python file in this directory.

## Directory Structure

```text
data/
|-- whitefly/                     # Model-ready whitefly branch
|   |-- train/
|   |-- val/
|   `-- test/
|-- fruit_fly/                    # Model-ready fruit-fly branch
|   |-- train/
|   |-- val/
|   `-- test/
|-- raw_test/                     # Preserved high-resolution test sources
|   |-- whitefly/
|   `-- fruit_fly/
|-- predict/                      # Version-controlled prediction examples
|-- A_Equal_division_of_5_datasets.py
|-- B_Produce_a_training_set_based_on_the_validation_set.py
|-- D_split-train-val-test-name-to-txt-1.py
|-- D_split-train-val-test-name-to-txt-2.py
|-- generate_mat.py
|-- Z_Check_the_contents_of_npy.py
`-- readme.md
```

## Current Dataset Directories

### `whitefly/`

The first MS-DM counting branch. Each split contains images and optional
same-stem NumPy point annotations:

```text
sample.jpg
sample.npy
```

The `.npy` array contains whitefly point coordinates in `(x, y)` order. An
image can exist without a whitefly `.npy` file when that image has no whitefly
annotation.

### `fruit_fly/`

The second MS-DM counting branch. It uses the same split names and file format
as `whitefly/`. File names must remain aligned across the two branches so one
image can be associated with both species.

### `raw_test/`

Preserved high-resolution test images and their original text annotations:

```text
raw_test/
|-- whitefly/
|   |-- images/
|   |-- annotations/
|   |-- train.txt
|   |-- val.txt
|   `-- test.txt
`-- fruit_fly/
    |-- images/
    |-- annotations/
    |-- train.txt
    |-- val.txt
    `-- test.txt
```

This directory is ignored by Git because it contains large image files. The
documented conversion pipeline is available at
[`../tools/preprocess/legacy_test_pipeline/README.md`](../tools/preprocess/legacy_test_pipeline/README.md).

### `predict/`

Small, version-controlled example images for demonstrating direct prediction on
unlabeled inputs. Unlike `raw_test/`, this directory is intentionally committed
to Git so users can run `predict.py` immediately after cloning the repository.

The included sample is:

```text
data/predict/IMG_20221230_135744.jpg
```

Run prediction on the bundled example:

```powershell
python predict.py
```

This is equivalent to:

```powershell
python predict.py --input-path data\predict
```

The generated files are written under `output/predict/`, which remains ignored
by Git.

## Normal Usage

### Training

`train.py` reads `whitefly/train`, `whitefly/val`, `fruit_fly/train`, and
`fruit_fly/val` automatically:

```powershell
python train.py
```

Explicit paths:

```powershell
python train.py `
  --whitefly-data-dir data\whitefly `
  --fruit-fly-data-dir data\fruit_fly
```

### Testing and prediction

`test.py` always reads only the two `test` splits; it never predicts files from
`train` or `val`:

```powershell
python test.py
```

Explicit paths:

```powershell
python test.py `
  --data-path data\whitefly `
  --fruit-fly-data-path data\fruit_fly
```

## Files in `data/`

The following Python files are historical preparation utilities. They are not
used by the current training or testing entry points.

### `A_Equal_division_of_5_datasets.py`

Purpose:

- Reads an image collection.
- Shuffles it with random seed `42`.
- Selects one of five folds as validation data.
- Copies each selected image and its two species annotation files into legacy
  staging directories.

Historical inputs:

```text
H:\Pest_monitoring_program\cross_validationForTrain\raw\img_split
H:\Pest_monitoring_program\cross_validationForTrain\raw\wf_split
H:\Pest_monitoring_program\cross_validationForTrain\raw\ff_split
```

Historical outputs:

```text
data/images/
data/mats/
data/mats-1/
```

Status: not directly runnable on the current project because the external `H:`
paths and legacy staging directories are unavailable. Update every path before
using it.

### `B_Produce_a_training_set_based_on_the_validation_set.py`

Purpose:

- Reads the historical full dataset.
- Skips images already copied as validation samples.
- Copies the remaining images and both species annotations into the legacy
  staging directories, thereby completing the training collection.

Historical input root:

```text
H:\Pest_monitoring_program\cross_validationForTrain\ed\
```

Historical outputs:

```text
data/images/
data/mats/
data/mats-1/
```

Status: not directly runnable until its external and output paths are updated.

### `D_split-train-val-test-name-to-txt-1.py`

Purpose:

- Scans `data/images/` in the historical staging layout.
- Writes image stems to `data/train.txt`.
- Its current condition assigns every discovered image to the training list;
  it does not create a meaningful random train/test split.

Important: the script opens its output files in write mode and overwrites
existing lists. Do not run it against the current dataset.

### `D_split-train-val-test-name-to-txt-2.py`

Purpose:

- Scans the historical `data/temp/` validation-image directory.
- Writes every discovered image stem to `data/val.txt`.

Important: `data/temp/` no longer exists, and the script overwrites
`data/val.txt`.

### `generate_mat.py`

Purpose:

- Reads space-separated `(x, y)` coordinates from text files.
- Converts them to the MATLAB `image_info` structure expected by the old
  NWPU-style preprocessing code.
- Writes `.mat` files back into the same historical `data/mats/` directory.

Status: this root-level version handles only the historical first annotation
directory and is not recommended for current test data. Use the maintained
two-species version instead:

```powershell
python tools\preprocess\legacy_test_pipeline\generate_mat.py
```

### `Z_Check_the_contents_of_npy.py`

Purpose: loads one `.npy` annotation and prints its point-coordinate array.

Usage:

1. Edit `file_path` inside the script.
2. Run:

```powershell
python data\Z_Check_the_contents_of_npy.py
```

The default sample path is `data/whitefly/train/0002.npy`.

## Historical Training-Data Preparation Order

The old scripts were intended to run in this order:

```text
1. A_Equal_division_of_5_datasets.py
      Select one validation fold and copy its files.
                         |
                         v
2. B_Produce_a_training_set_based_on_the_validation_set.py
      Add the remaining samples as training data.
                         |
                         v
3. D_split-train-val-test-name-to-txt-1.py
   D_split-train-val-test-name-to-txt-2.py
      Generate legacy train and validation name lists.
                         |
                         v
4. generate_mat.py
      Convert text point annotations to MATLAB format.
                         |
                         v
5. tools/preprocess/preprocess_dataset_nwpu.py
   tools/preprocess/preprocess_dataset_nwpu_another.py
      Produce model-ready .jpg + .npy datasets.
```

This sequence documents the historical intent only. It cannot be reproduced
unchanged because its external source paths and intermediate directories are
missing. Running the list-generation scripts can also overwrite existing split
files.

## Recommended Preparation Order for Preserved Test Data

For the sources under `data/raw_test/`, use the maintained pipeline:

```powershell
# 1. Convert text point annotations to MATLAB files.
python tools\preprocess\legacy_test_pipeline\generate_mat.py

# 2. Generate the model-ready whitefly branch.
python tools\preprocess\legacy_test_pipeline\preprocess_dataset_nwpu-test-04.py

# 3. Generate the model-ready fruit-fly branch and align images.
python tools\preprocess\legacy_test_pipeline\preprocess_dataset_nwpu-test-05.py

# 4. Run prediction on the prepared test split.
python test.py
```

Grid splitting is optional and is not part of the current full-image test
workflow. See the legacy test pipeline README before using the tile script.
