<<<<<<< Updated upstream
# Image Restoration Utilities

Tools and utilities for image restoration with support for composite degradations including low-light, haze, rain, and snow.
=======
# OneRestore Custom Tools

Tools and utilities for image restoration using the OneRestore framework with support for composite degradations including low-light, haze, rain, and snow.
>>>>>>> Stashed changes

## Overview

This collection provides utility scripts for dataset management, model training, inference, and deployment:

- **app.py** - Interactive Gradio web interface for real-time image restoration
- **fix_dataset.py** - Dataset alignment and train/test splitting with synchronized file management
- **makedataset.py** - Converts image patches into HDF5 format for efficient training
- **remove_optim.py** - Removes optimizer weights from model checkpoints
- **test.py** - Batch inference script with automatic or manual degradation detection
- **push_and_reload_from_hf.py** - Hugging Face Hub model management

## Requirements

- Python 3.7+
- PyTorch 1.13+
- CUDA 11.7 (recommended, CPU supported)
- Gradio
- H5py

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install gradio gradio-imageslider
```

## Tools Documentation

### 1. app.py - Interactive Web Interface

Gradio-based web interface for real-time image restoration with automatic or manual degradation type selection.

**Features:**
<<<<<<< Updated upstream
- Load images and process them through restoration pipeline
=======
- Load images and process them through OneRestore restoration pipeline
>>>>>>> Stashed changes
- Automatic degradation detection using visual embedder
- Manual degradation selection from dropdown
- Before/after image comparison slider
- Support for multiple degradation types: low, haze, rain, snow, and combinations

**Supported Degradations:**
- Single: low, haze, rain, snow
- Composite: low_haze, low_rain, low_snow, haze_rain, haze_snow, low_haze_rain, low_haze_snow

**Usage:**

```bash
python app.py
```

Launches Gradio interface at http://127.0.0.1:7860

**Model Paths:**
- Embedder: `./ckpts/embedder_model.tar`
<<<<<<< Updated upstream
- Restorer: `./ckpts/restorer_model.tar`
=======
- Restorer: `./ckpts/onerestore_cdd-11.tar`
>>>>>>> Stashed changes

### 2. fix_dataset.py - Dataset Alignment & Splitting

Ensures synchronized file management across all degradation classes and performs 80/20 train/test split.

**Functionality:**
- Re-merges test data back to training set for proper alignment
- Performs synchronized split with identical filenames across all 13 degradation classes
- Prevents file misalignment between clear, degraded, and composite images

**Supported Degradation Classes:**
```
clear, low, haze, blur, noise, 
low_haze, low_blur, low_noise, haze_blur, 
haze_noise, low_haze_blur, low_haze_noise, low_haze_blur_noise
```

**Usage:**

```bash
python fix_dataset.py
```

**Expected Directory Structure:**
```
./image/
├── CDD-11_train/
│   ├── clear/
│   ├── low/
│   ├── haze/
│   └── ... (all 13 classes)
└── CDD-11_test/
    └── (created after split)
```

### 3. makedataset.py - HDF5 Dataset Generation

Converts image patches into HDF5 format for efficient training data loading.

**Features:**
- Extracts overlapping patches from paired images (clean + degradations)
- Performs data augmentation (rotations, flips)
- Generates single HDF5 file containing all training patches
- Supports configurable patch size and stride

**Dataset Structure:**
- Reads from: ground truth (clear) + degradation images
- Creates stacked patches: [gt, low, haze, blur, noise, low_haze, low_blur, low_noise, haze_blur, haze_noise, low_haze_blur, low_haze_noise, low_haze_blur_noise]
- Each patch: shape (13, channels, height, width)

**Usage:**

```bash
python makedataset.py \
  --train-path ./image/CDD-11_train \
  --data-name dataset.h5 \
  --patch-size 256 \
  --stride 200
```

**Arguments:**
- `--train-path`: Path to training dataset directory
- `--data-name`: Output HDF5 filename (default: dataset.h5)
- `--patch-size`: Patch size in pixels (default: 256)
- `--stride`: Stride for patch extraction (default: 200)
- `--gt-name`: Ground truth folder name (default: clear)

**Output:**
- HDF5 file with sequential patch datasets (key format: "0", "1", "2", ...)

### 4. remove_optim.py - Model Checkpoint Cleanup

Removes optimizer states from model checkpoints to reduce file size and enable inference/deployment.

**Functionality:**
- Extracts state_dict from checkpoint files
- Removes optimizer weight data
- Handles multi-GPU trained models (strips 'module.' prefix)
- Saves clean, lightweight checkpoints

**Supported Models:**
<<<<<<< Updated upstream
- Restorer: Full restoration model
=======
- OneRestore: Full restoration model
>>>>>>> Stashed changes
- Embedder: Text/visual embedder with 13 degradation classes

**Usage:**

<<<<<<< Updated upstream
For Restorer:
```bash
python remove_optim.py \
  --type Restorer \
  --input-file ./ckpts/restorer_model.tar \
  --output-file ./ckpts/restorer_model.tar
=======
For OneRestore:
```bash
python remove_optim.py \
  --type OneRestore \
  --input-file ./ckpts/onerestore_model.tar \
  --output-file ./ckpts/onerestore_cdd-11.tar
>>>>>>> Stashed changes
```

For Embedder:
```bash
python remove_optim.py \
  --type Embedder \
  --input-file ./ckpts/embedder_model.tar \
  --output-file ./ckpts/embedder_model.tar
```

**Arguments:**
<<<<<<< Updated upstream
- `--type`: Model type (Restorer or Embedder)
=======
- `--type`: Model type (OneRestore or Embedder)
>>>>>>> Stashed changes
- `--input-file`: Path to checkpoint with optimizer
- `--output-file`: Path to save optimized checkpoint

### 5. test.py - Batch Inference

Command-line inference script for processing images with automatic or manual degradation detection.

**Features:**
- Process single images or batches
- Automatic degradation detection using visual embedder
- Manual degradation type override with text embedding
- Saves restored images to output directory
- Records processing times

**Usage:**

Automatic degradation detection:
```bash
python test.py \
  --embedder-model-path ./ckpts/embedder_model.tar \
<<<<<<< Updated upstream
  --restore-model-path ./ckpts/restorer_model.tar \
=======
  --restore-model-path ./ckpts/onerestore_cdd-11.tar \
>>>>>>> Stashed changes
  --input ./image/ \
  --output ./output/
```

Manual degradation specification:
```bash
python test.py \
  --embedder-model-path ./ckpts/embedder_model.tar \
<<<<<<< Updated upstream
  --restore-model-path ./ckpts/restorer_model.tar \
=======
  --restore-model-path ./ckpts/onerestore_cdd-11.tar \
>>>>>>> Stashed changes
  --prompt low_haze \
  --input ./image/ \
  --output ./output/
```

**Arguments:**
- `--embedder-model-path`: Path to embedder model
- `--restore-model-path`: Path to restorer model
- `--input`: Input image directory or file
- `--output`: Output directory for restored images
- `--prompt`: (Optional) Degradation type override

### 6. push_and_reload_from_hf.py - Hugging Face Integration

Manages model distribution and versioning through Hugging Face Hub.

**Functionality:**
- Downloads pre-trained models from Hugging Face
- Uploads models to Hub repositories
- Loads and validates models after upload
<<<<<<< Updated upstream
- Supports both Embedder and Restorer models
=======
- Supports both Embedder and OneRestore models
>>>>>>> Stashed changes

**Usage:**

```bash
python push_and_reload_from_hf.py
```

Performs:
1. Download embedder_model.tar from Hugging Face
2. Upload to `gy65896/embedder` repository
<<<<<<< Updated upstream
3. Download restorer_model.tar from Hugging Face
=======
3. Download onerestore_cdd-11.tar from Hugging Face
>>>>>>> Stashed changes
4. Upload to `gy65896/restorer` repository
5. Reload both models for verification

## Typical Workflow

### Dataset Preparation & Training

```bash
# 1. Fix and split dataset into synchronized train/test sets
python fix_dataset.py

# 2. Generate HDF5 dataset for training
python makedataset.py --train-path ./image/CDD-11_train --data-name dataset.h5 --patch-size 256 --stride 200

<<<<<<< Updated upstream
# 3. Train embedder (using train_Embedder.py)
=======
# 3. Train embedder (using train_Embedder.py from original OneRestore)
>>>>>>> Stashed changes
python train_Embedder.py --train-dir ./image/CDD-11_train --test-dir ./image/CDD-11_test --check-dir ./ckpts --epoch 200

# 4. Clean embedder checkpoint
python remove_optim.py --type Embedder --input-file ./ckpts/embedder_model.tar --output-file ./ckpts/embedder_model.tar

<<<<<<< Updated upstream
# 5. Train Restorer (using train_restoration_single-gpu.py or train_restoration_multi-gpu.py)
python train_restoration_single-gpu.py --embedder-model-path ./ckpts/embedder_model.tar --train-input ./dataset.h5 --test-input ./image/CDD-11_test --epoch 120

# 6. Clean restoration model checkpoint
python remove_optim.py --type Restorer --input-file ./ckpts/restorer_model.tar --output-file ./ckpts/restorer_model.tar
=======
# 5. Train OneRestore (using train_OneRestore_single-gpu.py or train_OneRestore_multi-gpu.py)
python train_OneRestore_single-gpu.py --embedder-model-path ./ckpts/embedder_model.tar --train-input ./dataset.h5 --test-input ./image/CDD-11_test --epoch 120

# 6. Clean restoration model checkpoint
python remove_optim.py --type OneRestore --input-file ./ckpts/onerestore_model.tar --output-file ./ckpts/onerestore_cdd-11.tar
>>>>>>> Stashed changes
```

### Inference & Deployment

```bash
# Batch inference with test.py
<<<<<<< Updated upstream
python test.py --embedder-model-path ./ckpts/embedder_model.tar --restore-model-path ./ckpts/restorer_model.tar --input ./image/ --output ./output/
=======
python test.py --embedder-model-path ./ckpts/embedder_model.tar --restore-model-path ./ckpts/onerestore_cdd-11.tar --input ./image/ --output ./output/
>>>>>>> Stashed changes

# Interactive web interface with app.py
python app.py

# Push models to Hugging Face
python push_and_reload_from_hf.py
```

## File Structure

```
./
├── app.py                           # Gradio web interface
├── fix_dataset.py                   # Dataset alignment & split
├── makedataset.py                   # HDF5 dataset generator
├── remove_optim.py                  # Checkpoint optimizer removal
├── test.py                          # Batch inference
├── push_and_reload_from_hf.py       # Hugging Face integration
<<<<<<< Updated upstream
├── train_restoration_single-gpu.py  # Single GPU training
├── train_restoration_multi-gpu.py   # Multi GPU training
=======
>>>>>>> Stashed changes
├── requirements.txt                 # Dependencies
├── ckpts/                           # Model checkpoints
├── image/                           # Input dataset
│   ├── CDD-11_train/
│   └── CDD-11_test/
└── output/                          # Inference output
```

## Key Notes

**Dataset Alignment:**
- `fix_dataset.py` must be run before `makedataset.py`
- Ensures all 13 degradation classes have synchronized filenames
- Critical for proper training patch generation

**Model Checkpoint Format:**
- Checkpoints with optimizers: `{model}_model.tar` → state_dict + optimizer
<<<<<<< Updated upstream
- Clean checkpoints: `{model}_model.tar` → state_dict only
=======
- Clean checkpoints: `{model}_cdd-11.tar` or `{model}_model.tar` → state_dict only
>>>>>>> Stashed changes
- Use `remove_optim.py` before deployment

**Degradation Classes:**
13 total classes combining 4 base degradations: low-light, haze, blur, noise
- 4 single: low, haze, blur, noise
- 3 dual combinations with low: low_haze, low_blur, low_noise
- 3 dual combinations without low: haze_blur, haze_noise, blur_noise (covered as composite)
- 3 triple combinations: low_haze_blur, low_haze_noise, low_haze_blur_noise
<<<<<<< Updated upstream

## Acknowledgments

This framework is based on [OneRestore](https://github.com/gy65896/OneRestore), a universal restoration framework for composite degradation.
=======
<<<<<<< HEAD

#### If you have any questions, please get in touch with me (guoyu65896@gmail.com).
=======
>>>>>>> 6a3f3d5 (Update README with project details)
>>>>>>> Stashed changes
