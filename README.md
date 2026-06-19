# Lensless Computational Imaging (Final Project for the AI360 Deep Learning Course)

**Author:** Alsu Aldag  
**Comet ML report:**  [link](https://www.comet.com/tasty-arbuz/pytorch-template/reports/aDGwLXTwSNtxIOhPM37sR3Muk)

This repository contains a project-style implementation of lensless image reconstruction methods. The project focuses on ADMM-based reconstruction, learned unrolled ADMM, modular LeADMM variants, and an additional ADMM-100 + Real-ESRGAN post-processing experiment.

The main goal is to reconstruct a target image from a lensless measurement and a mask/PSF using a physics-based forward model and learned reconstruction modules.

---

## References

This project is based on the following papers:

- [Towards Robust and Generalizable Lensless Imaging with Modular Learned Reconstruction](https://arxiv.org/abs/2502.01102)
- [Learned ADMM for lensless imaging](https://arxiv.org/abs/1908.11502)

---

## Implemented methods

- **ADMM-100**: fixed ADMM reconstruction with 100 iterations. This method does not require a learned checkpoint.
- **Unrolled ADMM-20**: trainable unrolled ADMM with learned reconstruction parameters.
- **Modular LeADMM-5**:
  - `leadmm5_pre8`: learned pre-processor + LeADMM-5;
  - `leadmm5_post8`: LeADMM-5 + learned post-processor;
  - `leadmm5_pre4_post4`: learned pre-processor + LeADMM-5 + learned post-processor.
- **ADMM-100 + Real-ESRGAN**: fixed ADMM-100 followed by a general-purpose GAN-based restoration model.

---

## Results

| Model | PSNR | SSIM | LPIPS | MSE |
|---|---|---|---|---|
| LeADMM-5 pre4+post4 | 16.5340 | 0.4661 | 0.451 | 0.0238 |
| LeADMM-5 post8 | 15.5691 | 0.4246 | 0.5503 | 0.0294 |
| LeADMM-5 pre8 | 13.2908 | 0.2399 | 0.6119 | 0.0485 |
| Unrolled ADMM-20 | 11.2894 | 0.3328 | 0.7529 | 0.0781 |
| ADMM-100 | 11.9706 | 0.3479 | 0.7794 | 0.0785 |

Full metrics and training curves are available in the [Comet ML report](https://www.comet.com/tasty-arbuz/pytorch-template/reports/aDGwLXTwSNtxIOhPM37sR3Muk).

---

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── train.py                         # Main training entrypoint
├── inference.py                     # Inference on a custom dataset
├── demo.ipynb                       # Colab demo
├── scripts/
│   ├── benchmark_one_reconstruction_speed.py  # benchmarks single model inference speed
│   ├── benchmark_reconstruction_speed.py      # benchmarks all models and logs to Comet
│   ├── calculate_metrics.py                   # computes PSNR / SSIM / LPIPS for reconstructions
│   ├── count_model_params.py                  # counts number of parameters
│   ├── download_checkpoints.py                # downloads pretrained checkpoints
│   ├── download_custom_dataset.py             # downloads custom dataset from Drive
│   ├── download_masks.py                      # downloads PSF masks
│   ├── estimate_psf_gain.py                   # estimates PSF gain
│   ├── evaluate_admm.py                       # evaluates ADMM baseline
│   ├── evaluate_admm_realesrgan.py            # ADMM + Real-ESRGAN pipeline
│   ├── train_unrolled_admm.py                 # training unrolled ADMM / LeADMM
│   └── visualize_reconstructions.py           # visualization of outputs
└── src/
    ├── configs/
    ├── datasets/
    ├── lensless_helpers/
    ├── logger/
    ├── loss/
    ├── metrics/
    ├── model/
    ├── trainer/
    └── utils/
```

## Installation

In Google Colab or a local Python environment:

```bash
git clone https://github.com/bbalsu/Lensless-Computational-Imaging.git
cd Lensless-Computational-Imaging
pip install -r requirements.txt
```

The code uses Hydra configs, so commands should usually be run from the repository root with:

```bash
PYTHONPATH=. python ...
```

---

## Device note

Most configs use GPU by default:

```bash
device=cuda
```

If CUDA is not available or if you intentionally want to run on CPU, explicitly pass:

```bash
device=cpu
```

This applies to training, evaluation, inference, and benchmarking. CPU execution is supported, but it is much slower.

---

## Data and required resources

Training and validation use the HuggingFace dataset [bezzam/DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K).

The dataset itself is loaded automatically by `DigiCamHFDataset` when training or evaluation starts. However, the mask files used to build PSFs are stored separately and must be downloaded before training or evaluating on DigiCam data:

```bash
PYTHONPATH=. python scripts/download_masks.py
```

This creates:

```text
data/masks/mask_0.npy
...
data/masks/mask_99.npy
```

Model checkpoints can be downloaded with:

```bash
PYTHONPATH=. python scripts/download_checkpoints.py --model all
```

To download only one checkpoint:

```bash
PYTHONPATH=. python scripts/download_checkpoints.py --model leadmm5_pre4_post4
```

Available checkpoint names:

```text
unrolled_admm
leadmm5_pre4_post4
leadmm5_pre8
leadmm5_post8
```

The fixed `admm` baseline does not require a checkpoint.

---

## Comet ML logging

The project supports Comet ML logging. Before enabling logging, set your Comet API key, for example in Colab:

```python
import os
os.environ["COMET_API_KEY"] = "PASTE_YOUR_COMET_API_KEY_HERE"
```

To enable Comet logging, pass:

```bash
logging.log_comet=true
writer.run_name=my-run-name
```

To run without Comet, pass:

```bash
logging.log_comet=false
```

---

## Training learned models

The recommended training entrypoint is:

```bash
PYTHONPATH=. python train.py
```

The script uses `src/configs/train_unrolled_admm.yaml` and supports Hydra overrides.

Available model options:

```text
model=unrolled_admm
model=leadmm5_pre8
model=leadmm5_post8
model=leadmm5_pre4_post4
```

Example: train Modular LeADMM-5 with pre- and post-processors:

```bash
PYTHONPATH=. python train.py \
  model=leadmm5_pre4_post4 \
  device=cuda \
  train_limit=null \
  val_limit=null \
  trainer.n_epochs=25 \
  dataloader.batch_size=4 \
  dataloader.num_workers=2 \
  optimizer.lr=1e-4 \
  lr_scheduler=null \
  loss.mse_weight=1.0 \
  loss.lpips_weight=1.0 \
  trainer.save_dir=checkpoints/leadmm5-pre4-post4 \
  logging.log_comet=true \
  trainer.log_first_n_images=10 \
  writer.run_name=leadmm5-pre4-post4
```

Main training options:

- `model=...`: selects a model config from `src/configs/model/`;
- `device=cuda`: runs on GPU; use `device=cpu` explicitly for CPU;
- `train_limit=null`: uses the full train split; replace with a number for a quick debug run;
- `val_limit=null`: uses the full validation split; replace with a number for a quick debug run;
- `trainer.n_epochs=25`: number of training epochs;
- `dataloader.batch_size=4`: batch size;
- `optimizer.lr=1e-4`: learning rate;
- `loss.mse_weight=1.0`: MSE loss weight;
- `loss.lpips_weight=1.0`: LPIPS perceptual loss weight;
- `trainer.save_dir=...`: checkpoint directory;
- `trainer.log_first_n_images=10`: number of validation examples logged to Comet;
- `writer.run_name=...`: Comet run name.

Before running this command on DigiCam data, make sure masks are available:

```bash
PYTHONPATH=. python scripts/download_masks.py
```

---

## Evaluation

Fixed ADMM-100 can be evaluated without training:

```bash
PYTHONPATH=. python scripts/evaluate_admm.py \
  device=cuda \
  split=val \
  limit=null \
  dataloader.batch_size=8 \
  logging.log_comet=true
```

ADMM-100 followed by Real-ESRGAN post-processing can be evaluated with:

```bash
PYTHONPATH=. python scripts/evaluate_admm_realesrgan.py \
  device=cuda \
  evaluation.split=val \
  evaluation.limit=null \
  evaluation.batch_size=8 \
  evaluation.log_first_n_images=10 \
  restoration.tile=256 \
  logging.log_comet=true \
  writer.run_name=admm100-realesrgan-full-val
```

The Real-ESRGAN experiment is used as a general-purpose restoration baseline on top of fixed ADMM-100. It is not trained specifically for lensless imaging.

---

## Reconstruction speed benchmark

To compare reconstruction speed across methods:

```bash
PYTHONPATH=. python scripts/benchmark_reconstruction_speed.py \
  device=cuda \
  benchmark.split=val \
  benchmark.limit=100 \
  benchmark.batch_size=1 \
  logging.log_comet=true \
  writer.run_name=reconstruction-speed-val100-bs1
```

The script saves a CSV table under the configured benchmark output directory and can also log the results to Comet.

For a single method, use:

```bash
PYTHONPATH=. python scripts/benchmark_one_reconstruction_speed.py \
  method.name=leadmm5_pre4_post4 \
  method.model_config=leadmm5_pre4_post4 \
  method.checkpoint_path=checkpoints/leadmm5_pre4_post4/best.pth \
  device=cuda \
  benchmark.split=val \
  benchmark.limit=100 \
  benchmark.batch_size=1
```

---

## Inference on a custom dataset

The custom dataset must have the following structure:

```text
NameOfDataset/
├── lensless/
│   ├── ImageID1.png
│   └── ImageID2.png
├── masks/
│   ├── ImageID1.npy
│   └── ImageID2.npy
└── lensed/                 # optional ground truth images
    ├── ImageID1.png
    └── ImageID2.png
```

Run inference with the final model:

```bash
PYTHONPATH=. python inference.py \
  model=leadmm5_pre4_post4 \
  datasets.inference.data_dir=/path/to/custom_dataset \
  inferencer.dataset_part=inference \
  inferencer.output_dir=outputs/inference/leadmm5_pre4_post4 \
  inferencer.auto_download=true \
  dataloader.batch_size=1
```

The output directory will contain reconstructed images with the same image ids as the input files.

---

## Demo notebook

The notebook `demo.ipynb` provides a Colab workflow for a user-facing demonstration:

1. Clone the repository and install dependencies.
2. Download model checkpoints.
3. Download a custom dataset from a Google Drive zip archive.
4. Run inference on the custom dataset.
5. Visualize lensless measurements, reconstructions, and ground truth images if available.
6. Calculate reconstruction metrics if ground truth images are provided.

The demo is intended to run in a fresh Google Colab session.

---

## Additional utilities

Download a custom dataset zip from Google Drive:

```bash
PYTHONPATH=. python scripts/download_custom_dataset.py \
  --url "GOOGLE_DRIVE_ZIP_URL" \
  --out-dir data/custom_demo \
  --zip-path data/custom_demo.zip \
  --force
```

Calculate metrics for saved reconstructions:

```bash
PYTHONPATH=. python scripts/calculate_metrics.py \
  --gt-dir /path/to/custom_dataset/lensed \
  --recon-dir outputs/inference/leadmm5_pre4_post4 \
  --out-dir outputs/metrics/leadmm5_pre4_post4 \
  --device cuda \
  --resize-target
```

Visualize saved reconstructions:

```bash
PYTHONPATH=. python scripts/visualize_reconstructions.py \
  --data-dir /path/to/custom_dataset \
  --recon-dir outputs/inference/leadmm5_pre4_post4 \
  --out-path outputs/demo_visualization.png \
  --n-images 5
```

---

## Acknowledgements

This repository is based on the PyTorch Project Template by Petr Grinberg:

```text
https://github.com/Blinorot/pytorch_project_template
```
