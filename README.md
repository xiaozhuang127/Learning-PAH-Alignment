# Learning Prototype-Archetype-Hypersphere Alignment for UDA SAR Target Recognition

PyTorch implementation of **Learning Prototype-Archetype-Hypersphere Alignment for Unsupervised Domain Adaptation SAR Target Recognition Using Simulated Data**.

This project studies simulated-to-real SAR target recognition, where labeled simulated samples are available but real target-domain labels are unavailable. The core idea is to construct prototype and archetype structures in the embedding space, then use hypersphere-based geometry to improve cross-domain feature alignment.

## Highlights

- Prototype-archetype-hypersphere alignment for unsupervised domain adaptation
- Designed for simulated-to-real SAR target recognition
- Modular PyTorch implementation for models, losses, augmentation, and data loading
- Supports reproducible training and ablation-oriented experimentation

## Method Overview

The method follows three main steps:

1. Learn feature embeddings for source and target SAR samples.
2. Estimate prototypes and archetypes that describe class-level and extreme feature structures.
3. Build hypersphere constraints in the embedding space to align target features with source-domain semantic geometry.

Recommended figure to add:

```text
simulated SAR images -> encoder -> prototypes / archetypes -> hypersphere alignment -> target recognition
real SAR images      -> encoder -> target embeddings  ----/
```

## Repository Structure

```text
.
|-- main.py                         # Training entry point
|-- losses.py                       # Alignment and classification losses
|-- dwt2_tensor_mixup.py            # Wavelet-domain augmentation / mixup
|-- model/
|   |-- resnet.py                   # Backbone networks
|   |-- basenet.py
|   `-- MME_basenet.py
|-- utils/
|   |-- myself_return_dataset.py    # Dataset loading
|   |-- randaugment.py              # Data augmentation
|   |-- lr_schedule.py              # Learning-rate schedule
|   `-- utils.py
`-- requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

The original experiments were developed with Python and PyTorch. If you use a newer CUDA/PyTorch stack, please verify the dependency versions before reproducing results.

## Quick Start

```bash
python main.py
```

For a stronger release, add the exact training commands used in the paper, including dataset path, source domain, target domain, batch size, learning rate, and random seed.

Example format:

```bash
python main.py \
  --source simulated \
  --target real \
  --net resnet34 \
  --batch-size 64 \
  --lr 0.001 \
  --seed 2026
```

## Results

Add the main experimental results here once they are ready.

| Method | Source Domain | Target Domain | Target Accuracy |
| --- | --- | --- | ---: |
| Source only | Simulated | Real | TBD |
| Baseline UDA | Simulated | Real | TBD |
| PAH Alignment | Simulated | Real | TBD |

## Ablation Study

Recommended ablations:

- Without archetype alignment
- Without prototype alignment
- Without hypersphere constraint
- Without wavelet-domain mixup
- Different backbone networks

| Variant | Target Accuracy | Delta |
| --- | ---: | ---: |
| Full PAH Alignment | TBD | - |
| w/o archetype alignment | TBD | TBD |
| w/o hypersphere constraint | TBD | TBD |

## Reproducibility Checklist

- [ ] Add dataset preparation instructions
- [ ] Add exact training commands
- [ ] Add pretrained/checkpoint links if available
- [ ] Add random seeds and hardware information
- [ ] Add result tables and ablation results
- [ ] Add method diagram
- [ ] Remove `__pycache__` and `.pyc` files from the repository

## Citation

If this work is associated with a paper or manuscript, add the BibTeX entry here.

```bibtex
@article{pah_alignment_sar_uda,
  title={Learning Prototype-Archetype-Hypersphere Alignment for Unsupervised Domain Adaptation SAR Target Recognition Using Simulated Data},
  author={Zhuang, ...},
  year={2026}
}
```

## Contact

For questions, please open an issue or contact the author through GitHub.

