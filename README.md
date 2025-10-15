# EAOA-C10: Energy-based Active Open-set Annotation (CIFAR-10)

This repository contains a clean, CIFAR-10-only implementation of an energy-based active learning approach for open-set annotation (EAOA). The pipeline maintains two heads: a C-way classifier for known classes and a C+1 detector for unknown/ID separation. It queries new labels by first filtering to low epistemic uncertainty (likely closed-set) and then selecting high aleatoric uncertainty samples. The candidate width is adapted per round to hit a target query precision.

## Key ideas (one slide)
- Two heads: C-way classifier (AU) + C+1 detector (EU with energy loss)
- Candidate filtering: low EU → AU is meaningful
- rKNN energy: neighborhood-based EU stabilization from labeled features
- Adaptive k1: widens/narrows candidate set to match target query precision

## Environment
- Python 3.8+
- PyTorch (CUDA build recommended if using GPU)
- scikit-learn, torchvision, numpy, tqdm, matplotlib

You can start from `environment.txt` as a reference for versions.

## Dataset
CIFAR-10 is downloaded automatically by `torchvision` to `./data/cifar10` on first run.

## Quick start (CIFAR-10 only)

```
python main.py --query-strategy eaoa_sampling --init-percent 1 --known-class 2 --query-batch 1500 --seed 1 --model resnet18 --dataset cifar10 --max-query 11 --max-epoch 200 --gpu 0

python main.py --query-strategy eaoa_sampling --init-percent 1 --known-class 3 --query-batch 1500 --seed 1 --model resnet18 --dataset cifar10 --max-query 11 --max-epoch 200 --gpu 0

python main.py --query-strategy eaoa_sampling --init-percent 1 --known-class 4 --query-batch 1500 --seed 1 --model resnet18 --dataset cifar10 --max-query 11 --max-epoch 200 --gpu 0
```


Flags you may tweak:
- `--known-class`: number of known classes C in the split
- `--init-percent`: percent of known-class data initially labeled (mini mode)
- `--query-batch`: labels acquired per AL round
- `--energy-weight, --m-in, --m-out`: energy-margin loss hyperparams (C+1 head)
- `--target_precision, --z, --k1, --a`: adaptive candidate sizing

GPU vs CPU:
- Use GPU: `--gpu 0` (and ensure `torch.cuda.is_available()` is True)
- Force CPU: `--use-cpu`

## Produced plots
Use the provided plotting utility:

```bash
python plot_logs.py *.log --dataset-filter cifar10 --compare --outdir plots
python plot_logs.py *.log --dataset-filter cifar10 --scatter-prec-vs-acc --outdir plots
python plot_logs.py cifar10_*.log --outdir plots
```

This generates:
- `plots/acc_vs_cycles_C.png` (Test accuracy vs AL cycles, C-way)
- `plots/precision_vs_final_acc_C.png` (Mean query precision vs final test accuracy)
- `plots/accuracy_vs_round.png` (Single-run accuracy curve)

## Project structure
- `main.py`: AL loop, model training, evaluation, EAOA querying, k1 adaptation
- `datasets.py`: CIFAR-10 loaders and labeled/unlabeled/test splits
- `resnet.py`: small-image ResNet18 returning logits and embeddings
- `query_strategies.py`: EAOA sampling (AU/EU, GMMs, rKNN, per-class select)
- `utils.py`: label mapping, split seed, meters
- `plot_logs.py`: log parsing and the three must-have plots

## Reference
If you use or extend this implementation, please cite the original paper that inspired this code and our cleaned CIFAR-10-only re-implementation:

- Original concept:
  - Zong, C.-C., & Huang, S.-J. (CVPR 2025). Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach. [[paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Zong_Rethinking_Epistemic_and_Aleatoric_Uncertainty_for_Active_Open-Set_Annotation_An_CVPR_2025_paper.html)

- This repository (EAOA-C10):
```
@software{EAOA_C10_2025,
  title        = {EAOA-C10: Energy-based Active Open-set Annotation (CIFAR-10)},
  author       = {Your Name},
  year         = {2025},
  url          = {https://github.com/your-org/EAOA-C10}
}
```

## Acknowledgement
Portions of the original design were inspired by the publicly available EOAL implementation by Safaei et al. and the original EAOA paper’s methodology.

## Citation

If you find this repo useful for your research, please consider citing the paper.

```
@inproceedings{zong2025rethinking,
  title={Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach},
  author={Zong, Chen-Chen and Huang, Sheng-Jun},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={10153--10162},
  year={2025}
}
```

## Acknowledgement

Thanks to Safaei et al. for publishing their code for [EOAL](https://github.com/bardisafa/EOAL). Our implementation is heavily based on their work.
