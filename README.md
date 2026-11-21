# Spectral Circuit Analysis (SCA)

SCA finds task-relevant units (attention heads and MLP blocks) in a transformer from the gradients of a task metric, without hand-built corrupted inputs. For each task, it collects per-prompt gradients over paraphrases of the same instance, splits the parameters into unit blocks, and runs a block-wise SVD. Each unit gets an importance score and a low-rank gradient subspace. Comparing these subspaces across tasks separates **universal** units from **family-specific** and **task-specific** ones. Activation-patching knockouts then check those classes causally.

Supported models: GPT-2 small and Pythia-160M (through TransformerLens / Hugging Face).

## Repository layout

```
run.py                  per-task SCA pipeline (gradients -> unit blocks -> block-wise SVD -> unit ranking)
run_baselines.py        raw-gradient magnitude / direction baselines for one task
run_knockout.py         head-level faithfulness sweeps (SCA vs EAP / EAP-IG) for one task
run_cross_task.py       cross-task subspace overlap, universality scores, task similarity
run_M_sweep.py          sensitivity to the number of paraphrases M
eval_recall_at_k.py     recall@K of a unit ranking against the IOI circuit (Wang et al., 2023)

configs/                one YAML per task (GPT-2 small) + configs/pythia_ioi_strong.yaml
data/                   paraphrase sets used for gradient collection, plus data/loader.py
eap_data/               clean / corrupted prompt CSVs used for patching
eap/, eap_pythia160/    EAP, EAP-IG and EAP-IG-KL head rankings (GPT-2 / Pythia-160M)
spectral/               gradient collection, block construction, SVD, per-task plots
circuits_discovery/     activation patching and faithfulness sweeps
cross_task/             unit classification, knockout validation, cross-task figures
  run_gpt_12tasks/      cross-task results for GPT-2 small (12 tasks)
  run_pythia_12tasks/   cross-task results for Pythia-160M (12 tasks)
utils/                  config / IO helpers, model loading, figure style
```

The EAP / EAP-IG rankings in `eap/` and `eap_pythia160/` were computed with [EAP-IG](https://github.com/hannamw/EAP-IG) on the prompts in `eap_data/`.

## Installation

Python 3.10 and a CUDA GPU are recommended.

```bash
git clone https://github.com/YujieZhangzz/Spectral-Circuit-Analysis.git && cd Spectral-Circuit-Analysis
pip install -r requirements.txt
```

All commands below run from the repository root. Per-task outputs go to `results/<experiment.name>/`.

## Usage

### 1. Per-task SCA

```bash
python run.py --config configs/gpt2_small_ioi_strong.yaml
```

This writes `ranked_units.csv`, `blocks.json`, `weighted_unit_subspaces.npz` and diagnostic plots. Gradient collection uses a memory-mapped temp file (`collector.tmpdir` in the config, `/dev/shm` by default). The file can be large, so point it at a disk with enough free space.

The 12 tasks used in the paper:

| task | config |
|---|---|
| IOI | `gpt2_small_ioi_strong.yaml` |
| colored objects | `gpt2_small_col_obj.yaml` |
| country–capital | `gpt2_small_country_cap.yaml` |
| person–country | `gpt2_small_person_country.yaml` |
| gender bias | `gpt2_small_gender_bias.yaml` |
| gender pronoun | `gpt2_small_gp.yaml` |
| person–pronoun | `gpt2_small_person_pro.yaml` |
| greater-than (year) | `gpt2_small_gt.yaml` |
| greater-than (price) | `gpt2_small_gt_price.yaml` |
| subject–verb agreement | `gpt2_small_sva.yaml` |
| SVA (BLiMP) | `gpt2_small_sva_blimp.yaml` |
| SVA (is / are) | `gpt2_small_sva_isare.yaml` |

Variants such as `*_boundary.yaml` and `gender_bias_{first,sec}.yaml` are ablations of the metric or target position.

### 2. Single-task faithfulness (optional)

```bash
python run_baselines.py --config configs/gpt2_small_ioi_strong.yaml
python run_knockout.py  --config configs/gpt2_small_ioi_strong.yaml --device cuda:0
python eval_recall_at_k.py results/gpt2_small_ioi_strong_N100_M5_linear/ranked_units.csv
```

### 3. Cross-task analysis

```bash
python run_cross_task.py --out-dir cross_task/run_gpt_12tasks --configs \
    configs/gpt2_small_{col_obj,country_cap,gender_bias,gp,gt,gt_price,ioi_strong,person_country,person_pro,sva,sva_blimp,sva_isare}.yaml

# universal / family-specific / task-specific / mixed
python cross_task/classify_units.py --use_weighted --auto_threshold

# activation-patching validation with a type-matched random baseline
python cross_task/knockout_validate_actpatch.py --device cuda:0 \
    --classify_subdir classify_weighted --out_subdir knockout_validation_v1_typematched

python cross_task/figs_composite_and_validation.py
python cross_task/plot_unit_map.py
python cross_task/plot_unit_circuit.py
python cross_task/plot_clustermap.py
```

The cross-task scripts read `cross_task/run_gpt_12tasks/` by default. For the Pythia-160M results, set `CT_RUN=run_pythia_12tasks`, and pass `--model EleutherAI/pythia-160m` to `knockout_validate_actpatch.py`. `figs_composite_and_validation.py` only handles GPT-2.

Everything after `run_cross_task.py` reads `cross_task_result.pkl`, which is included for both models. Classification and figures can therefore be reproduced without a GPU; the knockout validation still needs one.

`cross_task/parameter_study*.py` runs the threshold sensitivity study of the classifier.

## Included results

`cross_task/run_gpt_12tasks/` and `cross_task/run_pythia_12tasks/` contain the cross-task results reported in the paper:
- unit classifications (`classify/`, `classify_weighted/`)
- knockout validation tables (`knockout_validation_*/`)
- universal-unit lists
- the figures

Per-task gradient caches are not included because of their size; regenerate them with step 1.

## License

MIT, see [LICENSE](LICENSE).
