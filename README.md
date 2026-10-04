# LLM fine-tuning experiments
Our code is primarily based on [ZO-Muon](https://github.com/OPTML-Group/ZO-Muon).

## Installation
```
conda create -n muon python==3.9.19
conda activate muon
pip install -r requirements.txt
```
This environment supports fine-tuning the OPT and Qwen models.

## Usage

### Our proposed methods
Below is an example command for evaluating our proposed **DAZM** on OPT-13B RTE fine-tuning.
```
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-13b TASK=RTE MODE=ft LR=1e-2 BS=16 EPS=1e-3 RANK=64 STEP_INTERVAL=500  MULTIPLE_SAMPLE=True NUM_SAMPLES=4 STEPS=6000 EVAL_STEPS=1000 bash scripts/greedy_muon.sh
```



### Zeroth-Order Baselines

Run MeZO via:
```
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-13b TASK=RTE MODE=ft LR=1e-7 BS=16 EPS=1e-3 STEPS=15000 EVAL_STEPS=5000 bash scripts/mezo.sh
```

Run LOZO:
```
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-13b TASK=RTE MODE=ft LR=1e-7 BS=16 EPS=1e-3 RANK=4 STEP_INTERVAL=100 STEPS=15000 EVAL_STEPS=5000 bash scripts/lozo.sh
```

For the ZO-Muon variant where gradient orthogonalization is solved by SVD, we set `$OPT` to `muon_svd`:
```
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-13b TASK=RTE MODE=ft LR=1e-2 BS=16 EPS=1e-3 RANK=64 STEP_INTERVAL=100 OPT='muon_svd' MULTIPLE_SAMPLE=True NUM_SAMPLES=4 STEPS=6000 EVAL_STEPS=1000 bash scripts/lowdim.sh
```

### First-Order Methods
Full Adam fine-tuning:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-5 bash finetune.sh
```

LoRA fine-tuning:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL=facebook/opt-13b TASK=SST2 MODE=lora LR=1e-5 bash finetune.sh
```
