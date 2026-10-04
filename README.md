# LLM fine-tuning experiments
Our code is primarily based on [ZO-Muon](https://github.com/OPTML-Group/ZO-Muon), [LOZO](https://github.com/optsuite/LOZO.git), [HiZOO](https://github.com/Yanjun-Zhao/HiZOO) and [MeZO](https://github.com/princeton-nlp/MeZO.git).

## Installation
```
conda create -n zo python==3.9.19
conda activate zo
pip install -r requirements.txt
```
This environment supports fine-tuning the OPT, Llama3 and Gemma2 models.

## Usage

### Our proposed methods
Below is an example command for evaluating our proposed **ZO-MOPI** on OPT-13B RTE fine-tuning.
```
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-13b TASK=RTE MODE=ft LR=1e-2 BS=16 EPS=1e-3 RANK=64 STEP_INTERVAL=500  MULTIPLE_SAMPLE=True NUM_SAMPLES=8 STEPS=4000 EVAL_STEPS=1000 bash scripts/zo_mopi.sh
```



### Zeroth-Order Baselines

Run MeZO via:
```
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-13b TASK=RTE MODE=ft LR=1e-7 BS=16 EPS=1e-3 STEPS=20000 EVAL_STEPS=20000 bash scripts/mezo.sh
```

Run LOZO:
```
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-13b TASK=RTE MODE=ft LR=1e-7 BS=16 EPS=1e-3 RANK=4 STEP_INTERVAL=100 STEPS=20000 EVAL_STEPS=20000 bash scripts/lozo.sh
```

For the ZO-Muon variant where gradient orthogonalization is solved by SVD, we set `$OPT` to `muon_svd`:
```
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-13b TASK=RTE MODE=ft LR=1e-2 BS=16 EPS=1e-3 RANK=64 STEP_INTERVAL=100 OPT='muon_svd' MULTIPLE_SAMPLE=True NUM_SAMPLES=4 STEPS=8000 EVAL_STEPS=8000 bash scripts/lowdim.sh
```


### Compare with same Runtime
We can compare difference ZO methods with the same training runtime by setting `$MAX_TIME`, measured in seconds. See an example below:
```
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-13b TASK=RTE MODE=ft LR=1e-2 BS=16 EPS=1e-3 RANK=64 STEP_INTERVAL=100 OPT='muon' MULTIPLE_SAMPLE=True NUM_SAMPLES=4 STEPS=8000 EVAL_STEPS=8000 MAX_TIME=5000 bash scripts/lowdim.sh
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
