[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31012/)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

# Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control

[[Paper](https://arxiv.org/abs/2306.07863)][[Project Website](https://ltzheng.github.io/Synapse/)]

## News

[9/30/2023] We've updated the code format, added evaluation with Mind2Web, and transferred the results to Google Drive. Please kindly re-clone the repository. Our paper on arXiv will be updated soon. Stay tuned!

[6/13/2023] We preprinted our v1-paper on arXiv.

## Overview

We introduce Synapse, an agent that incorporates trajectory-as-exemplar prompting with the associated memory for solving computer control tasks.

![](assets/overview.png)

![](assets/trajectory_prompt.png)

## Install

```bash
conda create -n synapse python=3.10 -y
conda activate synapse
pip install -r requirements.txt
```

Install [Faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) and Download [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) dataset and element rankings. The directory structure should look like this:
```
Mind2Web
├── data
|   |── scores_all_data.pkl
│   ├── train
│   │   └── train_*.json
│   ├── test_task
│   │   └── test_task_*.json
│   ├── test_website
│   │   └── test_website_*.json
│   └── test_domain
│       └── test_domain_*.json
└── ...
```

## Run
Use `build_memory.py` to setup the exemplar memory.

The `memory` folder should contain the following two files:
`index.faiss` and `index.pkl`.

Run MiniWoB++ experiments:
```bash
python run_miniwob.py --env_name <subdomain> --seed 0 --num_episodes 50
python run_miniwob.py --env_name <subdomain> --no_memory --no_filter --seed 0 --num_episodes 50
```

Run Mind2Web experiments:
```bash
python run_mind2web.py --data_dir <path_to_mind2web_dataset> --benchmark <test_task/test_website/test_domain>
python run_mind2web.py --benchmark <test_task/test_website/test_domain> --no_memory --no_trajectory
```

## Fine-tuning

```bash
pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08
pip install git+https://github.com/huggingface/transformers.git@c030fc891395d11249046e36b9e0219685b33399
```

Build Mind2Web datasets for Synapse and memory:
```bash
python build_dataset.py --data_dir <MIND2WEB_DATA_PATH> --no_trajectory --top_k_elements 20 --benchmark train
python build_dataset.py --data_dir <MIND2WEB_DATA_PATH> --top_k_elements 20 --benchmark train
python build_memory.py --env mind2web --mind2web_data_dir <MIND2WEB_DATA_PATH>
```

Fine-tune:
```bash
python finetune_mind2web.py --data_dir <MIND2WEB_DATA_PATH> --base_model codellama/CodeLlama-7b-Instruct-hf --cache_dir <MODEL_PATH> --lora_dir <CHECKPOINT_PATH> --no_trajectory --top_k_elements 20
python finetune_mind2web.py --data_dir <MIND2WEB_DATA_PATH> --base_model codellama/CodeLlama-7b-Instruct-hf --cache_dir <MODEL_PATH> --lora_dir <CHECKPOINT_PATH> --top_k_elements 20
```

Evaluate:
```bash
python evaluate_mind2web.py --data_dir <MIND2WEB_DATA_PATH> --no_memory --no_trajectory --benchmark test_domain --base_model codellama/CodeLlama-7b-Instruct-hf --cache_dir <MODEL_PATH> --lora_dir <CHECKPOINT_PATH> --top_k_elements 20
python evaluate_mind2web.py --data_dir <MIND2WEB_DATA_PATH> --no_memory --benchmark test_domain --base_model codellama/CodeLlama-7b-Instruct-hf --cache_dir <MODEL_PATH> --lora_dir <CHECKPOINT_PATH> --top_k_elements 20
```

## Results

Synapse outperforms the state-of-the-art methods on both MiniWoB++ and Mind2Web benchmarks.

![](assets/miniwob_box_plot.png)

<div style="display: flex; justify-content: space-between;">
    <img src="assets/performance_human.png" alt="Human Performance" width="45%">
    <img src="assets/performance_ccnet.png" alt="CCNet Performance" width="45%">
</div>

<div style="display: flex; justify-content: space-between;">
    <img src="assets/performance_ccnet.png" alt="CCNet Performance" width="45%">
    <img src="assets/performance_webgum.png" alt="WebGUM Performance" width="45%">
</div>

![](assets/mind2web.png)

The trajectories of all experiments can be downloaded from [here](https://drive.google.com/file/d/1hiPQj7m06xU9FEhQTqIJfIlajbov5aWj/view?usp=sharing).

## Citation

```bibtex
@article{zheng2023synapse,
  title={Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control},
  author={Zheng, Longtao and Wang, Rundong and Wang, Xinrun and An, Bo},
  journal={arXiv preprint arXiv:2306.07863},
  year={2023}
}
```
