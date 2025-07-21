# Matrix mixer analysis for time series classification: attention to tokenization

This repository contains the official code for the paper:

**"Matrix mixer analysis for time series classification: attention to tokenization"**

Building upon transformer-based foundational models, our study addresses the impact of **different tokenization and temporal fusion strategies** across inherently diverse time series datasets.

Key contributions include:
* Redefining the matrix mixer framework as a general architectural design toolbox for TSC.
* Analyzing how matrix mixer structures and tokenization methods impact TSC model effectiveness.
* Providing insights into choosing optimal patch embedding configurations for enhanced model performance.

The models developed in this study achieved state-of-the-art level results on two widely used time series classification benchmarks, reaching average accuracies of **73.1% in supervised settings** and **86.0% in self-supervised settings** using a unified architecture.



## Dependencies

Install the required Python packages using `pip` from the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The key dependencies include:

* `einops==0.8.0` 
* `numpy==1.23.5` 
* `pandas==1.5.3` 
* `patool==1.12`
* `scikit-learn==1.2.2` 
* `scipy==1.10.1` 
* `sktime==0.16.1`
* `torch==2.5.0` 
* `torch-dct==0.1.6` 
* `torchscale==0.3.0` 
* `torchvision==0.20.0` 
* `tqdm==4.64.1` 
* `triton==2.2.0` 

## Data Preparation

This project utilizes datasets from public time series archives. Please follow the instructions below to download and prepare the data for both supervised and self-supervised experiments:

* **For Supervised Datasets:**
    The supervised experiments use datasets from the [UEA & UCR Time Series Classification Repository](https://www.timeseriesclassification.com/). Please refer to the data preparation instructions in the [THUMSL Time-Series-Library repository](https://github.com/thuml/Time-Series-Library) for details on how to structure these datasets for use with this codebase. Ensure the downloaded datasets are placed in the `./dataset/` directory, following the expected folder structure (e.g., `./dataset/EthanolConcentration/`).

* **For Self-Supervised Datasets:**
    The self-supervised pre-training and fine-tuning experiments use datasets primarily from the [SimMTM project's data setup](https://github.com/thuml/SimMTM). Please follow the data preparation guidelines provided in the [SimMTM GitHub repository](https://github.com/thuml/SimMTM) to obtain and organize the necessary datasets.

## Usage

This project supports two main operational modes: supervised training and self-supervised pre-training/fine-tuning.

### Supervised Training

To train a Matrix Mixer model in a supervised setting, navigate to the project root directory and run the `run.py` script. The `--root_path` parameter should point to your dataset location.

**Example Command:**

```bash
python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./dataset/EthanolConcentration/ \
    --model_id EthanolConcentration \
    --model Mixer \
    --data UEA \
    --e_layers 1 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 512 \
    --learning_rate 0.001 \
    --train_epochs 100 \
    --patience 10 \
    --context_points_fn 144 \
    --stride 8 \
    --features_domain dct \
    --hidden_depth 1 \
    --seq_mixing transformer \
    --patch_len 4
```

### Supervised Training

For pre-training:

```bash
python -u PITS_pretrain.py \
    --dset_pretrain Epilepsy \
    --n_epochs_pretrain 100 \
    --context_points 178 \
    --d_model 256 \
    --device_id 0 \
    --feature_domain dct_mean_rm \
    --max_len 1000 \
    --n_heads 8 \
    --d_ff 128 \
    --e_layers 8 \
    --moe mlp \ # Feature mixer
    --seed 2025 \
    --patch_len 64 \
    --dropout 0.2 \
    --hidden_depth 1 \
    --stride 32
```

and for fine-tuning:

```bash
python -u PITS_finetune.py \
    --dset_pretrain Epilepsy \
    --dset_finetune Epilepsy \
    --n_epochs_finetune_head 0 \
    --target_points 2 \
    --context_points 178 \
    --d_model 256 \
    --aggregate avg \
    --is_finetune_cls 1 \
    --cls 1 \
    --device_id 0 \
    --n_epochs_pretrain 100 \
    --h_mode fix \
    --feature_domain dct_mean_rm \
    --max_len 1000 \
    --n_heads 8 \
    --d_ff 128 \
    --e_layers 8 \
    --moe mlp \
    --seed 2025 \
    --patch_len 64 \
    --dropout 0.2 \
    --hidden_depth 1 \
    --stride 32 \
    --n_epochs_finetune_entire 400
```

## Results

For detailed experimental results of search experiments, please refer to the results folder within this repository. The configuration and command line arguments used for various experiments are designed to be reproducible with the provided code.

## Citation

If you find this repository data useful, please consider citing it. The manuscript has been submitted to Information Fusion. (A BibTeX entry will be provided here once a DOI is available.)

## Contact / Support

For any questions, issues, or collaborations, feel free to reach out via email:

azizi.mm@ut.ac.ir

## Acknowledgements

We gratefully acknowledge the developers of the following open-source code repositories, as significant portions of our codebase are built upon their foundational work:

[Time-Series-Library](https://github.com/thuml/Time-Series-Library) by THUML

[PITS](https://github.com/seunghan96/pits) by Seunghan96

Their contributions have been invaluable to this research.