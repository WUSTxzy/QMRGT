# VOLTA-MRGT

This is the version of the VOTAL-MRGT framework described in the paper:
> A Question-Guided Multi-hop Reasoning Graph Network for Visual Question Answering.

The code is based on [VOTAL](https://github.com/e-bug/volta), and the framework is described in this paper:
> Emanuele Bugliarello, Ryan Cotterell, Naoaki Okazaki and Desmond Elliott. [Multimodal Pretraining Unmasked: A Meta-Analysis and a Unified Framework of Vision-and-Language BERTs](https://arxiv.org/abs/2011.15124). _Transactions of the Association for Computational Linguistics_ 2021; 9 978–994.

## Repository Setup

1\. Create conda environment.
```text
conda create -n volta-mrgt python=3.6
conda activate volta-mrgt
pip install -r requirements.txt
```

2\. Install PyTorch
```text
conda install pytorch=1.4.0 torchvision=0.5 cudatoolkit=10.1 -c pytorch
```

3\. Install [apex](https://github.com/NVIDIA/apex).
If you use a cluster, you may want to first run commands like the following:
```text
module load cuda/10.1.105
module load gcc/8.3.0-cuda
```

4\. Setup the `refer` submodule for Referring Expression Comprehension:
```
cd tools/refer; make
```

5\. Install this codebase as a package in this environment.
```text
python setup.py develop
```


## Data

Check out [`data/README.md`](data/README.md) for links to preprocessed data and data preparation steps.

[`features_extraction/`](features_extraction) contains the latest feature extraction steps in `hdf5` and `npy` instead of `csv`, and with different backbones. Steps for the IGLUE datasets can be found in its [datasets sub-directory](features_extraction/datasets).

H5 versions: quickly convert H5 to LMDB locally using [this script](https://github.com/e-bug/volta/blob/main/features_extraction/h5_to_lmdb.py).


## Models

Check out [`MODELS.md`](MODELS.md) for links to pretrained models and how to define new ones in VOLTA.

Model configuration files are stored in [config/](config). They can control whether to perform graph reasoning with our proposed MRGT module.

## Training and Evaluation

VOLTA provides sample scripts to train and evaluate models in [examples/](examples).
These include ViLBERT, LXMERT and VL-BERT as detailed in the original papers, 
as well as ViLBERT, LXMERT, VL-BERT, VisualBERT and UNITER as specified in VOLTA controlled study.

Task configuration files are stored in [config_tasks/](config_tasks).



## Acknowledgement

The codebase heavily relies on these excellent repositories:
- [volta](https://github.com/e-bug/volta)
- [vilbert-multi-task](https://github.com/facebookresearch/vilbert-multi-task)
- [vilbert_beta](https://github.com/jiasenlu/vilbert_beta)
- [lxmert](https://github.com/airsplay/lxmert)
- [VL-BERT](https://github.com/jackroos/VL-BERT)
- [visualbert](https://github.com/uclanlp/visualbert)
- [UNITER](https://github.com/ChenRocks/UNITER)
- [pytorch-transformers](https://github.com/huggingface/pytorch-transformers)
- [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)
- [transformers](https://github.com/huggingface/transformers)
