# QMRGT: A Question-Guided Multi-hop Reasoning Graph Network for Visual Question Answering
This is the implementation of the framework described in the paper:

## GQA data download link:
[The website](https://cs.stanford.edu/people/dorarad/gqa/download.html) is available for the original GQA dataset. Download questions to `data/raw_data/` and the script to preprocess these datasets is under `data/process_raw_data/process_data.py`.

## Image feature download link:
```bash
    mkdir -p data/imgfeat
    wget https://nlp.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/vg_gqa_obj36.zip -P data/vg_gqa_imgfeat
    unzip data/imgfeat/vg_gqa_obj36.zip -d data && rm data/imgfeat/vg_gqa_obj36.zip
    wget https://nlp.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/gqa_testdev_obj36.zip -P data/vg_gqa_imgfeat
    unzip data/imgfeat/gqa_testdev_obj36.zip -d data && rm data/imgfeat/gqa_testdev_obj36.zip
```
## GQA-OOD data download link:
[The repositories](https://github.com/gqa-ood/GQA-OOD) is available for the original ood_testdev_all, ood_testdev_head and ood_testdev_tail. Download them to `data/raw_data/` and the script to preprocess these datasets is under `data/process_raw_data/process_data.py`.

## Train on GQA:
```python
python run_train.py
```


## Valid on GQA:
```python
python run_test.py --load [the file path to the model]
```
