import os
import collections
import sys

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from BERT.optimization import BertAdam

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

from utils.param import args
from model.qmrgt import QMRGT
from data_preprocessing.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator

class Model:
    def __init__(self):

        self.model = QMRGT(self.train_tuple.dataset.num_answers)
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]  # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)


    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)

def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)  # data
    tset = GQATorchDataset(dset)  # train_data
    evaluator = GQAEvaluator(dset)  # valid_data
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

if __name__ == "__main__"
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in arg.visable_gpus)

    cpu = torch.device('cpu')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpu = torch.cuda.device_count()
    print('number of the gpu devices------------------------------------->:', device, n_gpu)
    # Build Class
    model = Model()

    # Load Model
    if args.load is not None:
        model.load(args.load)

    # Test or Train
    if args.test is not None:
        print('Begin test dataset prediction')
        if 'submit' in args.test:
            model.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'testdev' in args.test:
            result = model.evaluate(
                get_tuple('testdev', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'testdev_predict.json')
            )
            print(result)
        if 'ood-tail' in args.test:
            result = gqa.evaluate(
                get_tuple('ood-tail', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'ood-tail_predict.json')
            )
            print(result)
        if 'ood-head' in args.test:
            result = gqa.evaluate(
                get_tuple('ood-head', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'ood-head_predict.json')
            )
            print(result)
        if 'ood-all' in args.test:
            result = gqa.evaluate(
                get_tuple('ood-all', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'ood-all_predict.json')
            )
            print(result)
    else:
        print('CMR: Please provide the correct test dataset path!!!')