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
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        self.model = QMRGT(self.train_tuple.dataset.num_answers)

        if args.multiGPU:
            self.model.multi_gpu()
        self.model = self.model.cuda()

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):

        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        best_valid = 0.

        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                if args.mce_loss:
                    max_value, target = target.max(1)
                    loss = self.mce_loss(logit, target) * logit.size(1)
                else:
                    loss = self.bce_loss(logit, target)
                    loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

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

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

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

    # Train

    print('Load Train data!')
    if model.valid_tuple is not None:
        print('Load Valid data!')
    else:
        print("No Valid data, only Train data!")
    model.train(model.train_tuple, model.valid_tuple)