import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
from param import args
from model.cim import CIM
from model.mrgt import MRGT
from BERT.modeling import BertLayerNorm, GeLU
from torch.nn.utils.weight_norm import weight_norm

# Max length including <bos> and <eos>
MAX_GQA_LENGTH = 20

class QMRGT(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        self.cim_encoder = CIM(
            args,
            max_seq_length=MAX_GQA_LENGTH
        )
        self.mrgt = MRGT()
        hid_dim = CIM.dim
        self.graph_reasoning = GraphReasoning()
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers),
        )
        self.logit_fc.apply(self.cim_encoder.model.init_bert_weights)
        self.graph_reasoning.apply(self.cim_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        # feats [128,36,2048]
        # boxes [128,36,4]
        # lang_feats [128, 20, 768]
        # visn_feats [128, 36, 768]
        # x [128, 768]
        embedding_output, lang_feats, visn_feats, output = self.cim_encoder(sent, (feat, pos))
        final_output = self.graph_reasoning(visn_feats, lang_feats, embedding_output)
        logit = self.logit_fc(final_output)

        return logit
