import os

import torch
import torch.nn as nn

import sys

from BERT.tokenization import BertTokenizer
from BERT.modeling import VISUAL_CONFIG, CrossEncoder



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
    return features


def set_visual_config():
    VISUAL_CONFIG.l_layers = 9
    VISUAL_CONFIG.x_layers = 5
    VISUAL_CONFIG.r_layers = 5

class CIM(nn.Module):
    def __init__(self, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length
        set_visual_config()

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        self.model = CrossEncoder.from_pretrained(
            "bert-base-uncased",
        )
    @property
    def dim(self):
        return 768

    def forward(self, sents, feats, visual_attention_mask=None):
        # visual_attention_mask:none

        train_features = convert_sents_to_features(
            sents, self.max_seq_length, self.tokenizer) # 长度为128的list

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()  # [128,20]
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()  # [128,20]
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()  # [128,20]
        embedding_output, feat_seq, output = self.model(input_ids, segment_ids, input_mask,
                            visual_feats=feats,
                            visual_attention_mask=visual_attention_mask)
        lang_feats, visn_feats = feat_seq

        return embedding_output, lang_feats, visn_feats, output

    # def save(self, path):
    #     torch.save(self.model.state_dict(),
    #                os.path.join("%s.pth" % path))
    #
    # def load(self, path):
    #     # Load state_dict from snapshot file
    #     state_dict = torch.load("%s.pth" % path)
    #     new_state_dict = {}
    #     for key, value in state_dict.items():
    #         if key.startswith("module."):
    #             new_state_dict[key[len("module."):]] = value
    #         else:
    #             new_state_dict[key] = value
    #     state_dict = new_state_dict
    #
    #     # Load weights to model
    #     self.model.load_state_dict(state_dict, strict=False)