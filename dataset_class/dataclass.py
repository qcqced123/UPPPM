import ast
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor
from dataset_class.text_preprocessing import add_special_token


class UPPPMDataset(Dataset):
    """ For Token Classification Task class """
    def __init__(self, cfg, df, is_valid=False):
        super().__init__()
        self.anchor_list = df.anchor.to_numpy()
        self.target_list = df.targets.to_numpy()
        self.context_list = df.context_text.to_numpy()
        self.score_list = df.scores.to_numpy()
        self.id_list = df.ids.to_numpy()
        self.cfg = cfg
        self.is_valid = is_valid

    def tokenizing(self, text: str) -> dict:
        inputs = self.cfg.tokenizer.encode_plus(
            text,
            max_length=self.cfg.max_len,
            padding='max_length',
            truncation=True,
            return_tensors=None,
            add_special_tokens=True,
        )
        return inputs

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, idx: int):
        """
        1) make Embedding Shape,
            - Data: [cls]+[anchor]+[sep]+[target]+[tar]+[target]+[tar]...+[tar]+[cpc_text]+[sep]
            - Label: [-1] * self.cfg.max_len, target value의 인덱스 위치에 score_class값 전달
        2) apply data augment
            - shuffle target values
        """
        add_special_token(self.cfg)
        scores = np.array(ast.literal_eval(self.score_list[idx]))  # len(scores) == target count
        target_mask = np.zeros(self.cfg.max_len)
        targets = np.array(ast.literal_eval(self.target_list[idx]))

        # Data Augment for train stage: shuffle target value's position index
        if not self.is_valid:
            indices = list(range(len(scores)))
            random.shuffle(indices)
            scores = scores[indices]
            targets = targets[indices]

        text = self.cfg.tokenizer.cls_token + self.anchor_list[idx] + self.cfg.tokenizer.sep_token
        for target in targets:
            text += target + self.cfg.tokenizer.tar_token
        text += self.context_list[idx] + self.cfg.tokenizer.sep_token

        # tokenizing & make label list
        inputs = self.tokenizing(text)
        label = torch.full([self.cfg.max_len], -1, dtype=torch.float)
        cnt_tar, cnt_sep, nth_target, prev_i = 0, 0, -1, -1
        for i, input_id in enumerate(inputs['input_ids']):
            if input_id == self.cfg.tokenizer.tar_token_id:
                cnt_tar += 1
                if cnt_tar == len(targets):
                    break
            if input_id == self.cfg.tokenizer.sep_token_id:
                cnt_sep += 1
            if cnt_sep == 1 and input_id not in [self.cfg.tokenizer.pad_token_id, self.cfg.tokenizer.sep_token_id,
                                                 self.cfg.tokenizer.tar_token_id]:
                if (i - prev_i) > 1:
                    nth_target += 1
                label[i] = scores[nth_target]
                target_mask[i] = 1
                prev_i = i

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs, target_mask, label


class TestDataset(Dataset):
    """ For Inference Dataset Class """
    def __init__(self, cfg, tokenizer, df):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer  # get from train.py
        self.df = df

    def tokenizing(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.cfg.max_len,
            padding='max_length',
            truncation=True,
            return_tensors=None,
            add_special_tokens=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v)
        return inputs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        inputs = self.tokenizing(self.df.iloc[idx, 1])
        return inputs
