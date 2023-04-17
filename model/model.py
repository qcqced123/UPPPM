import torch
import torch.nn as nn
import model.pooling as pooling
from torch import Tensor
from transformers import AutoConfig, AutoModel
from model.model_utils import freeze, reinit_topk, num_classes


class TokenModel(nn.Module):
    """ For Token Classification Pipeline """
    def __init__(self, cfg, n_vocabs: int):
        super().__init__()
        self.cfg = cfg
        self.auto_cfg = AutoConfig.from_pretrained(cfg.model_name, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(cfg.model_name, config=self.auto_cfg)
        self.model.resize_token_embeddings(n_vocabs)
        self.fc = nn.Linear(self.auto_cfg.hidden_size, num_classes(self.cfg.loss_fn))
        # self.model.load_state_dict(torch.load(cfg.checkpoint_dir + cfg.state_dict), strict=False)

        if cfg.reinit:
            self._init_weights(self.fc)
            reinit_topk(self.model, cfg.num_reinit)

        if cfg.freeze:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:cfg.num_freeze])

        if cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module) -> None:
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def feature(self, inputs: dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: dict) -> list[Tensor]:
        outputs = self.feature(inputs)
        logit = self.fc(outputs.last_hidden_state).squeeze(-1)
        return logit


class SentenceModel(nn.Module):
    """ For Sentence-Level Classification Pipeline """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.auto_cfg = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(cfg.model, config=self.auto_cfg)
        self.fc = nn.Linear(self.auto_cfg.hidden_size, num_classes(self.cfg.loss_fn))
        self.pooling = getattr(pooling, cfg.pooling)(self.auto_cfg)
        self.model.load_state_dict(
            torch.load(cfg.checkpoint_dir + cfg.state_dict),
            strict=False
        )  # load student model's weight: it already has fc layer, so need to init fc layer later

        if cfg.reinit:
            self._init_weights(self.fc)
            reinit_topk(self.model, cfg.num_reinit)

        if cfg.freeze:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:cfg.num_freeze])

        if cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module) -> None:
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def feature(self, inputs: dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: dict) -> list[Tensor]:
        outputs = self.feature(inputs)
        feature = outputs.last_hidden_state
        if self.cfg.pooling == 'WeightedLayerPooling':
            feature = outputs.hidden_states
        embedding = self.pooling(feature, inputs['attention_mask'])
        logit = self.fc(embedding)
        return logit


class FBPModel(nn.Module):
    """ Model class for Baseline Pipeline """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True
        )
        self.model = AutoModel.from_pretrained(
            cfg.model,
            config=self.auto_cfg
        )
        self.fc = nn.Linear(self.auto_cfg.hidden_size, 6)
        self.pooling = getattr(pooling, cfg.pooling)(self.auto_cfg)
        self.model.load_state_dict(
            torch.load(cfg.checkpoint_dir + cfg.state_dict),
            strict=False
        )  # load student model's weight: it already has fc layer, so need to init fc layer later

        if cfg.reinit:
            self._init_weights(self.fc)
            reinit_topk(self.model, cfg.num_reinit)

        if cfg.freeze:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:cfg.num_freeze])

        if cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module) -> None:
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def feature(self, inputs: dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: dict) -> list[Tensor]:
        outputs = self.feature(inputs)
        if self.cfg.pooling == 'WeightedLayerPooling':
            embedding = self.pooling(outputs.hidden_states)
            logit = self.fc(embedding)
        else:
            embedding = self.pooling(
                outputs.last_hidden_state,
                inputs['attention_mask']
            )
            logit = self.fc(embedding)
        return logit
