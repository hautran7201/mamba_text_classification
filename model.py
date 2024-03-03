import json
import torch
import torch.nn as nn
import numpy as np
import os
from transformers import Trainer
from transformers import AutoTokenizer , TrainingArguments
from dataclasses import dataclass, field, asdict
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf , load_state_dict_hf
from collections import namedtuple


class MambaConfig:
    d_model: int = 2560
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple:int = 8
    tie_embeddings = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_json_string(self):
        return json.dumps(asdict(self))

    def to_dict(self) :
        return asdict(self)

class MambaClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes, **kwargs):
        super(MambaClassificationHead, self).__init__()
        self.classification_head = nn.Linear(
            d_model,
            num_classes,
            **kwargs
        )

    def forward(self, hidden_states):
        return self.classification_head(hidden_states)

class MambaTextClassification(MambaLMHeadModel):
    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None
    ) -> None:
        super().__init__(config, initializer_cfg, device, dtype)

        self.classification_head = MambaClassificationHead(
            d_model=config.d_model,
            num_classes=2
        )

        self.device = device

        del self.lm_head

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.backbone(input_ids)
        mean_hidden_state = hidden_states.mean(dim=1)
        logits = self.classification_head(mean_hidden_state)

        if labels is None:
            ClassificationOutput = namedtuple(
                'ClassificationOutput',
                ['logits']    
            )
            return ClassificationOutput(logits=logits)
        else:
            ClassificationOutput = namedtuple(
                'ClassificationOutput',
                ['losses', 'logits']    
            )

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return ClassificationOutput(loss=loss, logits=logits)

    def predict(self, text, tokenizer, id2label=None):
        inputs = torch.tensor(tokenizer(text)['input_ids'], device=self.device)[None]
        
        with torch.no_grad():
            logits = self.forward(input).logits[0]
            label = np.argmax(logits.cpu().numpy())

        if id2label:
            return id2label[label]
        else:
            return label

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)        
        config = MambaConfig(**config_data)

        model = cls(config, device=device, dtype=dtype)

        model_state_dict = load_state_dict_hf(
            pretrained_model_name,
            device=device,
            dtype=dtype
        )

        model.load_state_dict(model_state_dict, strict=False)

        print (" Newly initialized embedding :", 
              set(model.state_dict().keys()) - set(model_state_dict.keys())
        )

        return model

class MambaTrainer(Trainer):
    def __init__(self, **kwargs):        
        super().__init__(**kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop('input_ids')
        labels = inputs.pop('labels')
        
        outputs = self.model(input_ids=input_ids, labels=labels)

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir

        if not os.path.exists(output_dir):
            os.makedirs ( output_dir )

        torch.save(self.model.state_dict(), f'{output_dir}/pytorch_model.bin')

        self.tokenizer.save_pretrained(output_dir)

        with open(f'{output_dir}/config.json', 'w') as f:
            json.dump(self.model.config.to_dict(), f)

        









