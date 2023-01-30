import os
from typing import Optional, Union
import wandb
from transformers import PegasusForConditionalGeneration
import torch
import pytorch_lightning as pl
from transformers.modeling_outputs import Seq2SeqLMOutput

from nn_mixed_embedding import MixedEmbedding


class PegasusSoftForConditionalGeneration(PegasusForConditionalGeneration):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.train(False)  #  freeze the model
        hard_embeddings: torch.nn.Embedding = model.get_input_embeddings()  # noqa
        mixed_embeddings = MixedEmbedding(hard_embeddings, wandb.config['n_soft_tokens'], model.config.d_model)
        model.set_input_embeddings(mixed_embeddings)
        model.__class__ = cls
        return model


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[tuple, Seq2SeqLMOutput]:
        B = input_ids.shape[0]
        S = self.get_input_embeddings().soft_embeddings.weight.shape[0]
        # this is just to make the API happy (the problem pos embeddings)
        input_ids = torch.cat([torch.full((B, S), 0).to(input_ids),
                               input_ids], dim=1)  # (B, S);(B, L) -> (B, S + L)
        # must be padded
        if attention_mask is not None:
            attention_mask = torch.cat([torch.full((B, S), 1).to(attention_mask), attention_mask],
                                       dim=1)  # (B, S);(B, L) -> (B, S + L)
        if labels is not None:
            labels = torch.cat([torch.full((B, S), 0).to(labels), labels],
                               dim=1)  # (B, S);(B, L) -> (B, S + L)
        return super().forward(input_ids, attention_mask, labels=labels)

