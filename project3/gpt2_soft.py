import torch
from typing import Optional, Union, Tuple
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.modeling_outputs import Seq2SeqLMOutput


class GPT2SoftConfig(GPT2Config):
    def __init__(self, n_soft_tokens: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.n_soft_tokens = n_soft_tokens


class GPT2SoftLMHeadModel(GPT2LMHeadModel):  # noqa

    def __init__(self, config: GPT2SoftConfig):
        super().__init__(config)
        # --- freeze the model --- #
        for param in self.parameters():
            param.requires_grad = False
        # --- this is the only layer to optimize for --- #
        self.soft_embeddings = torch.nn.Embedding(config.n_soft_tokens, config.n_embd)  # (S, H)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                *args, **kwargs) -> Union[Tuple, Seq2SeqLMOutput]:
        B = input_ids.shape[0]
        S = self.config.n_soft_tokens
        hard_embeds = self.get_input_embeddings()(input_ids) # (B, L) ->  (B, L, H)
        soft_embeds = self.soft_embeddings.weight.expand(input_ids.shape[0], -1, -1)  # (S, H) -> (B, S, H)
        mixed_embeds = torch.cat([hard_embeds, soft_embeds], dim=1)  # (B, L, H);(B, S, H) -> (B, L + S, H)
        # --- attention_mask & labels must be padded accordingly --- #
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.full((B, S), 1).to(attention_mask)],
                                       dim=1)  # (B, L);(B, S) -> (B, L + S)
        if labels is not None:
            labels = torch.cat([labels, torch.full((B, S), self.config.eos_token_id).to(labels)],
                               dim=1)  # (B, L);(B, S) -> (B, L + S)
        # we can directly feed pre-computed embeddings (inputs_embeds) to forward
        return super().forward(inputs_embeds=mixed_embeds,  # noqa
                               attention_mask=attention_mask,
                               labels=labels, *args, **kwargs)
