import torch

class MixedEmbedding(torch.nn.Embedding):
    def __init__(self, hard_embeddings: torch.nn.Embedding, n_soft_tokens: int, embedding_dim: int, **kwargs):
        super().__init__(n_soft_tokens, embedding_dim=embedding_dim, **kwargs)
        self.hard_embeddings = hard_embeddings
        self.soft_embeddings = torch.nn.Embedding(n_soft_tokens, embedding_dim)  # (S, H)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        :param input_ids: (B, S + L)
        :return: (B, S + L, H)
        """
        S = self.soft_embeddings.weight.shape[0]
        hard_embeddings = self.hard_embeddings(input_ids[:, S:])  # (B, S + L) -> (B, L) ->  (B, L, H)
        soft_embeddings = self.soft_embeddings.weight.expand(input_ids.shape[0], -1, -1)  # (S, H) -> (B, S, H)
        return torch.cat([soft_embeddings, hard_embeddings], dim=1)  # (B, S, H);(B, L, H) -> (B, S + L, H)