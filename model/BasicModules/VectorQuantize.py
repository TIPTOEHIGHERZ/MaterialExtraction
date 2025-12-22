import torch.nn as nn
import torch


class VectorQuantize(nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        embed_dim: int = 256,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(codebook_size, embed_dim)

        return
    
    def euclidean_distance(self, z_flattened: torch.Tensor):
        embedding = self.codebook.weight

        distance = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - \
            2 * (z_flattened @ embedding.T)
        
        return distance

    def forward(
        self,
        z: torch.Tensor
    ):
        bsz, seq_len = z.shape[:2]

        z_flattened = z.reshape(bsz * seq_len, -1)
        distance = self.euclidean_distance(z_flattened)

        min_encoding_indices = torch.argmin(distance, dim=1) # num_ele
        z_q = self.codebook(min_encoding_indices).reshape(bsz * seq_len, -1)
        z_q = z_q.reshape(bsz, seq_len, z_q.shape[-1])
        
        commitment_loss = self.commitment_cost * torch.mean((z_q.detach() - z) ** 2)
        codebook_loss = torch.mean((z_q - z.detach()) ** 2)

        z_q = z_q.reshape(bsz, seq_len, -1)
        z_q = z + (z_q - z).detach()

        loss_dict = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'loss': codebook_loss + commitment_loss * self.commitment_cost
        }

        return z_q, loss_dict
