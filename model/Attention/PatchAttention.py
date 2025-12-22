import torch
import torch.nn as nn
import math


class PatchAttention:
    def __init__(self, hidden_dim: int, patch_size, step_size: int):
        self.hidden_dim = hidden_dim
        self.step_size = step_size
        self.patch_size = patch_size
        self.overlap_size = [patch_size - step_size * i for i in range(1, patch_size // step_size)]
        self.n = None

        return
    
    def prepare_input(self, hidden_states: torch.Tensor, heads: int):
        if hidden_states.ndim == 4:
            b, heads, seq_len, hidden_dim = hidden_states.shape
            hidden_dim = heads * hidden_dim
            hidden_states = hidden_states.transpose(1, 2).reshape(b, seq_len, hidden_dim)

        b, seq_len, hidden_dim = hidden_states.shape
        image_size = int(math.sqrt(seq_len))

        hidden_states = hidden_states.reshape(b, image_size, image_size, hidden_dim).permute(0, 3, 1, 2)

        hidden_states = hidden_states.unfold(-2, self.patch_size, self.step_size)
        hidden_states = hidden_states.unfold(-2, self.patch_size, self.step_size)

        # [b, hidden_dim, n, n, patch_size, patch_size]
        self.n = hidden_states.shape[2] if self.n is None else self.n
        hidden_states = hidden_states.reshape(b, hidden_dim, self.n ** 2, self.patch_size ** 2)

        # [b, hidden_dim, n ** 2, patch_size ** 2]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(*hidden_states.shape[:2], heads, hidden_dim // heads, self.patch_size ** 2)
        hidden_states = hidden_states.reshape(*hidden_states.shape[:3], -1)
        hidden_states = hidden_states.permute(0, 2, 1, 3)

        return hidden_states
    
    def prepare_output(self, hidden_states: torch.Tensor, heads=None):
        # [b, heads, n ** 2, hidden_dim // heads * (patch_size ** 2)]
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(*hidden_states.shape[:2], -1, self.patch_size ** 2)

        # [b, n ** 2, hidden_dim, (patch_size ** 2)]
        b, hidden_dim = hidden_states.shape[0], hidden_states.shape[2]
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        # hidden_states = hidden_states.permute(0, 3, 2, 1)

        image_size = (self.n - 1) * self.step_size + self.patch_size

        # [b, hidden_dim, patch_size * patch_size, n * n]
        hidden_states = hidden_states.reshape(b, -1, self.n ** 2)
        hidden_states = nn.functional.fold(
            hidden_states, 
            output_size=(image_size, image_size),
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.step_size, self.step_size)
        )

        weight = torch.ones(b, hidden_dim, image_size, image_size, device=hidden_states.device)
        weight = nn.functional.unfold(weight, kernel_size=(self.patch_size, self.patch_size), stride=(self.step_size, self.step_size))
        weight = nn.functional.fold(
            weight, 
            output_size=(image_size, image_size), 
            kernel_size=(self.patch_size, self.patch_size), 
            stride=(self.step_size, self.step_size)
        )

        # [b, hidden_dim, image_size, image_size]
        hidden_states /= weight
        hidden_states = hidden_states.reshape(*hidden_states.shape[:2], -1).transpose(-1, -2)

        if heads is not None:
            hidden_states = hidden_states.reshape(*hidden_states.shape[:2], heads, -1).transpose(1, 2)

        return hidden_states


    def __call__(
            self, 
            query: torch.Tensor, 
            key: torch.Tensor, 
            value: torch.Tensor, 
            heads: int=8,
        ):
        heads_already = False
        if query.ndim == 4:
            heads_already = True
            b, heads, seq_len_q, hidden_dim = query.shape
            seq_len_kv = key.shape[2]
            hidden_dim = heads * hidden_dim

            query = query.transpose(1, 2).reshape(b, seq_len_q, hidden_dim)
            key = key.transpose(1, 2).reshape(b, seq_len_kv, hidden_dim)
            value = value.transpose(1, 2).reshape(b, seq_len_kv, hidden_dim)
        else:
            # input has shape of [b, seq_len, hidden_dim]
            b, seq_len, hidden_dim = query.shape

        query = self.prepare_input(query, heads)
        key = self.prepare_input(key, heads)
        value = self.prepare_input(value, heads)

        hidden_states = nn.functional.scaled_dot_product_attention(
            query,
            key,
            value
        )

        hidden_states = self.prepare_output(
            hidden_states, heads if heads_already else None
        )

        return hidden_states

if __name__ == '__main__':
    a = torch.rand(1, 8, 64 ** 2, 768 // 8)
    attn = PatchAttention(768, 8, 4)
    b = attn(a, a, a)
    print(b.shape)
    c = attn.prepare_output(attn.prepare_input(a, 8), 8)
    print(torch.equal(a, c))
