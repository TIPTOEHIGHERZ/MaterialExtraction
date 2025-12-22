import torch
import torch.nn as nn
import math


class FractalAttention:
    n = list()
    def __init__(self, hidden_dim: int, patch_num: int, step_ratio=0.5):
        self.hidden_dim = hidden_dim
        self.step_ratio = step_ratio
        self.patch_num = patch_num
        self.image_size = None

        return
    
    def prepare_input(self, hidden_states: torch.Tensor, heads: int):
        if hidden_states.ndim == 4:
            # [b, heads, seq_len, hidden_dim // heads]
            b, heads, seq_len, hidden_dim = hidden_states.shape
            hidden_dim = heads * hidden_dim
            hidden_states = hidden_states.transpose(1, 2).reshape(b, seq_len, hidden_dim)

        # [b, seq_len, hidden_dim]
        b, seq_len, hidden_dim = hidden_states.shape
        image_size = int(math.sqrt(seq_len))
        patch_size = image_size // self.patch_num
        step_size = max(int(patch_size * self.step_ratio), 1)
        patch_num = 1 + (image_size - patch_size) // step_size

        hidden_states = hidden_states.reshape(b, image_size, image_size, hidden_dim).permute(0, 3, 1, 2)

        hidden_states = hidden_states.unfold(-2, patch_size, step_size)
        hidden_states = hidden_states.unfold(-2, patch_size, step_size)

        # [b, hidden_dim, n, n, patch_size, patch_size]
        hidden_states = hidden_states.reshape(b, heads, hidden_dim // heads, patch_num ** 2, patch_size ** 2)
        hidden_states = hidden_states.permute(0, 1, 3, 2, 4)
        hidden_states = hidden_states.reshape(*hidden_states.shape[:3], -1)

        # [b, heads, n ** 2, hidden_dim // heads * patch_size ** 2]
        return hidden_states
    
    def prepare_next(self, hidden_states: torch.Tensor, hidden_dim: int):
        # [b, heads, n ** 2, hidden_dim // heads * patch_size ** 2]
        b, heads, n_2, d = hidden_states.shape
        patch_size = int(math.sqrt(d * heads // hidden_dim))

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(b * n_2, hidden_dim, patch_size ** 2).transpose(-1, -2)

        # [b * n ** 2, seq_len, hidden_dim]
        return hidden_states

    def prepare_output(self, hidden_states: torch.Tensor, patch_num: int, heads=None):
        # [b * n ** 2, seq_len, hidden_dim]
        b, seq_len, hidden_dim = hidden_states.shape
        patch_size = int(math.sqrt(seq_len))
        b = b // patch_num ** 2
        stride = (max(int(patch_size * self.step_ratio), 1),) * 2

        hidden_states = hidden_states.reshape(b, patch_num ** 2, seq_len, hidden_dim)
        hidden_states = hidden_states.permute(0, 3, 2, 1)

        image_size = self.patch_num * patch_size

        # [b, hidden_dim  * patch_size ** 2, n * n]
        hidden_states = hidden_states.reshape(b, -1, patch_num ** 2)
        hidden_states = nn.functional.fold(
            hidden_states, 
            output_size=(image_size, image_size),
            kernel_size=(patch_size, patch_size),
            stride=stride
        )

        weight = torch.ones(b, hidden_dim, image_size, image_size, device=hidden_states.device)
        weight = nn.functional.unfold(weight, kernel_size=(patch_size, patch_size), stride=stride)
        weight = nn.functional.fold(
            weight, 
            output_size=(image_size, image_size), 
            kernel_size=(patch_size, patch_size), 
            stride=stride
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
        if query.ndim == 4:
            # query has shape [b, heads, image_size ** 2, hidden_dim // heads]
            b, heads, seq_len, hidden_dim = query.shape
            hidden_dim *= heads
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            query = query.reshape(*query.shape[:2], -1)
            key = key.reshape(*key.shape[:2], -1)
            value = value.reshape(*value.shape[:2], -1)
        elif query.ndim == 3:
            b, seq_len, hidden_dim = query.shape
        else:
            raise NotImplementedError(f'query has {query.ndim} dims, which is not supported')

        self.image_size = int(math.sqrt(seq_len))
        depth = int(math.log(self.image_size, self.patch_num))
        
        self.n = list()
        for d in range(depth):
            pre_b = query.shape[0]
            query = self.prepare_input(query, heads)
            key = self.prepare_input(key, heads)
            value = self.prepare_input(value, heads)

            hidden_states = nn.functional.scaled_dot_product_attention(
                query, key, value
            )
            # hidden_states = query

            hidden_states = query = key = value = self.prepare_next(hidden_states, hidden_dim)
            self.n.append(int(math.sqrt(hidden_states.shape[0] // pre_b)))

        for d in range(depth):
            patch_num = self.n.pop(-1)
            hidden_states = self.prepare_output(hidden_states, patch_num)

        hidden_states = hidden_states.reshape(b, seq_len, heads, hidden_dim // heads).transpose(1, 2)

        return hidden_states
        

if __name__ == '__main__':
    a = torch.rand(1, 8, 64 ** 2, 768 // 8)
    attn = FractalAttention(768, 8, 0.5)
    b = attn(a, a, a)
    print(b.shape)
    # print(torch.equal(a, b))
    # print(torch.equal(a, c))
