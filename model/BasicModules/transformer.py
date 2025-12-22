import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.functionals import attention


class BasicTransformer(nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int=None,
        v_dim: int=None,
        heads: int=8,
        use_norm=True,
        use_bias=False,
    ):
        super().__init__()
        self.q_dim = q_dim
        self.k_dim = q_dim if k_dim is None else k_dim
        self.v_dim = q_dim if v_dim is None else v_dim

        self.heads = heads

        if use_norm:
            self.norm_q = nn.LayerNorm(self.q_dim)
            self.norm_k = nn.LayerNorm(self.k_dim)
            self.norm_v = nn.LayerNorm(self.v_dim)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
            self.norm_v = nn.Identity()


        self.to_q = nn.Linear(self.q_dim, self.q_dim, bias=use_bias)
        self.to_k = nn.Linear(self.k_dim, self.k_dim, bias=use_bias)
        self.to_v = nn.Linear(self.v_dim, self.v_dim, bias=use_bias)

        self.to_out = nn.Linear(self.q_dim, self.q_dim)

        if use_norm:
            self.norm_out = nn.LayerNorm(self.q_dim)
        else:
            self.norm_out = nn.Identity()

        return
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = self.norm_q(query)
        key = self.norm_k(key)
        value = self.norm_v(value)

        query = query.reshape(*query.shape[:2], self.heads, -1)
        query = query.transpose(1, 2)

        key = key.reshape(*key.shape[:2], self.heads, -1)
        key = key.transpose(1, 2)

        value = value.reshape(*value.shape[:2], self.heads, -1)
        value = value.transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, 
            key, 
            value, 
            attn_mask=attention_mask
        )

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(*hidden_states.shape[:2], -1)

        hidden_states = self.to_out(hidden_states)

        return self.norm_out(hidden_states)
