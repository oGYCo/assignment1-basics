import torch
import torch.nn as nn

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    x = x - x.max(dim=i, keepdim=True).values
    x = x.exp()
    return x / x.sum(dim=i, keepdim=True)

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    d_k = query.shape[-1]
    scores = query @ key.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    attn_weights = softmax(scores, i=-1)
    return attn_weights @ value

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.long()
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        result = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(result ** 2, dim=-1, keepdim=True) + self.eps)
        result = result / rms
        # The output should be in the same dtype as the input, so we convert it back before multiplying by the weight.
        return result.to(in_dtype) * self.weight
        
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device = None, dtype = None):
        """
        Initialize the feed-forward network.

        Args:
            d_model (int): The dimensionality of the input and output.
            d_ff (int): The dimensionality of the hidden layer.
            device: The device to run the network on.
            dtype: The data type of the network.

        SwiGLU: W2 @ (SiLU(W1 @ x) * (W3 @ x))
        """
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden1 = self.linear1(x)
        gate = torch.sigmoid(hidden1) * hidden1
        value = self.linear3(x)
        return self.linear2(gate * value)
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        """
        Args:
            theta (float): The theta parameter for the RoPE embeddings.
            d_k (int): The dimensionality of the query and key vectors.
            max_seq_len (int): The maximum sequence length for which to precompute embeddings.
            device: The device to sotre the buffer on.
        """
        super().__init__()
        self.theta = theta
        assert d_k % 2 == 0
        self.d_k = d_k

        idx = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        inv_freq = 1.0 / (theta ** idx)

        position_ids = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = position_ids[:, None] * inv_freq[None, :]

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.get_buffer("cos")[token_positions].to(dtype=x.dtype, device=x.device)
        sin = self.get_buffer("sin")[token_positions].to(dtype=x.dtype, device=x.device)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_out = x_even * cos - x_odd * sin
        x_odd_out = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = x_even_out
        out[..., 1::2] = x_odd_out
        return out

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device = None, dtype = None, use_rope: bool = False, theta: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta, 
                d_k=self.head_dim, 
                max_seq_len=max_seq_len, 
                device=device
            )

        self.q_proj = Linear(d_model, num_heads * self.head_dim, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, num_heads * self.head_dim, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, num_heads * self.head_dim, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(self, "rope") and token_positions is None:
            token_positions = torch.arange(T, device=x.device)

        q = self.rope(q, token_positions) if hasattr(self, "rope") else q
        k = self.rope(k, token_positions) if hasattr(self, "rope") else k

        mask = torch.tril(torch.ones((T, T), device=x.device)).bool()
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(attn_output)

class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device = None, dtype = None, theta: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads, device=device, dtype=dtype, use_rope=True, theta=theta, max_seq_len=max_seq_len)
        self.ffn = FFN(d_model, d_ff, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

def main():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(softmax(x, i=-1))

if __name__ == "__main__":
    main()