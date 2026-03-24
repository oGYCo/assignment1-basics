import torch
import torch.nn as nn

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