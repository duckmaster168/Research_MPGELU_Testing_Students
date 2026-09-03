import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MPGELU(nn.Module):
    """
    Proposed: Modified Parametric Gaussian Error Linear Unit (MP-GELU)
    Formula: f(x) = 0.5 * x * (1 + erf(lambda * x / sqrt(2)))
    where lambda(s) = 1 + softplus(s), strictly enforcing lambda > 1.
    """
    def __init__(self, s_param: float = 0.0, use_softplus: bool = True):
        super(MPGELU, self).__init__()
        self.s_param = nn.Parameter(torch.tensor(s_param, dtype=torch.float32))
        self.use_softplus = use_softplus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_softplus:
            lam = 1.0 + F.softplus(self.s_param)
        else:
            lam = 1.0 + torch.log1p(torch.exp(self.s_param))
            
        output = torch.mul(0.5 * x, (1.0 + torch.erf(torch.mul(lam, x) / np.sqrt(2))))
        return output

    def get_lambda(self) -> float:
        with torch.no_grad():
            if self.use_softplus:
                return (1.0 + F.softplus(self.s_param)).item()
            return (1.0 + torch.log1p(torch.exp(self.s_param))).item()


class PGELU(nn.Module):
    """
    Baseline: Parametric GELU (P-GELU)
    Labied et al. (2025) implementation using tanh approximation[cite: 3].
    """
    def __init__(self, alpha_param: float = 1.0, beta_param: float = 0.04):
        super(PGELU, self).__init__()
        self.alpha_param = nn.Parameter(torch.tensor(alpha_param, dtype=torch.float32))
        self.beta_param = nn.Parameter(torch.tensor(beta_param, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inner = torch.mul(self.alpha_param, x) + torch.mul(self.beta_param, torch.pow(x, 3))
        output = torch.mul(x, (1.0 + torch.tanh(inner)))
        return output


class LambdaGELU(nn.Module):
    """
    Baseline: Lambda-GELU
    Pérez-Corral et al. (2026) implementation using softplus reparameterization with temperature.
    """
    def __init__(self, s_param: float = 0.0, temperature: float = 0.1):
        super(LambdaGELU, self).__init__()
        self.s_param = nn.Parameter(torch.tensor(s_param, dtype=torch.float32))
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = 1.0 + F.softplus(self.s_param / self.temperature)
        output = torch.mul(0.5 * x, (1.0 + torch.erf(torch.mul(lam, x) / np.sqrt(2))))
        return output

    def get_lambda(self) -> float:
        with torch.no_grad():
            return (1.0 + F.softplus(self.s_param / self.temperature)).item()


def get_activation(act_name: str):
    """Factory function to build any requested baseline or proposed activation function."""
    act_name = act_name.lower().replace("_", "").replace("-", "")
    
    if act_name == "mpgelu":
        return MPGELU
    elif act_name == "pgelu":
        return PGELU
    elif act_name == "lambdagelu":
        return LambdaGELU
    elif act_name == "gelu":
        return nn.GELU
    elif act_name == "relu":
        return nn.ReLU
    elif act_name == "leakyrelu":
        return lambda: nn.LeakyReLU(negative_slope=0.01)
    else:
        raise ValueError(f"Unknown activation function name: '{act_name}'")