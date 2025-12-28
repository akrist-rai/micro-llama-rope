import torch
import torch.nn as nn

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d_model, max_seq_len=2048, base=10000.0):
        """
        Implements RoPE (Rotary Positional Embeddings) from scratch.
        
        Args:
            d_model (int): The dimension of the model (must be even).
            max_seq_len (int): The maximum sequence length.
            base (float): The base for the geometric progression of theta.
        """
        super().__init__()
        self.d_model = d_model
        
        # Calculate theta values (the rotation angles)
        # formula: theta_i = 1 / (base ^ (2i / d))
        theta = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        
        # Create position indices [0, 1, ..., max_seq_len-1]
        position_idx = torch.arange(max_seq_len).float()
        
        # Calculate outer product of positions and theta to get the rotation matrix arguments
        # shape: (max_seq_len, d_model / 2)
        idx_theta = torch.outer(position_idx, theta)
        
        # We use polar form (complex numbers) to represent rotation. 
        # e^(i*theta) = cos(theta) + i*sin(theta)
        # This precomputes 'cos' and 'sin' for all positions.
        self.register_buffer('freqs_cis', torch.polar(torch.ones_like(idx_theta), idx_theta))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_heads, head_dim)
        Returns:
            x_rotated: Tensor with RoPE applied.
        """
        batch_size, seq_len, n_heads, head_dim = x.shape
        
        # Ensure x is shaped for complex number operations
        # We view the last dimension as pairs of coordinates (real, imag)
        # shape becomes: (batch_size, seq_len, n_heads, head_dim // 2)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        
        # Slice the precomputed frequencies to the current sequence length
        # shape: (seq_len, head_dim // 2)
        freqs_cis = self.freqs_cis[:seq_len]
        
        # Reshape for broadcasting so we can multiply across batch and heads
        # shape: (1, seq_len, 1, head_dim // 2)
        freqs_cis = freqs_cis.view(1, seq_len, 1, x_complex.shape[-1])
        
        # Apply rotation via complex multiplication
        # (a + ib) * (cos + isin) -> rotates the vector
        x_rotated_complex = x_complex * freqs_cis
        
        # Convert back to real numbers and flatten the last dimension
        x_rotated = torch.view_as_real(x_rotated_complex).flatten(3)
        
        return x_rotated.type_as(x)

if __name__ == "__main__":
    # Simple test to verify shapes
    d_model = 64
    seq_len = 10
    rope = RotaryPositionalEmbeddings(d_model=d_model, max_seq_len=seq_len)
    
    # Dummy input: Batch=2, Seq=10, Heads=4, Dim=64
    x = torch.randn(2, 10, 4, 64)
    y = rope(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert x.shape == y.shape, "Shape mismatch!"
    print("RoPE implementation verified successfully.")
