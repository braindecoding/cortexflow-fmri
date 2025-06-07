"""
CortexFlow Diffusion Components
Advanced diffusion process for guided neural decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CortexFlowDiffusion(nn.Module):
    """Advanced diffusion model for CortexFlow framework."""
    
    def __init__(self, latent_dim=512, condition_dim=512, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.timesteps = timesteps
        
        # Noise schedule
        self.register_buffer('betas', self._cosine_beta_schedule(timesteps, beta_start, beta_end))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
        # Condition projection
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Noise prediction network
        self.noise_predictor = CortexFlowUNet(
            latent_dim=latent_dim,
            condition_dim=latent_dim,
            time_dim=latent_dim
        )

    def _cosine_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """Cosine noise schedule."""
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)

    def _get_time_embedding(self, timesteps, dim):
        """Sinusoidal time embedding."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x_t, t, condition, guidance_scale=1.0):
        """Reverse diffusion step."""
        batch_size = x_t.size(0)
        
        # Time embedding
        time_emb = self._get_time_embedding(t, self.latent_dim)
        time_emb = self.time_embedding(time_emb)
        
        # Project condition
        cond_proj = self.condition_proj(condition)
        
        # Predict noise
        if guidance_scale > 1.0:
            # Classifier-free guidance
            noise_cond = self.noise_predictor(x_t, cond_proj, time_emb)
            noise_uncond = self.noise_predictor(x_t, torch.zeros_like(cond_proj), time_emb)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = self.noise_predictor(x_t, cond_proj, time_emb)
        
        # Compute previous sample
        alpha_t = self.alphas[t].reshape(-1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, 1)
        beta_t = self.betas[t].reshape(-1, 1)
        
        # Compute x_{t-1}
        pred_x_start = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        pred_x_start = torch.clamp(pred_x_start, -1, 1)
        
        if t[0] > 0:
            alpha_cumprod_prev = self.alphas_cumprod[t - 1].reshape(-1, 1)
            posterior_variance = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
            noise = torch.randn_like(x_t)
            x_prev = torch.sqrt(alpha_cumprod_prev) * pred_x_start + torch.sqrt(1 - alpha_cumprod_prev - posterior_variance) * noise_pred + torch.sqrt(posterior_variance) * noise
        else:
            x_prev = pred_x_start
        
        return x_prev

    def sample(self, condition, guidance_scale=7.5, num_inference_steps=50):
        """Generate samples using DDPM sampling."""
        batch_size = condition.size(0)
        device = condition.device
        
        # Start with noise
        x_t = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Sampling timesteps
        timesteps = torch.linspace(self.timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        for t in timesteps:
            t_batch = t.repeat(batch_size)
            x_t = self.p_sample(x_t, t_batch, condition, guidance_scale)
        
        return x_t

    def compute_loss(self, x_start, condition, t=None):
        """Compute diffusion loss."""
        batch_size = x_start.size(0)
        device = x_start.device
        
        # Sample timesteps
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward process
        x_t = self.q_sample(x_start, t, noise)
        
        # Time embedding
        time_emb = self._get_time_embedding(t, self.latent_dim)
        time_emb = self.time_embedding(time_emb)
        
        # Project condition
        cond_proj = self.condition_proj(condition)
        
        # Predict noise
        noise_pred = self.noise_predictor(x_t, cond_proj, time_emb)
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss

class CortexFlowUNet(nn.Module):
    """U-Net architecture for noise prediction."""
    
    def __init__(self, latent_dim=512, condition_dim=512, time_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, latent_dim)
        
        # Condition and time projections
        self.cond_proj = nn.Linear(condition_dim, latent_dim)
        self.time_proj = nn.Linear(time_dim, latent_dim)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            self._make_layer(latent_dim, latent_dim * 2),
            self._make_layer(latent_dim * 2, latent_dim * 4),
            self._make_layer(latent_dim * 4, latent_dim * 8)
        ])
        
        # Bottleneck
        self.bottleneck = self._make_layer(latent_dim * 8, latent_dim * 8)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            self._make_layer(latent_dim * 16, latent_dim * 4),  # Skip connection
            self._make_layer(latent_dim * 8, latent_dim * 2),   # Skip connection
            self._make_layer(latent_dim * 4, latent_dim)        # Skip connection
        ])
        
        # Output projection
        self.output_proj = nn.Linear(latent_dim * 2, latent_dim)

    def _make_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, x, condition, time_emb):
        # Input processing
        h = self.input_proj(x)
        
        # Add condition and time
        cond = self.cond_proj(condition)
        time = self.time_proj(time_emb)
        h = h + cond + time
        
        # Encoder with skip connections
        skip_connections = []
        for layer in self.encoder_layers:
            h = layer(h)
            skip_connections.append(h)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder_layers):
            skip = skip_connections[-(i+1)]
            h = torch.cat([h, skip], dim=-1)
            h = layer(h)
        
        # Output
        h = torch.cat([h, x], dim=-1)  # Final skip connection
        output = self.output_proj(h)
        
        return output
