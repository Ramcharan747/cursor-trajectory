import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.act(out)

class TrajectoryCVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for 2D Cursor Trajectories.
    Takes a fixed-length sequence (L=64) of (x,y) points.
    Conditioned on the displacement vector (dx, dy).
    """
    def __init__(self, seq_len=64, latent_dim=64, cond_dim=2):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Encoder: (Batch, 2, 64) -> (Batch, latent_dim*2)
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=4, stride=2, padding=1), # (B, 32, 32)
            nn.BatchNorm1d(32),
            nn.GELU(),
            ResidualBlock1D(32, 64),
            nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=1), # (B, 64, 16)
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualBlock1D(64, 128),
            nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1), # (B, 128, 8)
            nn.BatchNorm1d(128),
            nn.GELU(),
            ResidualBlock1D(128, 256),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1), # (B, 256, 4)
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        
        self.fc_mu = nn.Linear(256 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4, latent_dim)
        
        # Decoder: (Batch, latent_dim + cond_dim) -> (Batch, 2, 64)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256 * 4),
            nn.GELU()
        )
        
        self.decoder_cnn = nn.Sequential(
            ResidualBlock1D(256, 256),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1), # (B, 128, 8)
            nn.BatchNorm1d(128),
            nn.GELU(),
            ResidualBlock1D(128, 128),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1), # (B, 64, 16)
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualBlock1D(64, 64),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1), # (B, 32, 32)
            nn.BatchNorm1d(32),
            nn.GELU(),
            ResidualBlock1D(32, 32),
            nn.ConvTranspose1d(32, 2, kernel_size=4, stride=2, padding=1), # (B, 2, 64)
        )

    def encode(self, x):
        # x shape: (B, 64, 2) -> (B, 2, 64)
        x = x.transpose(1, 2)
        h = self.encoder_cnn(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Prevent NaN explosions
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        # Concatenate latent vector and condition (displacement)
        h = torch.cat([z, cond], dim=-1)
        h = self.decoder_fc(h)
        h = h.view(h.size(0), 256, 4)
        out = self.decoder_cnn(h)
        # out shape: (B, 2, 64) -> (B, 64, 2)
        return out.transpose(1, 2)

    def forward(self, x, cond):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, cond)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, kl_weight=0.01):
        """
        Computes the CVAE loss: MSE Reconstruction + KL Divergence
        """
        # Reconstruction loss (Mean Squared Error)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        
        # KL Divergence
        # D_KL(Q(z|X) || P(z|X)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
