import torch
import torch.nn as nn
import torch.nn.functional as funct


class VectorQuantizerEMA(nn.Module):
    """
    Exponential Moving Average vector quantizer for VQ-VAE.

    This module quantizes continuous latent vectors into a discrete codebook representation.
    It uses Exponential Moving Average to update the codebook vectors.

    Attributes:
        num_embeddings (int): Number of discrete vectors in the codebook.
        embedding_dim (int): Dimensionality of each codebook vector.
        decay (float): The decay factor for the EMA updates (usually 0.9 to 0.99).
        epsilon (float): Small constant used for numerical stability in division.
        commitment_cost (float): Weight for the commitment loss (beta).
        embedding (nn.Parameter): The actual codebook weight matrix.
    """

    def __init__(self, num_embeddings = 64, embedding_dim = 64, decay = 0.99, epsilon = 1e-5, commitment_cost = 0.25):
        """
        Initializes the VectorQuantizerEMA module.

        Args:
            num_embeddings (int): Number of discrete vectors in the codebook.
            embedding_dim (int): Dimensionality of each codebook vector.
            decay (float): The decay factor for EMA updates.
            epsilon (float): Small constant for numerical stability.
            commitment_cost (float): Weight for the commitment loss.
        """

        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon
        self.commitment_cost = commitment_cost
        self.embedding = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim))
        self.register_buffer('cluster_size', torch.zeros(self.num_embeddings))
        self.register_buffer('ema_w', torch.empty(self.num_embeddings, self.embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets parameters and buffers for the vector quantization module.
        """

        # Initialize embeddings with uniform distribution
        nn.init.uniform_(self.embedding, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

        # Sync the Exponential Moving Average weights with the initial embedding state
        self.ema_w.data.copy_(self.embedding.data)
        self.cluster_size.zero_()

    def forward(self, z):
        """
        Applies vector quantization and EMA updates to the input.

        Args:
            z (torch.Tensor): Input latent tensor of shape (B, C, H, W).

        Returns:
            output (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                - z_q_st: Quantized latent tensor (B, C, H, W).
                - commitment_loss: Scalar commitment loss.
                - perplexity: Scalar codebook utilization metric.
        """

        ## Reshape for distance calculation
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z_perm.view(-1, self.embedding_dim)

        # Compute Euclidean distance
        emb = self.embedding
        z_sq = (z_flat ** 2).sum(dim = 1, keepdim = True)
        emb_sq = (emb ** 2).sum(dim = 1)
        distances = z_sq - 2 * (z_flat @ emb.t()) + emb_sq.unsqueeze(0)

        # Map each input vector to the closest codebook vector
        encoding_indices = torch.argmin(distances, dim = 1)
        encodings = funct.one_hot(encoding_indices, self.num_embeddings).type(z_flat.dtype)

        # Quantize latent and restore original spatial shape
        z_q_flat = encodings @ emb
        z_q = z_q_flat.view(z_perm.shape).permute(0, 3, 1, 2).contiguous()

        # Update codebook
        if self.training:
            # Calculate current batch usage
            cluster_size = encodings.sum(dim = 0)

            # Apply Exponential Moving Average smoothing to usage and weights
            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha = (1 - self.decay))
            embed_sum = encodings.t() @ z_flat
            self.ema_w.data.mul_(self.decay).add_(embed_sum, alpha = (1 - self.decay))

            # Normalize and update the embedding parameters
            n = self.cluster_size.sum()
            cluster_size_normalized = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            cluster_size_normalized = torch.clamp(cluster_size_normalized, min = self.epsilon)
            updated_emb = self.ema_w / cluster_size_normalized.unsqueeze(1)
            self.embedding.data.copy_(updated_emb)

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # Commitment loss computation
        commitment_loss = self.commitment_cost * funct.mse_loss(z_q.detach(), z)

        # Compute perplexity
        avg_probs = encodings.mean(dim = 0)
        perplexity = torch.exp(- (avg_probs * torch.log(avg_probs + 1e-10)).sum())

        return z_q_st, commitment_loss, perplexity


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and GroupNorm.

    Args:
        channels (int): Number of input and output channels.
        groups (int): Number of groups for GroupNorm. Defaults to 8.
    """

    def __init__(self, channels, groups = 8):
        """
        Initializes the ResidualBlock.

        Args:
            channels (int): Number of input and output channels.
            groups (int, optional): Number of groups for GroupNorm. Defaults to 8.
        """

        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1),
                                 nn.GroupNorm(groups, channels),
                                 nn.ReLU(inplace = True),
                                 nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1),
                                 nn.GroupNorm(groups, channels))

    def forward(self, x):
        """
        Applies the residual block to the input tensor.

        Args:
            x (torch.Tensor): input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: output tensor of the same shape as the input.
        """

        return funct.relu(x + self.net(x))


class Encoder(nn.Module):
    """
    Convolutional encoder for VQ-VAE.

    Args:
        in_channels (int): Number of input image channels.
        hidden_channels (int): Channels for intermediate layers.
        embedding_dim (int): Channels in the output latent map.
    """

    def __init__(self, in_channels, hidden_channels, embedding_dim):
        """
        Initializes the Encoder.

        Args:
            in_channels (int): Number of input image channels.
            hidden_channels (int): Number of channels in hidden layers.
            embedding_dim (int): Number of channels in the output embedding.
        """

        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 4, 2, 1),
                                 nn.ReLU(inplace = True),
                                 nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),
                                 nn.ReLU(inplace = True),
                                 ResidualBlock(hidden_channels),
                                 nn.Conv2d(hidden_channels, embedding_dim, kernel_size = 3, stride = 1, padding = 1))

    def forward(self, x):
        """
        Encodes image to latent space.

        Args:
            x (torch.Tensor): Image tensor (B, in_channels, H, W).

        Returns:
            output (torch.Tensor): Latent tensor (B, embedding_dim, H/4, W/4).
        """

        return self.net(x)


class Decoder(nn.Module):
    """
    Convolutional decoder for VQ-VAE.

    Args:
        in_channels (int): Channels of the input latent map.
        hidden_channels (int): Internal hidden width.
        out_channels (int): Output channels.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Initializes the Decoder.

        Args:
            in_channels (int): Channels of the input latent map.
            hidden_channels (int): Internal hidden width.
            out_channels (int): Output channels.
        """

        super().__init__()
        self.net = nn.Sequential(ResidualBlock(in_channels),
                                 nn.ConvTranspose2d(in_channels, hidden_channels, 4, 2, 1),
                                 nn.ReLU(inplace = True),
                                 nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1),
                                 nn.ReLU(inplace = True),
                                 nn.Conv2d(hidden_channels // 2, out_channels, 3, 1, 1))

    def forward(self, x):
        """
        Decodes latent features back to image/feature space.

        Args:
            x (torch.Tensor): Latent tensor (B, in_channels, H, W).

        Returns:
            output (torch.Tensor): Upsampled tensor (B, out_channels, H*4, W*4).
        """

        return self.net(x)


class VQVAE2(nn.Module):
    """
    Hierarchical VQ-VAE-2 with Top and Bottom latent maps.

    Args:
        in_channels (int): Input image channels. Defaults to 3.
        hidden_channels (int): Channels for hidden layers. Defaults to 128.
        embedding_dim (int): Codebook vector dimension. Defaults to 64.
        num_embeddings (int): Total codebook size. Defaults to 1024.
        decay (float): EMA decay rate. Defaults to 0.99.
        commitment_cost (float): Commitment loss weight. Defaults to 0.25.
    """

    def __init__(self, in_channels = 3, hidden_channels = 128, embedding_dim = 64, num_embeddings = 1024, decay = 0.99,
                 commitment_cost = 0.25):
        """
        Initializes the hierarchical VQ-VAE-2.

        Args:
            in_channels (int): Input image channels. Defaults to 3.
            hidden_channels (int): Channels for hidden layers. Defaults to 128.
            embedding_dim (int): Codebook vector dimension. Defaults to 64.
            num_embeddings (int): Total codebook size. Defaults to 1024.
            decay (float): EMA decay rate. Defaults to 0.99.
            commitment_cost (float): Commitment loss weight. Defaults to 0.25.
        """

        super().__init__()
        self.encoder_bottom = Encoder(in_channels, hidden_channels, embedding_dim)
        self.encoder_top = Encoder(embedding_dim, hidden_channels, embedding_dim)
        self.quantizer_top = VectorQuantizerEMA(num_embeddings, embedding_dim, decay = decay,
                                                commitment_cost = commitment_cost)
        self.quantizer_bottom = VectorQuantizerEMA(num_embeddings, embedding_dim, decay = decay,
                                                   commitment_cost = commitment_cost)
        self.pre_quant_bottom = nn.Conv2d(embedding_dim * 2, embedding_dim, kernel_size = 1)
        self.decoder_top = Decoder(embedding_dim, hidden_channels, embedding_dim)
        self.decoder_final = Decoder(embedding_dim * 2, hidden_channels, out_channels = in_channels)

    def forward(self, x):
        """
        Forward pass for hierarchical reconstruction.

        Args:
            x (torch.Tensor): Input image batch (B, in_channels, H, W).

        Returns:
            output (Tuple[torch.Tensor, torch.Tensor]):
                - recon: Reconstructed image tensor.
                - vq_loss: Combined quantization loss from both levels.
        """

        # Encode bottom-level features
        z_bottom = self.encoder_bottom(x)

        # Encode top-level features
        z_top = self.encoder_top(z_bottom)

        # Quantize top-level latent
        z_q_top, q_loss_top, perplexity_top = self.quantizer_top(z_top)

        # Decode the top to the bottom resolution
        dec_top = self.decoder_top(z_q_top)

        # Ensure top decoding matches bottom feature spatial size
        if dec_top.shape[2:] != z_bottom.shape[2:]:
            dec_top = funct.interpolate(dec_top, size = z_bottom.shape[2:], mode = 'nearest')

        # Combine bottom features with top decoding
        z_bottom_combined = torch.cat([z_bottom, dec_top], dim = 1)
        z_bottom_combined = self.pre_quant_bottom(z_bottom_combined)

        # Quantize bottom-level latent
        z_q_bottom, q_loss_bottom, perplexity_bottom = self.quantizer_bottom(z_bottom_combined)

        # Prepare for final decoding
        top_for_decode = dec_top
        if top_for_decode.shape[2:] != z_q_bottom.shape[2:]:
            top_for_decode = funct.interpolate(top_for_decode, size = z_q_bottom.shape[2:], mode = 'nearest')

        # Concatenate bottom quantized latent with top conditioning
        z_combined = torch.cat([z_q_bottom, top_for_decode], dim = 1)

        # Reconstruct image
        recon = self.decoder_final(z_combined)
        recon = torch.sigmoid(recon)

        # Aggregate vector quantization losses
        vq_loss = q_loss_top + q_loss_bottom

        return recon, vq_loss
