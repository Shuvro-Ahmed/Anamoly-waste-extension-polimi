import torch
import torch.nn as nn
import torch.nn.functional as functional


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for VQ-VAE.

    This module maps continuous encoder outputs to discrete embeddings using nearest neighbor search, computes the
    quantization loss, and implements the straight-through estimator for backpropagation.

    Attributes:
        num_embeddings (int): Number of discrete embeddings in the codebook.
        embedding_dim (int): Dimensionality of each embedding vector.
        commitment_cost (float): Weight for the commitment loss term.
        embedding (nn.Embedding): The learnable codebook storage.
    """

    def __init__(self, num_embeddings = 1024, embedding_dim = 64, commitment_cost = 0.25):
        """
        Initializes the VectorQuantizer with a uniform distribution.

        Args:
            num_embeddings: Number of discrete embeddings in the codebook.
            embedding_dim: Dimensionality of each embedding vector.
            commitment_cost: Weight for the commitment loss term to prevent encoder outputs from fluctuating too much.
        """

        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, z):
        """
        Quantizes the input features z using the codebook.

        Args:
            z: Encoder output of shape (Batch, Channels, Height, Width).

        Returns:
            output (Tuple[torch.Tensor, torch.Tensor]):
                - z_q: Quantized tensor with shape (B, C, H, W).
                - loss: Scalar tensor representing combined quantization and commitment loss.
        """

        # Flatten the input for the nearest neighbor search
        z_flattened = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_flattened.view(-1, self.embedding_dim)

        # Compute squared Euclidean distance between each z vector and embedding vectors
        distances = (torch.sum(z_flattened ** 2, dim = 1, keepdim = True) +
                     torch.sum(self.embedding.weight ** 2, dim = 1) -
                     2 * torch.matmul(z_flattened, self.embedding.weight.t()))

        # Select indices of the closest embeddings in the codebook
        encoding_indices = torch.argmin(distances, dim = 1)

        # Map indices back to embedding vectors and reshape to spatial dimensions
        z_q = self.embedding(encoding_indices).view(z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)

        # Restore original shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # Loss calculation
        loss = functional.mse_loss(z_q.detach(), z) + self.commitment_cost * functional.mse_loss(z_q, z.detach())

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, loss


class VQVAE(nn.Module):
    """
    Vector-Quantized Variational Autoencoder (VQ-VAE).

    Composed of a convolutional encoder, a vector quantizer bottleneck,
    and a transposed convolutional decoder.

    Attributes:
        encoder (nn.Sequential): Downsampling network.
        quantizer (VectorQuantizer): Discrete bottleneck layer.
        decoder (nn.Sequential): Upsampling reconstruction network.
    """

    def __init__(self, in_channels = 3, embedding_dim = 64, num_embeddings = 1024):
        """
        Initializes the VQ-VAE architecture.

        Args:
            in_channels: Number of input image channels.
            embedding_dim: Dimensionality of latent embeddings.
            num_embeddings: Number of discrete codebook entries.
        """

        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, embedding_dim, 3, stride = 1, padding = 1)
        )

        # Bottleneck
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 256, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, in_channels, 3, stride = 1, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Processes the input through the VQ-VAE pipeline.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            output (Tuple[torch.Tensor, torch.Tensor]):
                - x_recon: Reconstructed image tensor.
                - q_loss: Quantization loss from the bottleneck.
        """

        # Encode image to continuous latent space
        z_e = self.encoder(x)

        # Quantize latent to discrete codebook vectors
        z_q, q_loss = self.quantizer(z_e)

        # Decode quantized latent back to image space
        x_recon = self.decoder(z_q)

        return x_recon, q_loss
