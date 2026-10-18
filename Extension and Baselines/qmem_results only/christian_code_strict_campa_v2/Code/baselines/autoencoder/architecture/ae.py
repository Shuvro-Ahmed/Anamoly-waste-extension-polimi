import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    """
    Basic convolutional block.

    This block performs two successive convolution operations, each followed by a batch normalization.
    It maintains the spatial dimensions of the input via padding.

    Attributes:
        conv (nn.Sequential): Sequential container holding the double convolution layers, batch norms, and activations.
    """

    def __init__(self, input_channels: int, output_channels: int):
        """
        Initializes a convolutional block with double convolution layers.

        Args:
            input_channels (int): Number of channels in the input tensor.
            output_channels (int): Number of channels produced by the convolutions.
        """

        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, padding = 1),
                                  nn.BatchNorm2d(output_channels), nn.ReLU(inplace = True),
                                  nn.Conv2d(output_channels, output_channels, 3, padding = 1),
                                  nn.BatchNorm2d(output_channels), nn.ReLU(inplace = True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C_in, H, W].

        Returns:
            torch.Tensor: Processed tensor of shape [B, C_out, H, W].
        """

        return self.conv(x)


class UNetAE(nn.Module):
    """U-Net Autoencoder model for image reconstruction and anomaly detection.

        Consists of an encoder to capture context and a decoder that enables precise localization through skip
        connections.

        Attributes:
            c1, c2, c3, c4, c5 (ConvBlock): Encoder convolutional blocks.
            p1, p2, p3, p4 (nn.MaxPool2d): Downsampling layers.
            u6, u7, u8, u9 (nn.ConvTranspose2d): Upsampling layers.
            c6, c7, c8, c9 (ConvBlock): Decoder convolutional blocks.
            outc (nn.Conv2d): Final 1x1 convolution to map to output color channels.
        """

    def __init__(self, base_filters = 32):
        """
        Initializes the U-Net Autoencoder.

        Args:
            base_filters (int): Base number of filters (multiplied by 2 at each level).
        """

        super().__init__()

        # Encoder
        self.c1 = ConvBlock(4, base_filters)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = ConvBlock(base_filters, base_filters * 2)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.p3 = nn.MaxPool2d(2)
        self.c4 = ConvBlock(base_filters * 4, base_filters * 8)
        self.p4 = nn.MaxPool2d(2)
        self.c5 = ConvBlock(base_filters * 8, base_filters * 16)

        # Decoder
        self.u6 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride = 2)
        self.c6 = ConvBlock(base_filters * 16, base_filters * 8)
        self.u7 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride = 2)
        self.c7 = ConvBlock(base_filters * 8, base_filters * 4)
        self.u8 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride = 2)
        self.c8 = ConvBlock(base_filters * 4, base_filters * 2)
        self.u9 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride = 2)
        self.c9 = ConvBlock(base_filters * 2, base_filters)
        self.outc = nn.Conv2d(base_filters, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.

        Args:
            x (torch.Tensor): Input tensor containing masked image and mask [B, 4, H, W].

        Returns:
            output (torch.Tensor): Reconstructed RGB image [B, 3, H, W] with Sigmoid activation.
        """

        c1 = self.c1(x)
        p1 = self.p1(c1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        c3 = self.c3(p2)
        p3 = self.p3(c3)
        c4 = self.c4(p3)
        p4 = self.p4(c4)
        c5 = self.c5(p4)

        u6 = self.u6(c5)
        x6 = self.c6(torch.cat([u6, c4], dim = 1))
        u7 = self.u7(x6)
        x7 = self.c7(torch.cat([u7, c3], dim = 1))
        u8 = self.u8(x7)
        x8 = self.c8(torch.cat([u8, c2], dim = 1))
        u9 = self.u9(x8)
        x9 = self.c9(torch.cat([u9, c1], dim = 1))

        return torch.sigmoid(self.outc(x9))
