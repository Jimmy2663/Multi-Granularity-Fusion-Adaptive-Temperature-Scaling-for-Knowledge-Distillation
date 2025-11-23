"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn 
import torch.nn.functional as F

class VGG16(nn.Module):
    """Creates the VGG16 architecture.

    Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
    """
    def __init__(self, input_shape: int,  output_shape: int, dropout: float) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=64, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same" ),  
          nn.ReLU(),
          nn.Conv2d(in_channels=64, 
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding="same"),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )
        self.conv_block_2 = nn.Sequential(
          nn.Conv2d(64, 128, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(128,128, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.MaxPool2d(2)
        )
        self.conv_block_3 = nn.Sequential(
          nn.Conv2d(128, 256, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(256, 256, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(256, 256, kernel_size=3, padding="same"),
          nn.ReLU(),          
          nn.MaxPool2d(2)
        )
        self.conv_block_4 = nn.Sequential(
          nn.Conv2d(256, 512, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(512, 512, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(512, 512, kernel_size=3, padding="same"),
          nn.ReLU(),          
          nn.MaxPool2d(2)
        )
        self.conv_block_5 = nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(512, 512, kernel_size=3, padding="same"),
          nn.ReLU(),
          nn.Conv2d(512, 512, kernel_size=3, padding="same"),
          nn.ReLU(),          
          nn.MaxPool2d(2)
        )                         
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=512*7*7,
                    out_features=4096
                    ),
          nn.ReLU(),
          nn.Dropout(p=dropout),
          nn.Linear(in_features=4096,
                    out_features=4096
                    ),
          nn.ReLU(),
          nn.Dropout(p=dropout),
          nn.Linear(in_features=4096,
                    out_features=output_shape)           
        )
    
    def forward(self, x: torch.Tensor):
        #print(x.shape)
        x = self.conv_block_1(x)
        #print(x.shape)            
        x = self.conv_block_2(x)
        #print(x.shape)
        x = self.conv_block_3(x)
        #print(x.shape)
        x = self.conv_block_4(x)
        #print(x.shape)
        x = self.conv_block_5(x)
        #print(x.shape)                        
        x = self.classifier(x)
        return x
        # return self.classifier(self.block_2(self.block_1(x))) # <- leverage the benefits of operator fusion



class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation (SE) module for channel-wise feature recalibration.

    Args:
        in_ch (int): Number of input channels.
        squeeze_factor (int): Reduction factor for the squeeze operation.
    """
    def __init__(self, in_ch, squeeze_factor=4):
        super().__init__()
        squeeze_ch = in_ch // squeeze_factor
        self.fc1 = nn.Conv2d(in_ch, squeeze_ch, 1)
        self.fc2 = nn.Conv2d(squeeze_ch, in_ch, 1)
    
    def forward(self, x):
        """
        Forward pass for SE module.

        Args:
            x (Tensor): Input feature map of shape (N, C, H, W).

        Returns:
            Tensor: Channel-wise recalibrated feature map.
        """
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class InvertedResidualConfig:
    """
    Configuration container for an Inverted Residual block.

    Args:
        in_ch (int): Number of input channels.
        kernel (int): Kernel size for depthwise convolution.
        expanded_ch (int): Number of channels after expansion.
        out_ch (int): Number of output channels.
        use_se (bool): Whether to use Squeeze-and-Excitation.
        activation (str): Activation function, 'HS' for Hardswish, 'RE' for ReLU.
        stride (int): Stride for depthwise convolution.
        dilation (int): Dilation for depthwise convolution.
    """
    def __init__(self, in_ch, kernel, expanded_ch, out_ch, use_se, activation, stride, dilation):
        self.in_ch = in_ch
        self.kernel = kernel
        self.expanded_ch = expanded_ch
        self.out_ch = out_ch
        self.use_se = use_se
        self.activation = activation
        self.stride = stride
        self.dilation = dilation

class InvertedResidual(nn.Module):
    """
    Inverted Residual block with optional Squeeze-and-Excitation and activation.

    Args:
        cfg (InvertedResidualConfig): Configuration for the block.
    """
    def __init__(self, cfg: InvertedResidualConfig):
        super().__init__()
        layers = []
        # Choose activation function
        act = nn.Hardswish if cfg.activation == "HS" else nn.ReLU
        # Expansion phase
        if cfg.expanded_ch != cfg.in_ch:
            layers.append(nn.Conv2d(cfg.in_ch, cfg.expanded_ch, 1, bias=False))
            layers.append(nn.BatchNorm2d(cfg.expanded_ch))
            layers.append(act(inplace=True))
        # Depthwise convolution
        layers.append(nn.Conv2d(cfg.expanded_ch, cfg.expanded_ch, cfg.kernel, stride=cfg.stride,
                                padding=cfg.kernel//2, groups=cfg.expanded_ch, bias=False))
        layers.append(nn.BatchNorm2d(cfg.expanded_ch))
        layers.append(act(inplace=True))
        # Squeeze-and-Excitation
        if cfg.use_se:
            layers.append(SqueezeExcite(cfg.expanded_ch))
        # Projection phase
        layers.append(nn.Conv2d(cfg.expanded_ch, cfg.out_ch, 1, bias=False))
        layers.append(nn.BatchNorm2d(cfg.out_ch))
        self.block = nn.Sequential(*layers)
        # Use residual connection if stride == 1 and input/output channels match
        self.use_res_connect = (cfg.stride == 1 and cfg.in_ch == cfg.out_ch)
    
    def forward(self, x):
        """
        Forward pass for Inverted Residual block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        out = self.block(x)
        if self.use_res_connect:
            return x + out
        else:
            return out

class MobileNetV3Large(nn.Module):
    """
    MobileNetV3 Large architecture.

    Args:
        num_classes (int): Number of output classes for the classifier.
    """
    def __init__(self, num_classes=int):
        super().__init__()
        layers = []
        # Initial convolutional layer (stem)
        layers.append(nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.Hardswish(inplace=True))
        # MobileNetV3-Large block configurations
        bneck_cfgs = [
            # in, k, exp, out, se, nl, s, d
            [16, 3, 16, 16, False, "RE", 1, 1],
            [16, 3, 64, 24, False, "RE", 2, 1],
            [24, 3, 72, 24, False, "RE", 1, 1],
            [24, 5, 72, 40, True,  "RE", 2, 1],
            [40, 5, 120, 40, True,  "RE", 1, 1],
            [40, 5, 120, 40, True,  "RE", 1, 1],
            [40, 3, 240, 80, False, "HS", 2, 1],
            [80, 3, 200, 80, False, "HS", 1, 1],
            [80, 3, 184, 80, False, "HS", 1, 1],
            [80, 3, 184, 80, False, "HS", 1, 1],
            [80, 3, 480, 112, True, "HS", 1, 1],
            [112, 3, 672, 112, True, "HS", 1, 1],
            [112, 5, 672, 160, True, "HS", 2, 1],
            [160, 5, 960, 160, True, "HS", 1, 1],
            [160, 5, 960, 160, True, "HS", 1, 1],
        ]
        input_ch = 16
        # Build all Inverted Residual blocks
        for cfg in bneck_cfgs:
            layers.append(InvertedResidual(InvertedResidualConfig(*cfg)))
            input_ch = cfg[3]
        # Final convolutional layers before classifier
        layers.append(nn.Conv2d(input_ch, 960, 1, bias=False))
        layers.append(nn.BatchNorm2d(960))
        layers.append(nn.Hardswish(inplace=True))
        self.features = nn.Sequential(*layers)
        # Global average pooling layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass for MobileNetV3 Large.

        Args:
            x (Tensor): Input tensor of shape (N, 3, H, W).

        Returns:
            Tensor: Output logits of shape (N, num_classes).
        """
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x





