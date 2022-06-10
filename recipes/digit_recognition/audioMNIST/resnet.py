"""
This file contains a very simple TDNN module to use for speaker-id.

To replace this model, change the `!new:` tag in the hyperparameter file
to refer to a built-in SpeechBrain model or another file containing
a custom PyTorch module.

Authors
 * Nauman Dawalatabad 2020
 * Mirco Ravanelli 2020
"""


import torch  # noqa: F401
import torch.nn as nn
import speechbrain as sb
from speechbrain.nnet.pooling import StatisticsPooling, Pooling1d, AdaptivePool
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import BatchNorm1d


class ResNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        intermediate_channels,
        identity_downsample=None,
        stride=1
    ):

        super(ResNetBlock, self).__init__()
        self.expansion = 4
        self.conv1 = Conv1d(
                          in_channels=in_channels,
                          out_channels=intermediate_channels,
                          kernel_size=1,
                          stride=1)
        self.bn1 = BatchNorm1d(input_size=intermediate_channels)
        
        self.conv2 = Conv1d(
                          in_channels=intermediate_channels,
                          out_channels=intermediate_channels,
                          kernel_size=3,
                          stride=stride)
        self.bn2 = BatchNorm1d(input_size=intermediate_channels)

        self.conv3 = Conv1d(
                          in_channels=intermediate_channels,
                          out_channels=intermediate_channels * self.expansion,
                          kernel_size=1,
                          stride=1)
        self.bn3 = BatchNorm1d(input_size=intermediate_channels * self.expansion)

        self.relu = torch.nn.LeakyReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
          identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNet(torch.nn.Module):
    def __init__(
        self, 
        ResNetBlock,
        layers,
        in_channels,
        num_classes
    ):
        super(ResNet, self).__init__()


        self.in_channels = 64
        self.conv1 = Conv1d(
                        in_channels=in_channels,
                        out_channels=64,
                        kernel_size=7,
                        stride=2
                    )
        self.bn1 = BatchNorm1d(input_size=64)
        self.relu = torch.nn.LeakyReLU()


        self.maxpool = Pooling1d(
                          pool_type="max",
                          kernel_size=3,
                          stride=2)

        self.layer1 = self._make_layer(
            ResNetBlock, layers[0], intermediate_channels=64, stride=1
        )

        self.layer2 = self._make_layer(
            ResNetBlock, layers[1], intermediate_channels=128, stride=1
        )

        self.layer3 = self._make_layer(
            ResNetBlock, layers[2], intermediate_channels=256, stride=1
        )

        self.layer4 = self._make_layer(
            ResNetBlock, layers[3], intermediate_channels=512, stride=1
        )

        self.avgpool = AdaptivePool(1)
        self.fc = Linear(
                input_size=512 * 4,
                n_neurons=num_classes
            )

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # import pdb ; pdb.set_trace()
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

        

    def _make_layer(self, ResNetBlock, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        rb_layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                Conv1d(
                        in_channels=self.in_channels,
                        out_channels=intermediate_channels * 4,
                        kernel_size=1,
                        stride=stride,
                    ),
                BatchNorm1d(input_size=intermediate_channels * 4),
            )

        rb_layers.append(
            ResNetBlock(
                self.in_channels,
                intermediate_channels,
                identity_downsample=identity_downsample,
                stride=stride
            )
        )

        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            rb_layers.append(
                ResNetBlock(
                    self.in_channels,
                    intermediate_channels
                )
            )

        return Sequential(*rb_layers)
        

class ResNet50(torch.nn.Module):
    def __init__(
          self,
          device="cpu",
          resnet_layers=[3, 4, 6, 3],
          in_channels=40,
          lin_neurons=512
      ):
          super(ResNet50, self).__init__()
          self.resnet = ResNet(
                ResNetBlock,
                resnet_layers,
                in_channels,
                lin_neurons)

        
    def forward(self, x, lens=None):
        x = self.resnet(x)
        return x


class Xvector(torch.nn.Module):
    """This model extracts X-vectors for speaker recognition

    Arguments
    ---------
    activation : torch class
        A class for constructing the activation layers.
    tdnn_blocks : int
        Number of time-delay neural (TDNN) layers.
    tdnn_channels : list of ints
        Output channels for TDNN layer.
    tdnn_kernel_sizes : list of ints
        List of kernel sizes for each TDNN layer.
    tdnn_dilations : list of ints
        List of dilations for kernels in each TDNN layer.
    lin_neurons : int
        Number of neurons in linear layers.

    Example
    -------
    >>> compute_xvect = Xvector()
    >>> input_feats = torch.rand([5, 10, 40])
    >>> outputs = compute_xvect(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 512])
    """

    def __init__(
        self,
        device="cpu",
        activation=torch.nn.LeakyReLU,
        tdnn_blocks=5,
        tdnn_channels=[512, 512, 512, 512, 1500],
        tdnn_kernel_sizes=[5, 3, 3, 1, 1],
        tdnn_dilations=[1, 2, 3, 1, 1],
        lin_neurons=512,
        in_channels=40,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        # TDNN has convolutional layers with the given dilation factors
        # and kernel sizes. We here loop over all the convolutional layers
        # that we wanna add. Note that batch normalization is used after
        # the activations function in this case. This improves the
        # speaker-id performance a bit.
        for block_index in range(tdnn_blocks):
            out_channels = tdnn_channels[block_index]
            self.blocks.extend(
                [
                    Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=tdnn_kernel_sizes[block_index],
                        dilation=tdnn_dilations[block_index],
                    ),
                    activation(),
                    BatchNorm1d(input_size=out_channels),
                ]
            )
            in_channels = tdnn_channels[block_index]

        # Statistical pooling. It converts a tensor of variable length
        # into a fixed-length tensor. The statistical pooling returns the
        # mean and the standard deviation.
        self.blocks.append(StatisticsPooling())

        # Final linear transformation.
        self.blocks.append(
            Linear(
                input_size=out_channels * 2,  # mean + std,
                n_neurons=lin_neurons,
                bias=True,
                combine_dims=False,
            )
        )

    def forward(self, x, lens=None):
        """Returns the x-vectors.

        Arguments
        ---------
        x : torch.Tensor
        """

        for layer in self.blocks:
            try:
                x = layer(x, lengths=lens)
            except TypeError:
                x = layer(x)
        return x


class Classifier(sb.nnet.containers.Sequential):
    """This class implements the last MLP on the top of xvector features.
    Arguments
    ---------
    input_shape : tuple
        Expected shape of an example input.
    activation : torch class
        A class for constructing the activation layers.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of output neurons.

    Example
    -------
    >>> input_feats = torch.rand([5, 10, 40])
    >>> compute_xvect = Xvector()
    >>> xvects = compute_xvect(input_feats)
    >>> classify = Classifier(input_shape=xvects.shape)
    >>> output = classify(xvects)
    >>> output.shape
    torch.Size([5, 1, 1211])
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=512,
        out_neurons=1211,
    ):
        super().__init__(input_shape=input_shape)

        self.append(activation(), layer_name="act")
        self.append(sb.nnet.normalization.BatchNorm1d, layer_name="norm")

        if lin_blocks > 0:
            self.append(sb.nnet.containers.Sequential, layer_name="DNN")

        # Adding fully-connected layers
        for block_index in range(lin_blocks):
            block_name = f"block_{block_index}"
            self.DNN.append(
                sb.nnet.containers.Sequential, layer_name=block_name
            )
            self.DNN[block_name].append(
                sb.nnet.linear.Linear,
                n_neurons=lin_neurons,
                bias=True,
                layer_name="linear",
            )
            self.DNN[block_name].append(activation(), layer_name="act")
            self.DNN[block_name].append(
                sb.nnet.normalization.BatchNorm1d, layer_name="norm"
            )

        # Final Softmax classifier
        self.append(
            sb.nnet.linear.Linear, n_neurons=out_neurons, layer_name="out"
        )
        self.append(
            sb.nnet.activations.Softmax(apply_log=True), layer_name="softmax"
        )