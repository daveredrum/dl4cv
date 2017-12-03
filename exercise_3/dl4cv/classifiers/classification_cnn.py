"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - filter_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        # 0, 0, 0, _, _, _, _, _, 0, 0, 0
        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################
        padding_1 = ((height - 1) * stride_conv + kernel_size - height) // 2
        self.conv1 = nn.Conv2d(channels, num_filters, kernel_size, stride=stride_conv, padding=padding_1)
        self.conv1.weight.data.mul_(weight_scale)
        self.pool1 = nn.MaxPool2d(pool, stride_pool)
        feat_size_1 = height // pool

        padding_2 = ((feat_size_1 - 1) * stride_conv + kernel_size - feat_size_1) // 2
        self.conv2 = nn.Conv2d(num_filters, 2 * num_filters, kernel_size, stride=stride_conv, padding=padding_2)
        self.conv2.weight.data.mul_(weight_scale)
        self.pool2 = nn.MaxPool2d(pool, stride_pool)
        feat_size_2 = feat_size_1 // pool

        padding_3 = ((feat_size_2 - 1) * stride_conv + kernel_size - feat_size_2) // 2
        self.conv3 = nn.Conv2d(2 * num_filters, 4 * num_filters, kernel_size, stride=stride_conv, padding=padding_2)
        self.conv3.weight.data.mul_(weight_scale)
        self.pool3 = nn.MaxPool2d(pool, stride_pool)
        feat_size_3 = feat_size_2 // pool
        
        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(4 * num_filters * feat_size_3 * feat_size_3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x).view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = F.relu(x)
        x = self.dropout(self.fc2(x))
        x = F.relu(x)
        x = self.fc3(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
