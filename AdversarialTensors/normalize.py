#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 17:15:33 2023

@author: cagri
"""
import torch.nn as nn
import torch
class Normalize(nn.Module):
    """
    A PyTorch Module to normalize an input tensor using mean and standard deviation.

    Attributes
    ----------
    mean : torch.Tensor
        The mean values for each channel.
    std : torch.Tensor
        The standard deviation values for each channel.
    """
    def __init__(self, mean, std):
        """
        Initialize the Normalize module.

        Parameters
        ----------
        mean : list or tuple
            The mean values for each channel.
        std : list or tuple
            The standard deviation values for each channel.
        """
        super(Normalize, self).__init__()
        self.register_buffer('mean',
                             torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std',
                             torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        """
        Executes the normalization on the given tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x : torch.Tensor
            Normalized tensor.
        """
        x = x - self.mean
        x = x / self.std
        return x