#@author: Manish Bhattarai, Mehmet Cagri

import torch
import math
from torch.nn import functional as f

class Patcher(object):
    def __init__(self, patch_size, stride=1, padding=0, dilation=1,channels=3):
        """
        A utility class to handle image patch extraction and merging.

        Parameters
        ----------
        patch_size : int
            The size of the patch to be extracted.
        stride : int, optional
            The stride of the patch. Default is 1.
        padding : int, optional
            The amount of padding to add to the image. Default is 0.
        dilation : int, optional
            The spacing between the kernel points. Default is 1.
        channels : int, optional
            The number of channels in the image. Default is 3.

        Attributes
        ----------
        w : int
            The width of the image.
        h : int
            The height of the image.
        stds : tensor, optional
            The standard deviations of the patches, if calculated.
        means : tensor, optional
            The means of the patches, if calculated.
        """
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.channels  = channels
        self.w = None
        self.h = None

        self.stds = None
        self.means = None


    def extract_patches(self, data):
        """
        Extracts patches from the given image data.

        Parameters
        ----------
        data : torch.Tensor
            The image data from which patches are to be extracted.

        Returns
        -------
        torch.Tensor
            The patches extracted from the image data.
        """
        import torch.nn.functional as F

        batch_size, self.channels, self.w,self.h = data.shape
        data = f.unfold(data, kernel_size=self.patch_size,
                           dilation=self.dilation, padding=self.padding,
                           stride=self.stride)
        patch_count = data.shape[-1]
        data = data.permute(0,2,1)
        data = data.reshape((batch_size, patch_count, self.channels, self.patch_size,self.patch_size))

        # change order of the patches (W,H part)
        data = data.permute(0, 1, 2, 4, 3)
        return data


    def merge_patches(self, data_patches, mode='avg'):
        """
        Merges the extracted patches to form an image.

        Parameters
        ----------
        data_patches : torch.Tensor
            The patches to be merged.
        mode : str, optional
            The merging mode. It can be 'avg' or 'sum'. Default is 'avg'.

        Returns
        -------
        torch.Tensor
            The merged image.
        """
        image_size = (self.w,self.h)
        (b_size,patch_count,channels,patch_size1,patch_size2) = data_patches.shape
        data_patches = data_patches.permute(0, 1, 2, 4, 3)
        data_patches = data_patches.reshape(b_size, patch_count, channels*patch_size1*patch_size2)
        data_patches = data_patches.permute(0, 2, 1)
        data_patches2 = torch.ones_like(data_patches)
        #print(data_patches.shape)
        res_cnt = f.fold(data_patches2, image_size, kernel_size=(patch_size1,patch_size2),
                           dilation=self.dilation, padding=self.padding,
                           stride=self.stride)
        data_patches = f.fold(data_patches, image_size, kernel_size=(patch_size1,patch_size2),
                           dilation=self.dilation, padding=self.padding,
                           stride=self.stride)
        if mode == 'avg':
           data_patches = data_patches / res_cnt

        return data_patches

class patch_transform:
    """
    A utility class for transforming images into patches and vice versa.

    Parameters
    ----------
    patch_size : int
        The size of the patch.
    stride : int, optional
        The stride of the patch. Default is 1.
    padding : int, optional
        The padding of the patch. Default is 0.
    dilation : int, optional
        The dilation of the patch. Default is 1.
    channels : int, optional
        The number of channels in the image. Default is 3.

    Attributes
    ----------
    patcher : Patcher
        A Patcher object configured with the given parameters.
    """
    def __init__(self,  patch_size, stride=1, padding=0, dilation=1,channels=3):
        self.patcher = Patcher( patch_size, stride, padding, dilation,channels)


    def fit(self,x,mode='patch'):
        """
        Extracts patches from the input image or merges patches into an image.

        Parameters
        ----------
        x : torch.Tensor
            The input image to be transformed.
        mode : str, optional
            The mode of transformation. 'patch' to extract patches, 'merge' to merge patches. Default is 'patch'.

        Returns
        -------
        torch.Tensor
            A batch of patches if mode is 'patch', or a merged image if mode is 'merge'.
        """
        if mode == 'patch':
           return self.patcher.extract_patches(x)
        elif mode =='merge':
            return self.patcher.merge_patches(x,mode='avg')
