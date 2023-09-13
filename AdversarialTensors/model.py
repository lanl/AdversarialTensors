import torch
#@author: Manish Bhattarai, Mehmet Cagri
class FinalModel(torch.nn.Module):
    def __init__(self, model, denoiser=None):
        """
        Initializes the FinalModel instance.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to be used for inference.
        denoiser : Denoiser, optional
            The denoising model to be applied to the input. Defaults to None.
        """
        super(FinalModel, self).__init__()
        self.model = model
        self.denoiser = denoiser
        try:
            self.ranks = self.denoiser.ranks
        except:
            self.ranks = None

    def forward(self, x, ranks=None, recon_err=False):
        """
        Forward pass through the model, with optional denoising.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        ranks : list or tuple, optional
            Multi-ranks for the denoiser's Tucker decomposition. Defaults to None.
        recon_err : bool, optional
            Whether to return reconstruction error from the denoiser. Defaults to False.

        Returns
        -------
        x : torch.Tensor
            The output tensor from the model.
        """

        if self.denoiser is not None:
            if self.ranks is None:
                if ranks is None:
                   raise ValueError("Need to pass valid ranks")
                else:
                    self.ranks = ranks
            x = self.denoiser(x,self.ranks, recon_err)
            if recon_err == True:
                x, rec_err = x
                x = self.model(x)
                return x, rec_err
        x = self.model(x)
        return x


