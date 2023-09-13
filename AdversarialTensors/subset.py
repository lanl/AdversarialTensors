from torch.utils.data import Dataset
from torchvision import transforms


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Parameters
    ----------
    dataset : Dataset
        The whole Dataset
    indices : sequence
        Indices in the whole set selected for subset
    transform : callable, optional
        A function/transform that takes in a sample and returns a transformed version.
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset  # The original dataset
        self.indices = indices  # Indices specifying the subset
        self.transform = transform  # Transformations to apply on images

    def __getitem__(self, idx):
        """
        Fetch a single item from the dataset, apply the transform if any, and return it.

        Parameters
        ----------
        idx : int
            Index to fetch from the subset.

        Returns
        -------
        tuple
            Tuple containing transformed image and corresponding label.
        """

        im, labels = self.dataset[
            self.indices[idx]]  # Fetch image and label from the original dataset at the specified index
        if self.transform:
            return self.transform(im), labels  # Apply the transform and return
        else:
            return im, labels  # Return without applying any transform

    def __len__(self):
        """
        Get the number of items in the subset.

        Returns
        -------
        int
            The number of items in the subset.
        """

        return len(self.indices)
