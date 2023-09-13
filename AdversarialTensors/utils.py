#@author: Manish Bhattarai, Mehmet Cagri
import torch


def eval_accuracy_dataloader(model, dataloader,device='cuda'):
    """
    Evaluate the model's accuracy using a given DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model for evaluation.
    dataloader : torch.utils.data.DataLoader
        DataLoader instance containing test data.
    device : str, optional
        Device to run the model on. Default is 'cuda'.

    Returns
    -------
    accuracy : float
        Accuracy of the model on the test data.
    """
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    accuracy = correct / total
    return accuracy

def eval_accuracy_dataloader_with_attack(model, dataloader, attacker, device='cuda'):
    """
    Evaluate the model's accuracy under adversarial attack using a given DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model for evaluation.
    dataloader : torch.utils.data.DataLoader
        DataLoader instance containing test data.
    attacker : object
        Instance responsible for generating adversarial attacks.
    device : str, optional
        Device to run the model on. Default is 'cuda'.

    Returns
    -------
    accuracy : float
        Accuracy of the model on the test data under adversarial attack.
    """
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = attacker.generate(inputs, targets)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    accuracy = correct / total
    return accuracy

def eval_accuracy_w_reconst_dataloader(model, dataloader,device='cuda'):
    """
    Evaluate the model's accuracy and average reconstruction error using a given DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model for evaluation.
    dataloader : torch.utils.data.DataLoader
        DataLoader instance containing test data.
    device : str, optional
        Device to run the model on. Default is 'cuda'.

    Returns
    -------
    accuracy : float
        Accuracy of the model on the test data.
    avg_rec_err : float
        Average reconstruction error on the test data.
    """
    total = 0
    correct = 0
    avg_rec_err = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, reconst_err = model(inputs, recon_err=True)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        avg_rec_err += reconst_err.sum().item()
    accuracy = correct / total
    avg_rec_err = avg_rec_err / total
    return accuracy, avg_rec_err

def eval_accuracy(model,X,y,correct=0,total=0):
    """
    Evaluate the model's accuracy on provided tensors.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model for evaluation.
    X : torch.Tensor
        Tensor containing the input data.
    y : torch.Tensor
        Tensor containing the ground truth labels.
    correct : int, optional
        Count of correctly classified samples. Default is 0.
    total : int, optional
        Count of total samples evaluated. Default is 0.

    Returns
    -------
    accuracy : float
        Accuracy of the model on the input data.
    correct : int
        Updated count of correctly classified samples.
    total : int
        Updated count of total samples evaluated.
    """
    outputs = model(X)
    _, predicted = outputs.max(1)
    total += y.size(0)
    correct += predicted.eq(y).sum().item()
    accuracy = correct / total
    return accuracy,correct,total