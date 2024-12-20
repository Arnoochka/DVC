import os
import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_acc(net: nn.Module, test_loader: DataLoader, device: str = 'cuda') -> float:
    """
    Evaluates accuracy of the model on the provided DataLoader.

    :param net: Trained neural network model.
    :param test_loader: DataLoader for the test dataset.
    :param device: Device to perform computations ('cuda' or 'cpu').
    :return: Accuracy of the model on the test dataset.
    """
    net.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc

def evaluate_roc(net: nn.Module, test_loader: DataLoader, device: str = 'cuda', num_classes: int = 3):
    """
    Evaluates the ROC curves and AUC for a multi-class classification problem.

    :param net: Trained neural network model.
    :param test_loader: DataLoader for the test dataset.
    :param device: Device to perform computations ('cuda' or 'cpu').
    :param num_classes: Number of classes in the dataset.
    :return: Dictionary containing fpr, tpr, and roc_auc for each class.
    """
    net.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            all_labels.append(labels.cpu())
            all_preds.append(outputs.cpu())

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve((all_labels == i).numpy(), all_preds[:, i].numpy())
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc, all_labels, torch.argmax(all_preds, dim=1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs("results/inference", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    model_path = 'models/model.pkl'

    net: nn.Module = None
    with open('models/model.pkl', 'rb') as f:
        net = pickle.load(f)

    net.to(device)

    train_loader_path = 'data/features/train.pkl'
    test_loader_path = 'data/features/test.pkl'

    with open(train_loader_path, 'rb') as f:
        train_dataloader = pickle.load(f)

    with open(test_loader_path, 'rb') as f:
        test_dataloader = pickle.load(f)

    train_metric = evaluate_acc(net, train_dataloader, device=device)
    test_metric = evaluate_acc(net, test_dataloader, device=device)

    with open('results/metrics/train_metric.json', 'w') as f:
        json.dump({"accuracy": train_metric}, f)

    with open('results/metrics/test_metric.json', 'w') as f:
        json.dump({"accuracy": test_metric}, f)
        
    num_classes = 3
    fpr, tpr, roc_auc, all_labels, all_preds = evaluate_roc(net, test_dataloader, device=device, num_classes=num_classes)

    # Plot ROC Curves
    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('results/inference/roc.png')
    plt.close()

    # Compute and plot confusion matrix
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_classes))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('results/inference/confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    main()
