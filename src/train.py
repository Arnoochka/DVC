from torch import nn, optim, cuda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import torch
from net import Net
from tqdm import tqdm
import os
import yaml
import csv

def train(net: nn.Module,
          train_loader: DataLoader,
          optimizer : optim.Optimizer,
          num_epochs: int,
          device: str = 'cuda') -> tuple:
    
    cuda.empty_cache()
    loss_function = nn.CrossEntropyLoss()
    acc_history = []
    loss_history = []

    with tqdm(total=len(train_loader)*num_epochs, position=0, leave=True) as pbar:

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0 
            
            for batch_num, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = loss_function(outputs, labels)
              
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                _, predicted = outputs.max(1)
                batch_total = labels.size(0)
                batch_correct = predicted.eq(labels).sum().item()
                batch_acc = batch_correct/batch_total
                
                pbar.set_description("Epoch: %d, Batch: %2d, Loss: %.2f, Acc: %.2f" % (epoch, batch_num, running_loss, batch_acc))
                pbar.update()

                total += batch_total
                correct += batch_correct

            acc = correct/total 
            acc_history.append(acc)
            avg_loss = running_loss / len(train_loader)
            loss_history.append(avg_loss)

        pbar.close()

    return acc_history, loss_history

def print_history(acc_history: list, loss_history: list):
    acc_data = [['epoch', 'accuracy']] + [[i+1, x] for i, x in enumerate(acc_history)]
    loss_data = [['epoch', 'loss']] + [[i+1, x] for i, x in enumerate(loss_history)]
    
    os.makedirs("results/train", exist_ok=True)
    
    with open("results/train/accuracy.tsv", "w") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerows(acc_data)
    
    with open("results/train/loss.tsv", "w") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerows(loss_data)

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("results", "train"), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    params = yaml.safe_load(open("params.yaml"))["train"]
    lr = params["lr"]
    epochs = params["epochs"]
    input_size = params["input_size"]
    num_classes = params["num_classes"]
    dimension = params["dimension"]
    
    train_dataloader = None
    with open('data/features/train.pkl', 'rb') as f:
        train_dataloader = pickle.load(f)
        
    
    net = Net(input_size, num_classes, dimension).to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    acc_history, loss_history = train(net, train_dataloader, optimizer, num_epochs=epochs, device=device)
    print_history(acc_history, loss_history)
    
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(net, f)
        
if __name__ == "__main__":
    main()