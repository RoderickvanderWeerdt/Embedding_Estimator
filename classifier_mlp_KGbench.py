from torch import nn
import pandas as pd

from dataset_KGbench import Emb_KGbench_Dataset, Features_KGbench_Dataset, ToTensor
from torch.utils.data import DataLoader
import torch

def shuffle_dataset(dataset_fn):
    df = pd.read_csv(dataset_fn, sep=",")
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv("shuffled_dataset.csv", index=False)


def perform_prediction(dataset_fn, show_all, list_of_value_headers, save_model=False, test_set_fn=""): #embeddings=<list_of_value_headers=[] else features
    if test_set_fn == "":
        shuffle_dataset(dataset_fn)
        if list_of_value_headers == []:
            training_data = Emb_KGbench_Dataset(csv_file="shuffled_dataset.csv", train=True, transform=ToTensor())
            test_data = Emb_KGbench_Dataset(csv_file="shuffled_dataset.csv", train=False, transform=ToTensor())
        else:
            training_data = Features_KGbench_Dataset(csv_file="shuffled_dataset.csv", list_of_value_headers=list_of_value_headers, train=True, transform=ToTensor())
            test_data = Features_KGbench_Dataset(csv_file="shuffled_dataset.csv", list_of_value_headers=list_of_value_headers, train=False, transform=ToTensor())
    else:
        # print("implent this before you use it!")
        # return 0
        train_set_fn = dataset_fn
    #     if list_of_value_headers == []:
        shuffle_dataset(dataset_fn)
        training_data = Emb_KGbench_Dataset(csv_file="shuffled_dataset.csv", train=True, transform=ToTensor(), train_all=True)
        shuffle_dataset(test_set_fn)
        test_data = Emb_KGbench_Dataset(csv_file="shuffled_dataset.csv", train=False, transform=ToTensor(), train_all=True)
    #     else:
    #         shuffle_dataset(dataset_fn)
    #         training_data = Features_KGBench_Dataset(csv_file="shuffled_dataset.csv", list_of_value_headers=list_of_value_headers, train=True, transform=ToTensor(), train_all=True)
    #         shuffle_dataset(test_set_fn)
    #         test_data = Features_KGBench_Dataset(csv_file="shuffled_dataset.csv", list_of_value_headers=list_of_value_headers, train=False, transform=ToTensor(), train_all=True)

    batch_size = 8

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    train_tss = []
    test_tss = []

    for sample in test_dataloader:
        X = sample['embedding']
        embedding_size = X.shape[1]
        y = sample['target_class']
        if show_all: print("Shape of X [N, C, H, W]: ", X.shape)
        if show_all: print("Shape of y: ", y.shape, y.dtype)
        break

    # exit()

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if show_all: print("Using {} device".format(device))

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, output_size):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, output_size)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork(embedding_size, training_data.get_n_classes()).to(device)
    model = model.float()
    if show_all: print(model)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        total = 0
        correct = 0
        for batch, sample in enumerate(dataloader):
            X = sample['embedding']
            y = sample['target_class']
            X, y = X.to(device), y.to(device)
            
            train_tss.append(y)

            # Compute prediction error
            pred = model(X.float())
            # print("pred", pred)
            # print("y", y)
            loss = loss_fn(pred, y.long())
            _, predicted = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            # print(batch)
            # print(predicted, y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                if show_all: print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                # print("x", X[0], "y", y[0], "pred", pred[0])
                # print("y", y)
                # print("x", X[0], "y", torch.round(y[0]), "pred", torch.round(pred[0]))
        
        correct /= total
        if show_all: print(f"Train Accuracy: {(100*correct):>0.1f}%")
        return correct


    def test(dataloader, model, loss_fn):
        num_batches = len(dataloader)
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for sample in dataloader:
                X = sample['embedding']
                y = sample['target_class']
                X, y = X.to(device), y.to(device)
                test_tss.append(y)
                pred = model(X.float())
                _, predicted = torch.max(pred.data, 1)
                # print("pred", predicted)
                # print("y",y)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        correct /= total
        if show_all: print(f"Test Accuracy: {(100*correct):>0.1f}%")
        return correct

    epochs = 40
    model = model.float()
    test_res = []
    train_res = []
    for t in range(epochs):
        if show_all: print(f"Epoch {t+1}\n-------------------------------")
        train_res.append(train(train_dataloader, model, loss_fn, optimizer))
        test_res.append(test(test_dataloader, model, loss_fn))
    if show_all: print("Done!")
    if save_model: torch.save(model.state_dict(), dataset_fn[:-4]+".pytorch-model")
    return {"train_result":train_res[-1],"test_result":test_res[-1], "targets_x": train_tss, "targets_y": test_tss}