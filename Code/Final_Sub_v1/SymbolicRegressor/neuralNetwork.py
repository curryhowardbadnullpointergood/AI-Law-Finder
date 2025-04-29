import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim  
import numpy as np 




class SymbolicNetwork(nn.Module):
    def __init__(self, n_input, n_output=1):
        super(SymbolicNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_output)
        )

    def forward(self, x):
        return self.model(x)




def prepare_data(data_x, data_y, batch_size, train_split=0.8):
    x_tensor = torch.tensor(data_x, dtype=torch.float32)
    y_tensor = torch.tensor(data_y, dtype=torch.float32).unsqueeze(1)  # Ensure y is (N, 1)

    num_samples = x_tensor.size(0)
    split_idx = int(num_samples * train_split)

    x_train, x_val = x_tensor[:split_idx], x_tensor[split_idx:]
    y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader





def train_network(model, train_loader, val_loader, epochs, learning_rate, device):
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                predictions = model(x_val)
                loss = loss_fn(predictions, y_val)
                val_loss += loss.item() * x_val.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")









def predict(model, x_numpy, device):
    model.eval()
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(x_tensor)

    return predictions.cpu().numpy()





def get_gradient(model, x_numpy, device):
    model.eval()

    x_tensor = torch.tensor(x_numpy, dtype=torch.float32, requires_grad=True).to(device)

    output = model(x_tensor)

    gradients = torch.autograd.grad(
        outputs=output,
        inputs=x_tensor,
        grad_outputs=torch.ones_like(output),
        create_graph=False,
        retain_graph=False
    )[0]

    return gradients.cpu().numpy()




























































print("hello")
