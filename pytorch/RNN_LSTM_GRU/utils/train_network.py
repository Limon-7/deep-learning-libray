import torch
from tqdm import tqdm
from utils import config

def train_model(model, train_loader):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.NUM_EPOCHS):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=config.DEVICE).squeeze(1)
            targets = targets.to(device=config.DEVICE)
            # Forward
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient descent or adam step
            optimizer.step()