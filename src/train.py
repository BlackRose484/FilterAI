from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_model(model, train_data, test_data, num_epochs=35):
    best_test_loss = 1e9
    train_loss_history = []
    test_loss_history = []
    loss_fn = MSELoss()
    optimizer = Adam(params=model.model.classifier.parameters(), lr=1e-3)
    for epoch in range(num_epochs):
        # Train model
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_data)

        for images, landmarks in train_pbar:
            images = images.to(DEVICE)
            landmarks = landmarks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, landmarks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_pbar.set_postfix({"Train loss": loss.item()})

        train_loss = running_loss / len(train_data.dataset)
        train_loss_history.append(train_loss)

        # test model
        model.eval()
        test_loss = 0.0
        test_pbar = tqdm(test_data)

        with torch.no_grad():
            for images, landmarks in test_pbar:
                images = images.to(DEVICE)
                landmarks = landmarks.to(DEVICE)

                outputs = model(images)
                loss = loss_fn(outputs, landmarks)

                test_loss += loss.item() * images.size(0)
                test_pbar.set_postfix({"Test loss": loss.item()})

        test_loss = test_loss / len(test_data.dataset)
        test_loss_history.append(test_loss)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Model saving ...")

    return model
