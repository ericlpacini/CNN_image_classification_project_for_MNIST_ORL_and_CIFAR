import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import sys

from dataset import CustomImageDataset
from models import Method_CNN_MNIST, Method_CNN_ORL, Method_CNN_CIFAR, Method_CNN_CIFAR_Deep


def train_and_evaluate(load_dataset_name, display_name, model, epochs=10, batch_size=32, lr=0.001):
    print(f"\n--- Training CNN on {display_name.upper()} ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)

    train_dataset = CustomImageDataset(load_dataset_name, split='train')
    test_dataset = CustomImageDataset(load_dataset_name, split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, marker='o', color='b', label='Training Loss')
    plt.title(f'Training Convergence Plot - {display_name.upper()}')
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'convergence_{display_name.lower()}.png')
    plt.close()
    print(f"Saved convergence plot as convergence_{display_name.lower()}.png")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\nResults for {display_name.upper()}:")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")


if __name__ == '__main__':
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else 'orl'

    if arg == 'mnist':
        model = Method_CNN_MNIST()
        load_name = 'MNIST'
        display_name = 'mnist'
    elif arg == 'cifar':
        model = Method_CNN_CIFAR()
        load_name = 'CIFAR'
        display_name = 'cifar'
    elif arg == 'cifar_deep':
        model = Method_CNN_CIFAR_Deep()
        load_name = 'CIFAR'
        display_name = 'cifar_deep'
    else:
        model = Method_CNN_ORL()
        load_name = 'ORL'
        display_name = 'orl'

    train_and_evaluate(load_name, display_name, model, epochs=10, batch_size=32)