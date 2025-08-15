import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataprocess import MicrobiomeDataset
from model import MicrobiomeTransformerClassifier
from tqdm import tqdm
from sklearn.metrics import recall_score

import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        inputs = batch['input'].to(device)     # [B, 479, 100]
        masks = batch['mask'].to(device)       # [B, 479]
        labels = batch['label'].to(device)     # [B]

        # Forward pass
        logits = model(inputs, masks)
        loss = criterion(logits, labels)

        # Backward + Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            inputs = batch['input'].to(device)     # [B, 479, 100]
            masks = batch['mask'].to(device)       # [B, 479]
            labels = batch['label'].to(device)     # [B]

            logits = model(inputs, masks)
            loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    recall = recall_score(all_labels, all_preds, average='binary') * 100

    return avg_loss, accuracy, recall

def main():
    parser = argparse.ArgumentParser(description="Train Microbiome Model")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Updated Datasets
    train_dataset = MicrobiomeDataset("train_undersampled.csv", "glove_taxa_embeddings.csv", has_labels=True)
    val_dataset = MicrobiomeDataset("val_477_updated.csv", "glove_taxa_embeddings.csv", has_labels=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = MicrobiomeTransformerClassifier()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")

        val_loss, val_acc, val_recall = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}% | Val Recall: {val_recall:.2f}%")

        # Save every 2 epochs
        if (epoch + 1) % 1 == 0:
            save_path = f"The_undersampled_microbiome_model_128_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved to: {save_path}")

if __name__ == '__main__':
    main()
