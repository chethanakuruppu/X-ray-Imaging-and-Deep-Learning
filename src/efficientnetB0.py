# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time, os
from PIL import Image
from tempfile import TemporaryDirectory
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_recall_curve, average_precision_score,
    roc_curve, auc
)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = '/Users/chootydoony/Documents/Miun/EL035A_Project/Project-II/Thin_wedge/Dataset_4'

# Helper
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Training loop
def train_model(model, criterion, optimizer, scheduler, num_epochs=100, patience=10):
    since = time.time()
    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    best_acc, epochs_no_improve = 0.0, 0

    with TemporaryDirectory() as tempdir:
        best_model_path = os.path.join(tempdir, 'best_model.pt')
        torch.save(model.state_dict(), best_model_path)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}\n' + '-'*10)
            for phase in ['train', 'val']:
                model.train() if phase == 'train' else model.eval()
                running_loss, running_corrects = 0.0, 0

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc.item())
                else:
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_acc.item())
                    if scheduler: scheduler.step(epoch_loss)

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_path)
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            if epoch >= 15 and epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        model.load_state_dict(torch.load(best_model_path))

    # Plot
    def plot_metric(metric, name):
        plt.figure()
        plt.plot(moving_average(metric['train']), label='Train')
        plt.plot(moving_average(metric['val']), label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.title(f'{name} Curve ({norm_type})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f"{name.lower()}_{norm_type}.png"))

    plot_metric({'train': train_loss, 'val': val_loss}, 'Loss')
    plot_metric({'train': train_acc, 'val': val_acc}, 'Accuracy')

    return model

# Evaluation
def evaluate_on_test(model, norm_type):
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), current_transforms['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    model.eval()

    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    # Report
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(data_dir, f"test_report_{norm_type}.txt"), 'w') as f:
        f.write(report + f"\nTest Accuracy: {acc:.4f}\n")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(data_dir, f"cm_{norm_type}.png"))

    # ROC and PR curves
    y_true_onehot = np.eye(len(class_names))[y_true]
    y_scores = np.array(y_scores)

    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_scores[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_scores[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]} (AP={ap:.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'PR Curve ({norm_type})')
    plt.legend()
    plt.savefig(os.path.join(data_dir, f"pr_curve_{norm_type}.png"))

    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC Curve ({norm_type})')
    plt.legend()
    plt.savefig(os.path.join(data_dir, f"roc_curve_{norm_type}.png"))

# Run experiment
def run_experiment(use_imagenet_stats):
    global dataloaders, dataset_sizes, class_names, current_transforms, norm_type
    norm_type = 'imagenet' if use_imagenet_stats else 'custom'
    print(f"\n--- Running {norm_type.upper()} Normalization ---")

    mean = [0.485, 0.456, 0.406] if use_imagenet_stats else [0.9892, 0.9934, 0.9901]
    std = [0.229, 0.224, 0.225] if use_imagenet_stats else [0.0884, 0.0590, 0.0779]

    current_transforms = {
        'train': transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), current_transforms[x])
        for x in ['train', 'val']
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=(x == 'train'))
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # EfficientNet-B0
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, len(class_names))
    )
    model = model.to(device)

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    # Train and test
    model = train_model(model, criterion, optimizer, scheduler)
    evaluate_on_test(model, norm_type)

# Run both normalizations
run_experiment(use_imagenet_stats=True)
run_experiment(use_imagenet_stats=False)

# Print summaries
for norm_type in ['imagenet', 'custom']:
    print(f"\n>>> {norm_type.upper()} Normalization Summary:")
    with open(os.path.join(data_dir, f"test_report_{norm_type}.txt")) as f:
        for line in f:
            if "Accuracy" in line or "precision" in line or "f1-score" in line:
                print(line.strip())
