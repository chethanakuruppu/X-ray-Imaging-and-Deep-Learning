# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)

cudnn.benchmark = True
plt.ion()

# Set device
data_dir = '/Users/chootydoony/Documents/Miun/EL035A_Project/Project-II/Thin_wedge/Dataset_4'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Helper to show image (optional)
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(np.clip(inp, 0, 1))
    if title:
        plt.title(title)
    plt.pause(0.001)

# Helper to smooth graphs
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Train model with early stopping and LR scheduler
def train_model(model, criterion, optimizer, scheduler, num_epochs=100, patience=10):
    since = time.time()

    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    output_dir = data_dir
    best_acc = 0.0
    epochs_no_improve = 0

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                model.train() if phase == 'train' else model.eval()
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
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
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc.item())
                else:
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc.item())

                    if scheduler:
                        scheduler.step(epoch_loss)

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            if epoch >= 15 and epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs with no improvement (after 15 epochs).")
                break
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        model.load_state_dict(torch.load(best_model_params_path))

    # Plot loss and accuracy curves
    smoothed_train_acc = moving_average(train_acc_history)
    smoothed_val_acc = moving_average(val_acc_history)
    smoothed_train_loss = moving_average(train_loss_history)
    smoothed_val_loss = moving_average(val_loss_history)

    plt.figure()
    plt.plot(smoothed_train_loss, label='Train Loss (smoothed)')
    plt.plot(smoothed_val_loss, label='Val Loss (smoothed)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Smoothed)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"loss_plot_{norm_type}.png"))

    plt.figure()
    plt.plot(smoothed_train_acc, label='Train Acc (smoothed)')
    plt.plot(smoothed_val_acc, label='Val Acc (smoothed)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy (Smoothed)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"accuracy_plot_{norm_type}.png"))

    return model

# Evaluate test data and generate plots
def evaluate_on_test(model, norm_type):
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), current_transforms['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    acc = accuracy_score(y_true, y_pred)
    report_file = os.path.join(data_dir, f"test_report_{norm_type}.txt")
    with open(report_file, "w") as f:
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
    print(f"Report saved to {report_file}")

    # Confusion matrix (absolute)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix ({norm_type})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"confusion_matrix_{norm_type}.png"))

    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='PuRd', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Normalized Confusion Matrix ({norm_type})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"normalized_confusion_matrix_{norm_type}.png"))

    # PR Curves
    y_true_onehot = np.eye(len(class_names))[y_true]
    y_scores = np.array(y_scores)
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_scores[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_scores[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]} (AP={ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({norm_type})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"pr_curve_{norm_type}.png"))

    # ROC Curves
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({norm_type})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"roc_curve_{norm_type}.png"))

# Main experiment function
def run_experiment(use_imagenet_stats):
    global dataloaders, dataset_sizes, class_names, current_transforms, norm_type

    print(f"\n=== Running with {'ImageNet' if use_imagenet_stats else 'Custom'} Normalization ===")
    norm_type = 'imagenet' if use_imagenet_stats else 'custom'

    mean = [0.485, 0.456, 0.406] if use_imagenet_stats else [0.9892, 0.9934, 0.9901]
    std = [0.229, 0.224, 0.225] if use_imagenet_stats else [0.0884, 0.0590, 0.0779]

    current_transforms = {
        'train': transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), current_transforms[x]) for x in ['train', 'val']}
    #dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=0) for x in ['train', 'val']}
    dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True, num_workers=0),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=64, shuffle=False, num_workers=0)
}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
#Model_train_ResNet18
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(class_names))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.0001)
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=2000)

    model = train_model(model, criterion, optimizer, scheduler)
    evaluate_on_test(model, norm_type)

# Run both experiments
run_experiment(use_imagenet_stats=True)
run_experiment(use_imagenet_stats=False)

# Print report summaries
for norm_type in ['imagenet', 'custom']:
    print(f"\n>>> {norm_type.upper()} Normalization Report:")
    with open(os.path.join(data_dir, f"test_report_{norm_type}.txt")) as f:
        for line in f:
            if "Accuracy" in line or "precision" in line or "f1-score" in line:
                print(line.strip())
