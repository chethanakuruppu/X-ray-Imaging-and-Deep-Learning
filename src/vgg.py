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

data_dir = '/Users/chootydoony/Documents/Miun/EL035A_Project/Project-II/Thin_wedge/dataset_2'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(np.clip(inp, 0, 1))
    if title:
        plt.title(title)
    plt.pause(0.001)

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def train_model(model, criterion, optimizer, scheduler, num_epochs=100, patience=10):
    since = time.time()
    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []
    best_acc = 0.0
    epochs_no_improve = 0
    output_dir = data_dir

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}\n' + '-' * 10)
            for phase in ['train', 'val']:
                model.train() if phase == 'train' else model.eval()
                running_loss = 0.0
                running_corrects = 0

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

            if epoch >= 30 and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        model.load_state_dict(torch.load(best_model_params_path))

    plt.figure()
    plt.plot(moving_average(train_loss_history), label='Train Loss')
    plt.plot(moving_average(val_loss_history), label='Val Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(os.path.join(output_dir, f"loss_plot_{norm_type}.png"))

    plt.figure()
    plt.plot(moving_average(train_acc_history), label='Train Acc')
    plt.plot(moving_average(val_acc_history), label='Val Acc')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig(os.path.join(output_dir, f"accuracy_plot_{norm_type}.png"))

    return model

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

    report = classification_report(y_true, y_pred, target_names=class_names)
    acc = accuracy_score(y_true, y_pred)
    with open(os.path.join(data_dir, f"test_report_{norm_type}.txt"), "w") as f:
        f.write("Classification Report:\n" + report + f"\nTest Accuracy: {acc:.4f}\n")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix ({norm_type})')
    plt.savefig(os.path.join(data_dir, f"confusion_matrix_{norm_type}.png"))

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Normalized Confusion Matrix ({norm_type})')
    plt.savefig(os.path.join(data_dir, f"normalized_confusion_matrix_{norm_type}.png"))

    y_true_onehot = np.eye(len(class_names))[y_true]
    y_scores = np.array(y_scores)

    plt.figure()
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_scores[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_scores[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]} (AP={ap:.2f})')
    plt.legend()
    plt.title(f'PR Curve ({norm_type})')
    plt.savefig(os.path.join(data_dir, f"pr_curve_{norm_type}.png"))

    plt.figure()
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title(f'ROC Curve ({norm_type})')
    plt.savefig(os.path.join(data_dir, f"roc_curve_{norm_type}.png"))

def run_experiment(use_imagenet_stats):
    global dataloaders, dataset_sizes, class_names, current_transforms, norm_type

    norm_type = 'imagenet' if use_imagenet_stats else 'custom'
    mean = [0.485, 0.456, 0.406] if use_imagenet_stats else [0.8816, 0.9289, 0.8946]
    std = [0.229, 0.224, 0.225] if use_imagenet_stats else [0.2705, 0.1777, 0.2310]

    current_transforms = {
        'train': transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
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
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=(x == 'train')) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # VGG16 model
    model = models.vgg16(weights='IMAGENET1K_V1')
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, len(class_names))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0002, weight_decay=0.0001)
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=2000)

    model = train_model(model, criterion, optimizer, scheduler)
    evaluate_on_test(model, norm_type)

# Run
run_experiment(use_imagenet_stats=True)
run_experiment(use_imagenet_stats=False)

# Summary Reports
for norm_type in ['imagenet', 'custom']:
    print(f"\n>>> {norm_type.upper()} Normalization Report:")
    with open(os.path.join(data_dir, f"test_report_{norm_type}.txt")) as f:
        for line in f:
            if "Accuracy" in line or "precision" in line or "f1-score" in line:
                print(line.strip())
