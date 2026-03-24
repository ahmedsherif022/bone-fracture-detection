import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model (must match your training model)
class bone_fr(nn.Module):
    def __init__(self, input_dim=3*150*150, hidden1=1024, hidden2=1024, hidden3=512, num_classes=1, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden3, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join('saved_models', 'bone_fraction.pth')
    if not os.path.exists(model_path):
        print('Model not found at', model_path)
        return None
    model = bone_fr().to(device)
    ckpt = torch.load(model_path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    # try strict load, otherwise fallback to non-strict
    try:
        model.load_state_dict(sd)
    except Exception as e:
        print('Warning loading state_dict strict=True ->', e)
        model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    print('Model loaded from', model_path)
    return model


def print_class_mapping(root):
    train_root = os.path.join(root, 'train')
    print('\nClass mapping from folders (train):', train_root)
    if not os.path.exists(train_root):
        print('  train folder not found')
        return []
    classes = sorted([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))])
    for i, c in enumerate(classes):
        print(f'  {i} -> {c}')
    return classes


def evaluate_on_test(model, dataset_root, batch_size=64, num_workers=0):
    img_size = 150
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    test_dir = os.path.join(dataset_root, 'test')
    if not os.path.exists(test_dir):
        print('Test folder not found at', test_dir)
        return

    test_ds = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    y_true = []
    y_pred = []
    probs = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            # logits shape (N,1)
            p = torch.sigmoid(logits).cpu().numpy().ravel()
            preds = (p > 0.5).astype(int)
            y_true.append(labels.numpy())
            y_pred.append(preds)
            probs.append(p)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    probs = np.concatenate(probs)

    print('\nConfusion matrix:')
    print(confusion_matrix(y_true, y_pred))
    print('\nClassification report:')
    print(classification_report(y_true, y_pred, digits=4))

    # show some misclassified examples (first 10)
    mis = np.where(y_true != y_pred)[0]
    print(f'\nTotal test samples: {len(y_true)}, misclassified: {len(mis)}')
    if len(mis) > 0:
        print('\nSome misclassified sample indices and probs:')
        for idx in mis[:10]:
            print(f'  idx={idx} true={y_true[idx]} pred={y_pred[idx]} prob={probs[idx]:.4f}')


if __name__ == '__main__':
    # update this path if your dataset is elsewhere
    dataset_root = r'D:\downloads\bone frature dataset\Bone_Fracture_Binary_Classification\Bone_Fracture_Binary_Classification'
    print_class_mapping(dataset_root)

    model = load_model()
    if model is None:
        raise SystemExit(1)

    evaluate_on_test(model, dataset_root)
