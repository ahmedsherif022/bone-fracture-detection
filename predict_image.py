import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model Architecture
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


# Load model
def load_model(model_path=None, strict=True):
    model = bone_fr()
    if model_path is None:
        model_path = os.path.join('saved_models', 'bone_fraction.pth')

    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location='cpu')

    # try to extract state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        sd = checkpoint
    else:
        sd = checkpoint

    # Remove 'module.' prefix if present (saved from DataParallel)
    def _strip_module_prefix(state_dict):
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith('module.'):
                new_k = k[len('module.'):]
            new_sd[new_k] = v
        return new_sd

    if any(k.startswith('module.') for k in list(sd.keys())):
        sd = _strip_module_prefix(sd)

    print('\n🔎 Checkpoint keys and shapes:')
    for k, v in sd.items():
        try:
            print(f"  {k}: {tuple(v.shape)}")
        except Exception:
            print(f"  {k}: (unknown shape)")

    # compare expected shapes
    expected = model.state_dict()
    mismatch = False
    print('\n🔧 Comparing checkpoint -> model parameter shapes:')
    for k_exp, v_exp in expected.items():
        if k_exp in sd:
            try:
                if tuple(v_exp.shape) != tuple(sd[k_exp].shape):
                    print(f"  MISMATCH {k_exp}: model expects {tuple(v_exp.shape)}, checkpoint has {tuple(sd[k_exp].shape)}")
                    mismatch = True
            except Exception:
                print(f"  UNABLE TO COMPARE {k_exp}")
        else:
            print(f"  MISSING in checkpoint: {k_exp}")
            mismatch = True

    if mismatch:
        print('\n⚠️  Detected parameter shape mismatches or missing keys.')
        print('    Attempting to load with strict=False (will ignore unmatched keys).')
        try:
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print('❌ Failed to load state_dict even with strict=False:')
            print(e)
            return None
    else:
        model.load_state_dict(sd)

    model.to(device)
    model.eval()
    print(f"✅ Model loaded from {model_path}")
    return model


# Preprocess image
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return None

    try:
        img = Image.open(image_path).convert('RGB')
        print(f"✅ Image loaded: {image_path}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

    # Resize to 150x150
    img = img.resize((150, 150))

    # Convert to tensor
    img_array = np.array(img) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

    # Normalize: (x - 0.5) / 0.5 -> scale to [-1,1]
    img_tensor = img_tensor * 2.0 - 1.0

    return img_tensor.unsqueeze(0).to(device)


# Predict
def predict_single(model, image_path, verbose=True):
    img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return None

    with torch.no_grad():
        logits = model(img_tensor)
        probability = torch.sigmoid(logits)[0][0].item()
        prediction = 1 if probability > 0.5 else 0

    return {
        'prediction': prediction,
        'probability': probability,
        'raw': float(logits[0][0].item())
    }


def print_class_mapping(dataset_root=None):
    """Print sorted class folder names and mapping used by ImageFolder."""
    if dataset_root is None:
        # default from the notebook
        dataset_root = r"D:\downloads\bone frature dataset\Bone_Fracture_Binary_Classification\Bone_Fracture_Binary_Classification"
    train_root = os.path.join(dataset_root, 'train')
    print('\n📁 Checking class folders in:', train_root)
    if not os.path.exists(train_root):
        print('  (train folder not found)')
        return []
    classes = sorted([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))])
    for idx, name in enumerate(classes):
        print(f"  {idx} -> {name}")
    return classes


def test_on_sample_images(model, dataset_root=None):
    """Run predictions on one sample image per class (if dataset available)."""
    classes = print_class_mapping(dataset_root)
    if not classes:
        print('\nNo classes found to test sample predictions.')
        return
    samples = []
    for cls in classes:
        class_dir = os.path.join(dataset_root, 'train', cls)
        imgs = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        if imgs:
            samples.append(os.path.join(class_dir, imgs[0]))
    print('\n🔬 Running sample predictions:')
    for p in samples:
        try:
            print('\n--', p)
            res = predict_single(model, p, verbose=False)
            if res is None:
                print('  could not preprocess')
            else:
                print(f"  pred={res['prediction']} prob={res['probability']:.4f} raw={res['raw']:.4f}")
        except Exception as e:
            print('  error:', e)


# Main - set your image path and optionally dataset root here
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict a single image with the saved model')
    parser.add_argument('--image', '-i', help='Path to image to predict')
    parser.add_argument('--model', '-m', help='Path to model .pth', default=os.path.join('saved_models', 'bone_fraction.pth'))
    parser.add_argument('--dataset', '-d', help='Dataset root to run sample checks (optional)')
    parser.add_argument('--flip-labels', action='store_true', help='Flip class labels (if mapping inverted)')
    args = parser.parse_args()

    if not args.image:
        print('Please provide an image with --image or -i')
        print('Example: python predict_image.py -i "C:\\path\\to\\xray.jpg"')
        sys.exit(1)

    image_path = args.image
    dataset_root = args.dataset

    print('\n=== Loading model and running diagnostics ===')
    model = load_model(model_path=args.model)
    if model is None:
        sys.exit(1)

    if dataset_root:
        test_on_sample_images(model, dataset_root=dataset_root)

    print('\n=== Predicting chosen image ===')
    res = predict_single(model, image_path)
    if res is not None:
        # default mapping
        classes = ['No Fracture', 'Fracture']
        if args.flip_labels:
            classes = ['Fracture', 'No Fracture']
        predicted_class = classes[res['prediction']]
        prob = res['probability']
        raw = res['raw']
        print('\n' + '='*60)
        print('📊 PREDICTION RESULTS')
        print('='*60)
        print(f"Prediction:         {predicted_class}")
        print(f"Fracture Prob:      {prob*100:.2f}%")
        print(f"No Fracture Prob:   {(1-prob)*100:.2f}%")
        print(f"Raw Output Value:   {raw:.4f}")
        print('='*60 + '\n')
        if res['prediction'] == 1:
            print('⚠️  WARNING: Potential fracture detected!')
        else:
            print('✅ Good news: No fracture detected!')
