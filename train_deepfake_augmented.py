import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import librosa

# =====================================================
# 1. CONFIGURATION
# =====================================================
SAMPLE_RATE = 16000
CUT_LENGTH = 64600
BATCH_SIZE = 32  
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REAL_DIR = "/home/eeuser/augmented_testset/real"
FAKE_DIR = "/home/eeuser/augmented_testset/fake"

# =====================================================
# 2. FOCAL LOSS IMPLEMENTATION
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Get the probability of the true class
        pt = torch.exp(-ce_loss)
        # Apply focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# =====================================================
# 3. DATASET & DATALOADER
# =====================================================
class DeepfakeDataset(Dataset):
    def __init__(self, real_files, fake_files):
        self.files = real_files + fake_files
        self.labels = [1] * len(real_files) + [0] * len(fake_files) 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio, _ = librosa.load(self.files[idx], sr=SAMPLE_RATE, mono=True)
        if len(audio) > CUT_LENGTH:
            audio = audio[:CUT_LENGTH]
        else:
            audio = np.pad(audio, (0, CUT_LENGTH - len(audio)), 'constant')
        return torch.tensor(audio, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

print("Scanning augmented dataset...")
real_files = glob.glob(f"{REAL_DIR}/*.wav")
fake_files = glob.glob(f"{FAKE_DIR}/*.wav")
print(f"Found {len(real_files)} real files and {len(fake_files)} fake files.")

train_real, val_real = train_test_split(real_files, test_size=0.15, random_state=42)
train_fake, val_fake = train_test_split(fake_files, test_size=0.15, random_state=42)

train_loader = DataLoader(DeepfakeDataset(train_real, train_fake), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DeepfakeDataset(val_real, val_fake), batch_size=BATCH_SIZE, shuffle=False)

# =====================================================
# 4. MODEL ARCHITECTURE
# =====================================================
class LightweightDeepfakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=80, stride=4, padding=38)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

model = LightweightDeepfakeNet().to(DEVICE)

# ✨ Use Focal Loss instead of CrossEntropyLoss
criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

# =====================================================
# 5. TRAINING LOOP
# =====================================================
print(f"🚀 Starting Focal Loss training on {DEVICE.upper()}...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for audio, labels in train_loader:
        audio, labels = audio.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(audio)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for audio, labels in val_loader:
            audio, labels = audio.to(DEVICE), labels.to(DEVICE)
            outputs = model(audio)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        acc = 100 * correct / total
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Focal Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

# =====================================================
# 6. EXPORT TO ONNX
# =====================================================
print("\n📦 Exporting to ONNX...")
dummy_input = torch.randn(1, CUT_LENGTH).to(DEVICE)
torch.onnx.export(
    model, dummy_input, "deepfake_detector.onnx",
    input_names=["audio"], output_names=["logits"],
    dynamic_axes={"audio": {0: "batch_size"}},
    opset_version=14
)
print("✅ Success! Model saved as deepfake_detector.onnx")
