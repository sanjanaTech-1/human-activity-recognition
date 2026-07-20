# ============================================================
# UCF101 3D RESNET VIDEO CLASSIFICATION
# 5 CLASSES | 500 VIDEOS PER CLASS | 10 EPOCHS
# ============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision.models.video import r3d_18

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\Hp\Downloads\UCF101\UCF-101"

NUM_CLASSES = 5
VIDEOS_PER_CLASS = 100

NUM_EPOCHS = 10
BATCH_SIZE = 2

NUM_FRAMES = 16
IMAGE_SIZE = 112

LEARNING_RATE = 0.0001
NUM_WORKERS = 0

MODEL_SAVE_PATH = "best_3d_resnet_ucf101.pth"


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. SELECT 5 CLASSES
# ============================================================

all_classes = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print("\nTotal Classes Found:", len(all_classes))

selected_classes = []

for class_name in all_classes:

    class_path = os.path.join(DATASET_PATH, class_name)

    videos = [
        file
        for file in os.listdir(class_path)
        if file.lower().endswith((".avi", ".mp4", ".mov", ".mkv"))
    ]

    if len(videos) >= VIDEOS_PER_CLASS:
        selected_classes.append(class_name)

    if len(selected_classes) == NUM_CLASSES:
        break


print("\nSelected Classes:")

for i, class_name in enumerate(selected_classes):
    print(i, ":", class_name)


if len(selected_classes) < NUM_CLASSES:
    raise ValueError(
        f"Not enough classes having {VIDEOS_PER_CLASS} videos each!"
    )


class_to_idx = {
    class_name: index
    for index, class_name in enumerate(selected_classes)
}

idx_to_class = {
    index: class_name
    for class_name, index in class_to_idx.items()
}


# ============================================================
# 5. CREATE VIDEO LIST
# ============================================================

video_data = []

for class_name in selected_classes:

    class_path = os.path.join(DATASET_PATH, class_name)

    videos = sorted([
        file
        for file in os.listdir(class_path)
        if file.lower().endswith((".avi", ".mp4", ".mov", ".mkv"))
    ])

    videos = videos[:VIDEOS_PER_CLASS]

    for video in videos:

        video_path = os.path.join(class_path, video)

        video_data.append(
            (video_path, class_to_idx[class_name])
        )


print("\nTotal Videos:", len(video_data))


# ============================================================
# 6. TRAIN VALIDATION TEST SPLIT
# ============================================================

paths = [item[0] for item in video_data]
labels = [item[1] for item in video_data]

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths,
    labels,
    test_size=0.20,
    stratify=labels,
    random_state=SEED
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=SEED
)


print("\nDataset Split:")
print("Training Videos:", len(train_paths))
print("Validation Videos:", len(val_paths))
print("Testing Videos:", len(test_paths))


# ============================================================
# 7. VIDEO DATASET
# ============================================================

class UCF101Dataset(Dataset):

    def __init__(
        self,
        video_paths,
        labels,
        num_frames=16,
        image_size=112,
        training=False
    ):

        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.image_size = image_size
        self.training = training

        self.mean = torch.tensor(
            [0.43216, 0.394666, 0.37645],
            dtype=torch.float32
        ).view(3, 1, 1, 1)

        self.std = torch.tensor(
            [0.22803, 0.22145, 0.216989],
            dtype=torch.float32
        ).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):

        video_path = self.video_paths[index]
        label = self.labels[index]

        try:

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError("Could not open video")

            total_video_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if total_video_frames <= 0:
                raise ValueError("Video has no frames")

            if self.training:
                start = random.randint(
                    0,
                    max(0, total_video_frames - self.num_frames)
                )

                frame_indices = np.linspace(
                    start,
                    min(
                        total_video_frames - 1,
                        start + self.num_frames - 1
                    ),
                    self.num_frames
                ).astype(int)

            else:

                frame_indices = np.linspace(
                    0,
                    total_video_frames - 1,
                    self.num_frames
                ).astype(int)

            frames = []

            for frame_index in frame_indices:

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(frame_index)
                )

                ret, frame = cap.read()

                if not ret:
                    continue

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                frame = cv2.resize(
                    frame,
                    (self.image_size, self.image_size)
                )

                frames.append(frame)

            cap.release()

            if len(frames) == 0:
                raise ValueError("Could not read video frames")

            while len(frames) < self.num_frames:
                frames.append(frames[-1].copy())

            frames = frames[:self.num_frames]

            video = np.stack(frames)

            video = torch.from_numpy(video).float() / 255.0

            video = video.permute(
                0, 3, 1, 2
            )

            if self.training and random.random() < 0.5:

                video = torch.flip(
                    video,
                    dims=[3]
                )

            video = video.permute(
                1, 0, 2, 3
            )

            video = (
                video - self.mean
            ) / self.std

            return video, torch.tensor(label).long()

        except Exception as e:

            print("\nError Loading Video:")
            print(video_path)
            print(e)

            dummy_video = torch.zeros(
                3,
                self.num_frames,
                self.image_size,
                self.image_size
            )

            return dummy_video, torch.tensor(label).long()


# ============================================================
# 8. DATASETS
# ============================================================

train_dataset = UCF101Dataset(
    train_paths,
    train_labels,
    NUM_FRAMES,
    IMAGE_SIZE,
    training=True
)

val_dataset = UCF101Dataset(
    val_paths,
    val_labels,
    NUM_FRAMES,
    IMAGE_SIZE,
    training=False
)

test_dataset = UCF101Dataset(
    test_paths,
    test_labels,
    NUM_FRAMES,
    IMAGE_SIZE,
    training=False
)


# ============================================================
# 9. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 10. LOAD 3D RESNET
# ============================================================

print("\nLoading 3D ResNet Model...")

model = r3d_18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

print("3D ResNet-18 Loaded Successfully")


# ============================================================
# 11. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    patience=2,
    factor=0.5
)


# ============================================================
# 12. TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        loader,
        desc="Training"
    )

    for videos, labels in progress_bar:

        videos = videos.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(videos)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        accuracy = 100 * correct / total

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{accuracy:.2f}%"
        )

    epoch_loss = running_loss / len(loader)

    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 13. VALIDATION FUNCTION
# ============================================================

def validate(model, loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        progress_bar = tqdm(
            loader,
            desc="Validation"
        )

        for videos, labels in progress_bar:

            videos = videos.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(videos)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            accuracy = 100 * correct / total

            progress_bar.set_postfix(
                accuracy=f"{accuracy:.2f}%"
            )

    epoch_loss = running_loss / len(loader)

    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 14. TRAIN MODEL
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_val_accuracy = 0.0

print("\nStarting Training...\n")


for epoch in range(NUM_EPOCHS):

    print("=" * 70)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print("=" * 70)

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader
    )

    val_loss, val_accuracy = validate(
        model,
        val_loader
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    scheduler.step(val_accuracy)

    print("\nEpoch Results:")

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "classes": selected_classes
            },
            MODEL_SAVE_PATH
        )

        print("\nBest Model Saved!")


# ============================================================
# 15. LOAD BEST MODEL
# ============================================================

print("\nLoading Best Model...")

checkpoint = torch.load(
    MODEL_SAVE_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# 16. TEST MODEL
# ============================================================

print("\nTesting Model...\n")

all_predictions = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():

    for videos, labels in tqdm(
        test_loader,
        desc="Testing"
    ):

        videos = videos.to(DEVICE)

        labels = labels.to(DEVICE)

        outputs = model(videos)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


test_accuracy = 100 * correct / total

print("\n========================================")

print(
    f"FINAL TEST ACCURACY: {test_accuracy:.2f}%"
)

print("========================================")


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=selected_classes,
        zero_division=0
    )
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=selected_classes,
    yticklabels=selected_classes
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("3D ResNet Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 19. TRAINING GRAPH
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.subplot(1, 2, 1)

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Graph")
plt.legend()


plt.subplot(1, 2, 2)

plt.plot(
    train_accuracies,
    label="Training Accuracy"
)

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Graph")
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 20. SINGLE VIDEO PREDICTION
# ============================================================

def predict_video(video_path):

    model.eval()

    dataset = UCF101Dataset(
        [video_path],
        [0],
        NUM_FRAMES,
        IMAGE_SIZE,
        training=False
    )

    video, _ = dataset[0]

    video = video.unsqueeze(0)

    video = video.to(DEVICE)

    with torch.no_grad():

        output = model(video)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            1
        )

    predicted_class = predicted_class.item()

    confidence = confidence.item() * 100

    print("\nPrediction Result")

    print(
        "Activity:",
        idx_to_class[predicted_class]
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    return idx_to_class[predicted_class], confidence


# ============================================================
# EXAMPLE PREDICTION
# ============================================================

# prediction_video = r"C:\path\to\your\video.avi"

# predict_video(prediction_video)


print("\nTraining Completed Successfully!")
