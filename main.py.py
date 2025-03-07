

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# Define the device (GPU or CPU)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
################################# Input here ###################################
root_path='/home/shreya/Downloads/'
crop='radish'
csv_file='/home/shreya/Downloads/radish_train.csv'
crop='okra'
n_images=4
epochs=10
plant_input=4
days_input=59
batch_size = 8
seed=42
height, width = 224, 224
# Transformations for resizing and converting to tensor
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
##############################################################################

# Set random seeds for reproducibility
def set_seed(seed=42):
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # Numpy random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed (for CUDA)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in CUDA (if using GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Initialize seed for reproducibility
set_seed(42)

num_images = n_images  # Number of images (24 RGB images)
input_channels = num_images*3  # 24 RGB images (3 channels each)
patch_size = 16  # Size of each patch
num_patches = (height // patch_size) * (width // patch_size)  # Number of patches (14 * 14 for 224x224 images)
projection_dim = 256  # Embedding dimension for each patch
num_heads = 8  # Number of attention heads
num_layers = 6  # Number of transformer layers
mlp_dim = 512  # Dimension of the MLP head
num_classes = 1  # Number of output classes (for day or leaf count prediction)
dropout_rate = 0.1  # Dropout rate


class CropDataset(Dataset):
    def __init__(self, root_dir, csv_file, images_per_level, crop, plants, days,
                 levels=['L1', 'L2', 'L3', 'L4', 'L5'], transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            csv_file (str): Path to the CSV file containing ground truth (filename, leaf_count, age).
            images_per_level (int): Number of images to select per level (should be factors of 24).
            crop (str): Crop type (e.g., "radish").
            plants (int): Number of plants (e.g., 4).
            days (int): Number of days (e.g., 59).
            levels (list): List of levels (e.g., ['L1', 'L2', 'L3', 'L4', 'L5']).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.images_per_level = images_per_level
        self.crop = crop
        self.plants_num = plants
        self.max_days = days
        self.levels = levels
        self.transform = transform
        self.image_data = self._load_metadata()
        self.image_paths = self._load_image_paths()

    def _load_metadata(self):
        """Load CSV file into a pandas DataFrame and map filenames to leaf counts and ages."""
        df = pd.read_csv(self.csv_file)
        df["filename"] = df["filename"].astype(str)  # Ensure filenames are strings
        return df.set_index("filename")  # Use filename as the index for quick lookup

    def _select_angles(self):
        """
        Select angles dynamically for a given level.
        """
        images_needed = self.images_per_level
        selected_angles = [i for i in range(0, 360, int(360 / images_needed))]

        initial_angles = [i for i in range(15, selected_angles[1], 15)]
        multiple_selections = [selected_angles]

        for initial_angle in initial_angles:
            selection = [initial_angle]
            while len(selection) < images_needed:
                next_angle = (selection[-1] + int(360 / images_needed)) % 360
                if next_angle not in selection:
                    selection.append(next_angle)
            multiple_selections.append(selection)
        print(multiple_selections)
        return multiple_selections

    def _load_image_paths(self):
        """
        Load image paths for all levels and plants based on the selection of angles.
        """
        image_paths = []
        multiple_selections = self._select_angles()

        for plant in range(1, self.plants_num + 1):
            plant_path = os.path.join(self.root_dir, crop, f"p{plant}")
            if not os.path.isdir(plant_path):
                print(f"Plant directory not found: {plant_path}")
                continue
            for day in range(1, self.max_days + 1):
                day_path = os.path.join(self.root_dir, crop, f"p{plant}", f"d{day}")
                if not os.path.isdir(day_path):
                    continue
                for selected_angles in multiple_selections:
                    for level in self.levels:
                        level_path = os.path.join(self.root_dir,self.crop, f"p{plant}", f"d{day}", level)
                        level_image_paths = [
                            os.path.join(level_path, f"{self.crop}_p{plant}_d{day}_{level}_{angle}.png")
                            for angle in selected_angles
                        ]
                        filename = os.path.join(self.crop,f"p{plant}", f"d{day}", level,f"{self.crop}_p{plant}_d{day}_{level}_{selected_angles[0]}.png")
                        leaf_count = self.image_data.loc[filename, "leaf_count"]
                        # print(level_image_paths)
                        image_paths.append((level_image_paths, leaf_count,day))  # Append day number along with image paths

        print(f"Total samples loaded: {len(image_paths)}")
        # print(f"individual sample size: {len(image_paths[0][0])}")
        return image_paths


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        """
        Get a batch of images from the dataset corresponding to the angles selected.
        """
        images = []
        leaf_count = self.image_paths[idx][1]
        age = self.image_paths[idx][2]
        # print(leaf_count,age)
        all_images= self.image_paths[idx][0]
        # print("length of all images:", len(all_images))
        for img_path in all_images:  # Get the image paths for this sample
            if os.path.isfile(img_path):
                  level_image = Image.open(img_path)
                  if self.transform:
                      level_image = self.transform(level_image)
                  images.append(level_image)
            else:
                    print(f"Path is not a valid file: {img_path}")

        images = torch.cat(images, dim=0)

        return images, torch.tensor(leaf_count, dtype=torch.float32), torch.tensor(age, dtype=torch.float32)  # Return both images and the corresponding day as ground truth

dataset = CropDataset(root_dir=root_path,
                      csv_file=csv_file,
                      images_per_level=n_images,
                      crop=crop,
                      plants=plant_input,
                      days=days_input,
                      transform=transform)
# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
print(train_size,val_size)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



class VisionTransformer(nn.Module):
    def __init__(self, input_channels, patch_size, num_patches, projection_dim, num_heads, num_layers, mlp_dim, num_images, dropout_rate=0.1):
        super(VisionTransformer, self).__init__()

        self.num_images = num_images  # Total number of images (24 images in your case)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim

        # Separate patch embedding layers for each image (RGB)
        self.patch_embeds = nn.ModuleList([
            nn.Conv2d(input_channels // num_images, projection_dim, kernel_size=patch_size, stride=patch_size)
            for _ in range(num_images)
        ])

        # Positional Encoding (Learnable)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, projection_dim))

        # Transformer Encoder Layers (modified to return attention weights)
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=projection_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout_rate,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # MLP Head for classification/regression
        self.mlp_head = nn.Sequential(
            nn.Linear(projection_dim * num_images, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Step 1: Patch Embedding (Separate for each image)
        patch_embeddings = []
        for i in range(self.num_images):
            # Split the input channels into separate images (3 channels each for RGB)
            img_x = x[:, i*3:(i+1)*3, :, :]  # Shape: (batch_size, 3, height, width)
            patch_embed = self.patch_embeds[i](img_x)  # Apply separate embedding
            patch_embed = patch_embed.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, projection_dim)
            patch_embeddings.append(patch_embed)

        # Step 2: Add Positional Encoding
        patch_embeddings = [pe + self.positional_encoding for pe in patch_embeddings]

        # Step 3: Transformer Encoder Layers (Self-attention + Feed Forward)
        attention_weights = []  # To store attention weights
        for layer in self.attention_layers:
            layer_attention_weights = []  # Store the attention weights for each layer
            for i in range(self.num_images):
                # Modified to return attention weights (self-attention)
                attn_output, attn_weights = layer.self_attn(patch_embeddings[i], patch_embeddings[i], patch_embeddings[i])
                patch_embeddings[i] = attn_output
                layer_attention_weights.append(attn_weights)
            attention_weights.append(layer_attention_weights)

        # Step 4: Concatenate the projections from each image (Shape: (batch_size, num_patches, projection_dim * num_images))
        x = torch.cat(patch_embeddings, dim=-1)  # Concatenate across the last dimension (projection_dim)

        # Step 5: Pooling (Take mean across all patches)
        x = x.mean(dim=1)  # Mean pooling over patches (Shape: (batch_size, projection_dim * num_images))

        # Step 6: MLP Head for classification/regression
        output = self.mlp_head(x)

        return output, attention_weights  # Return attention weights too

# Create two independent instances of the model

def create_model():
    return VisionTransformer(input_channels, patch_size, num_patches, projection_dim, num_heads, num_layers, mlp_dim, num_images, dropout_rate)

# Create two independent instances of the model
model = [create_model(), create_model()]

model[0].to(device)
model[1].to(device)
optimizer = [optim.Adam(model[0].parameters(), lr=0.0001), optim.Adam(model[1].parameters(), lr=0.0001)]
criterion = nn.MSELoss()

def train_and_validate(train_loader, val_loader, num_epochs=10):
    train_losses_leaf, train_losses_age, val_losses_leaf, val_losses_age = [], [], [], []
    train_mae_leaf, train_mae_age, val_mae_leaf, val_mae_age = [], [], [], []
    train_r2_leaf, train_r2_age, val_r2_leaf, val_r2_age = [], [], [], []

    for epoch in range(num_epochs):
        # Training Phase
        for i in range(2):
            model[i].train()

        total_loss = [0, 0]
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        all_preds, all_labels = [[], []], [[], []]
        
        for batch_idx, (images, leaf_labels, age_labels) in enumerate(train_loader_tqdm):
            images, leaf_labels, age_labels = images.to(device), leaf_labels.to(device), age_labels.to(device)
            
            for i in range(2):
                optimizer[i].zero_grad()
                preds, _ = model[i](images)
                loss = criterion(preds.squeeze(), leaf_labels if i == 0 else age_labels)
                loss.backward()
                optimizer[i].step()
                total_loss[i] += loss.item()
                
                all_preds[i].extend(preds.squeeze().cpu().detach().numpy())
                all_labels[i].extend((leaf_labels if i == 0 else age_labels).cpu().numpy())

            train_loader_tqdm.set_postfix({"Leaf RMSE": (total_loss[0]/(batch_idx+1))**0.5, "Age RMSE": (total_loss[1]/(batch_idx+1))**0.5})

        train_losses_leaf.append(total_loss[0]**0.5)
        train_losses_age.append(total_loss[1]**0.5)
        train_mae_leaf.append(mean_absolute_error(all_labels[0], all_preds[0]))
        train_mae_age.append(mean_absolute_error(all_labels[1], all_preds[1]))
        train_r2_leaf.append(r2_score(all_labels[0], all_preds[0]))
        train_r2_age.append(r2_score(all_labels[1], all_preds[1]))

        # Validation Phase
        for i in range(2):
            model[i].eval()

        total_val_loss = [0, 0]
        all_val_preds, all_val_labels = [[], []], [[], []]

        with torch.no_grad():
            for images, leaf_labels, age_labels in val_loader:
                images, leaf_labels, age_labels = images.to(device), leaf_labels.to(device), age_labels.to(device)
                
                for i in range(2):
                    preds, _ = model[i](images)
                    loss = criterion(preds.squeeze(), leaf_labels if i == 0 else age_labels)
                    total_val_loss[i] += loss.item()
                    
                    all_val_preds[i].extend(preds.squeeze().cpu().numpy())
                    all_val_labels[i].extend((leaf_labels if i == 0 else age_labels).cpu().numpy())
                val_loader_tqdm.set_postfix({"Val Leaf RMSE": (total_val_loss[0]/(batch_idx+1))**0.5, "Val Age RMSE": (total_val_loss[1]/(batch_idx+1))**0.5})

        val_losses_leaf.append(total_val_loss[0]**0.5)
        val_losses_age.append(total_val_loss[1]**0.5)
        val_mae_leaf.append(mean_absolute_error(all_val_labels[0], all_val_preds[0]))
        val_mae_age.append(mean_absolute_error(all_val_labels[1], all_val_preds[1]))
        val_r2_leaf.append(r2_score(all_val_labels[0], all_val_preds[0]))
        val_r2_age.append(r2_score(all_val_labels[1], all_val_preds[1]))

        print(f"Epoch {epoch+1}/{num_epochs} - Train MAE Leaf: {train_mae_leaf[-1]:.4f}, Train MAE Age: {train_mae_age[-1]:.4f}, R² Leaf: {train_r2_leaf[-1]:.4f}, R² Age: {train_r2_age[-1]:.4f}")
        print(f"Validation - MAE Leaf: {val_mae_leaf[-1]:.4f}, MAE Age: {val_mae_age[-1]:.4f}, R² Leaf: {val_r2_leaf[-1]:.4f}, R² Age: {val_r2_age[-1]:.4f}")
        torch.save(model[0].state_dict(), f"okra_vit_leaf_count_{epoch+1}.pth")
        torch.save(model[1].state_dict(), f"okra_vit_age_prediction_{epoch+1}.pth")

    torch.save(model[0].state_dict(), "okra_vit_leaf_count.pth")
    torch.save(model[1].state_dict(), "okra_vit_age_prediction.pth")
    print("Models saved successfully!")

    print("Train Losses Leaf:", train_losses_leaf)
    print("Validation Losses Leaf:", val_losses_leaf)
    print("Train Losses Age:", train_losses_age)
    print("Validation Losses Age:", val_losses_age)
    print(num_epochs)


    plt.figure(figsize=(10,5))
    plt.plot(range(1, num_epochs+1), train_losses_leaf, label='Train Leaf RMSE')
    plt.plot(range(1, num_epochs+1), val_losses_leaf, label='Validation Leaf RMSE')
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Leaf Count Training and Validation RMSE")
    plt.savefig("leaf_training_validation_rmse.png")
    plt.savefig("leaf_training_validation_rmse.pdf")
    print('1')
    plt.figure(figsize=(10,5))
    plt.plot(range(1, num_epochs+1), train_losses_age, label='Train Age RMSE')
    plt.plot(range(1, num_epochs+1), val_losses_age, label='Validation Age RMSE')
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Age Prediction Training and Validation RMSE")
    plt.savefig("age_training_validation_rmse.png")
    plt.savefig("age_training_validation_rmse.pdf")
    print("Graphs saved as PNG and PDF!")

train_and_validate(train_loader, val_loader, num_epochs=epochs)
