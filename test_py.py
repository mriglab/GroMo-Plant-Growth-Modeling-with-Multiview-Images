
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

################################# Input here ###################################
root_path='/home/shreya/Downloads/test/'
crop='radish'
csv_file='/home/shreya/Downloads/radish_test.csv'
n_images=4
plant_input=2
days_input=86
batch_size = 8
seed=42
height, width = 224, 224
# Transformations for resizing and converting to tensor
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
##############################################################################

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
                        print(filename)
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

test_dataset = CropDataset(root_dir=root_path,
                      csv_file=csv_file,
                      images_per_level=n_images,
                      crop=crop,
                      plants=plant_input,
                      days=days_input,
                      transform=transform)

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




# DataLoader for training and validation sets
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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

# Load the trained model
model_path = "/content/drive/MyDrive/ACM grand challenge/Crops data/For_age_prediction/results/okra_all files/okra_vit_age_prediction_10.pth"  # Change this to your actual model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming your model class is defined as MyModel
model = VisionTransformer(input_channels, patch_size, num_patches, projection_dim, num_heads, num_layers, mlp_dim, num_images, dropout_rate)  # Replace with your actual model class
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# Initialize lists to store predictions and actual values
y_true = []
y_pred = []

# Run inference on the test set
with torch.no_grad():
    for images, count, age in test_loader:  # Assuming test_loader gives (images, labels)
        images = images.to(device)
        count = count.to(device)
        age = age.to(device)

        outputs, attention = model(images)  # Ensure outputs are properly shaped
        y_true.extend(age.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())

# Convert to NumPy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Compute metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Print results
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Load the trained model
model_path = "/content/drive/MyDrive/ACM grand challenge/Crops data/For_age_prediction/results/okra_all files/okra_vit_leaf_count_8.pth"  # Change this to your actual model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming your model class is defined as MyModel
model = VisionTransformer(input_channels, patch_size, num_patches, projection_dim, num_heads, num_layers, mlp_dim, num_images, dropout_rate)  # Replace with your actual model class
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# Initialize lists to store predictions and actual values
y_true = []
y_pred = []

# Run inference on the test set
with torch.no_grad():
    for images, count, age in test_loader:  # Assuming test_loader gives (images, labels)
        images = images.to(device)
        count = count.to(device)
        count = count.to(device)

        outputs, attention = model(images)  # Ensure outputs are properly shaped
        y_true.extend(count.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())

# Convert to NumPy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Compute metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Print results
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")