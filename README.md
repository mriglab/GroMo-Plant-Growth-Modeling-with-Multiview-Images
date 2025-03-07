# GroMo-Plant-Growth-Modeling-with-Multiview-Images
Plant Growth Modelling using age estimation (in days) and leaf counting

# Multi View Vision Transformer (MVVT) for Plant Age and Leaf Count Estimation

This repository implements a Vision Transformer (ViT) model for estimating the age (in days) and the number of leaves of a plant using multi-view image data. The model is trained using a dataset containing images of plants at different growth stages.

## Features
- Uses a **Vision Transformer (ViT)** for feature extraction and prediction.
- Supports training with configurable hyperparameters.
- Handles multiple views of a plant by concatenating latent representations.
- Computes RMSE loss for both **plant age estimation** and **leaf count prediction**.

## Important Parameters
- **Number of Plants (`n_plants`)**: Defines how many different plants are included in the dataset.
- **Maximum Days of Crop (`max_days`)**: The maximum number of days considered for plant growth.
- **Number of Multi-View Images (`n_images`)**: The number of images selected from the total 24 available multi-view images per plant.

## Dataset Structure
The dataset consists of images of multiple plants (`p1`, `p2`, ..., `pn`) captured over different days (`d1`, `d2`, ..., `dm`) and categorized into five levels (`L1`, `L2`, `L3`, `L4`, `L5`). Each plant has **24 images** per growth cycle, representing different angles with a **15-degree gap** between consecutive images.

### Naming Convention
Each image follows the format:
```
radish_pX_dY_LZ_A.jpg
```
where:
- `X` represents the plant ID (`p1, p2, ...`)
- `Y` represents the day (`d1, d2, ...`)
- `Z` represents the level (`L1, L2, L3, L4, L5`)
- `A` represents the angle (ranging from `0` to `345` degrees in 15-degree increments)

### Directory Structure
```
/dataset/
    ├── train/
    │   ├── p1/
    │   │   ├── d1/
    │   │   │   ├── radish_p1_d1_L1_0.jpg
    │   │   │   ├── radish_p1_d1_L1_15.jpg
    │   │   │   ├── ...
    │   │   │   ├── radish_p1_d1_L5_345.jpg
    │   │   ├── d2/
    │   │   ├── ...
    │   ├── p2/
    │   ├── ...
```
Each plant has images captured at different time points (`d1`, `d2`, ...), categorized into five levels (`L1` to `L5`), with a total of 24 images per level taken from different angles.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install torch torchvision numpy tqdm
```

## Usage

### 1. Dataset Preparation
Modify the dataset path in `CropDataset` to point to your dataset location.

### 2. Training
Run the following command to train the ViT model:
```bash
python main.py
```
Hyperparameters such as the number of epochs, batch size, and learning rate can be modified directly in `main.py`.

### 3. Model Architecture
The Vision Transformer (ViT) extracts features from the images, flattens them, and passes them through a transformer encoder. The final representation is used to predict both leaf count and plant age.

### 4. Output Metrics
- **Leaf Count RMSE**: Measures the error in predicting the number of leaves.
- **Age RMSE**: Measures the error in predicting the plant’s age.

## Results
During training, the loss function minimizes the combined RMSE loss for both leaf count and plant age. The model prints training losses per epoch.

## Future Improvements
- Implement data augmentation for robustness.
- Explore attention-based fusion of multi-view features.
- Compare ViT with CNN-based baselines.

## License
This project is open-source under the MIT License.


