# Brain Tumor Classification using CNN and Transfer Learning

This project uses Convolutional Neural Networks and transfer learning with MobileNetV2 and ResNet50 to classify brain tumor images into two categories.

## Dataset
The model is trained on this dataset from [Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection).

- Directory structure:  
  `brain tumor/`  
  ├── `no/`  
  └── `yes/`

- Total images:  
  - Training: 203  
  - Validation: 50  
- Classes: 2

## Preprocessing

- Normalization: Rescale pixel values to [0, 1]
- Augmentation:
  - Rotation (±10°)
  - Width/height shift (±5%)
  - Zoom (±5%)
  - Horizontal and vertical flips
  - Brightness adjustment ([0.5, 1.4])
- Input shape: `(224, 224, 3)`
- One-hot encoding for labels

## Models

### Model 1: Custom CNN

```python
Sequential([
  Conv2D(64), MaxPooling2D(),
  Conv2D(32), MaxPooling2D(),
  Conv2D(32), MaxPooling2D(),
  Flatten(),
  Dense(512, relu), Dropout(0.5),
  Dense(2, softmax)
])
```
Optimizer: Adam (lr=0.00008)  
Loss: Categorical Crossentropy  
Epochs: 20  
Train Accuracy: ~83%  
Val Accuracy: ~76%

### Model 2: MobileNetV2 (Transfer Learning)

```python
Sequential([
  MobileNetV2(weights='imagenet', include_top=False, trainable=False),
  GlobalAveragePooling2D(),
  Dense(512, relu), Dropout(0.2),
  Dense(2, softmax)
])
```
Optimizer: Adam (lr=0.00008)  
Loss: Categorical Crossentropy  
Epochs: 20  
Train Accuracy: ~96%  
Val Accuracy: ~98% (best epoch)

### Model 3: ResNet50 (Transfer Learning)

```python
Sequential([
  ResNet50(weights='imagenet', include_top=False, trainable=False),
  Flatten(),
  BatchNormalization(),
  Dense(32, relu),
  Dense(2, softmax)
])
```
Optimizer: RMSProp (lr=1e-5)  
Loss: Categorical Crossentropy  
Epochs: 20  
Train Accuracy: ~83%  
Val Accuracy: ~78% (best epoch)

## Evaluation

- Metrics: Accuracy and Loss
- Visualization: Training and validation curves for each model
- MobileNetV2 outperforms custom CNN and ResNet50 on small dataset

## How to Run

- Mount Google Drive in Colab
- Copy dataset to working directory
- Run the notebook cells step by step
- Observe training logs and plots
- Use best performing model for inference

## Requirements

```bash
tensorflow
keras
numpy
matplotlib
opencv-python
scikit-learn
```

## Observations

- Transfer learning improves performance, especially on small datasets
- MobileNetV2 generalizes better with frozen layers
- Custom CNN still performs decently but is limited without deeper architecture

## Next Steps

- Fine-tune last layers of MobileNetV2 or ResNet50
- Add early stopping and learning rate scheduler
- Use Grad-CAM for interpretability
- Expand dataset size
