# 3DFFL-Privacy-Preserving-Federated-Few-Shot-Learning-for-3D-Point-Clouds-in-Autonomous-Vehicles


This project presents a comprehensive framework for federated few-shot learning (3DFFL), focusing on 3D point cloud classification. The approach integrates Federated Learning (FL) with Few-Shot Learning (FSL) techniques, using PointNet++ for feature extraction and ProtoNet for classification. The framework ensures data privacy and leverages collaborative learning to handle data scarcity and heterogeneity.

## Key Features

- **PointNet++ for feature extraction**
- **Attention mechanism**
- **Loss functions for embedding and learnable tasks**
- **Differential privacy updates**
- **Data augmentation with Mixup**
- **Federated training process**
- **Few-shot learning with Prototypical Networks**

## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Loading Data](#loading-data)
   - [Federated Training](#federated-training)
   - [Few-Shot Training](#few-shot-training)
   - [Saving and Loading the Model](#saving-and-loading-the-model)
4. [Key Components](#key-components)
   - [PointNet++](#pointnet)
   - [Attention Layer](#attention-layer)
   - [Loss Functions](#loss-functions)
   - [Differential Privacy](#differential-privacy)
   - [Data Augmentation](#data-augmentation)
   - [Federated Training Process](#federated-training-process)
   - [Few-Shot Learning](#few-shot-learning)
5. [Example](#example)
6. [References](#references)

## Requirements
- Python 3.x
- PyTorch
- NumPy
- scikit-learn

## Installation
```bash
pip install torch numpy scikit-learn
```

## Usage

### Loading Data
The `load_data` function generates synthetic data for testing purposes:
```python
def load_data():
    N = 100  # Number of samples
    P = 1024 # Number of points per sample
    D = 3    # Dimensionality of each point
    C = 10   # Number of classes
    return [(torch.rand(N, D, P), torch.randint(0, 2, (N, C)).float()) for _ in range(5)]
```

### Federated Training
To perform federated training with the provided framework:
```python
local_datasets = load_data()
global_model = GlobalModel(input_dim=3, feature_dim=128, num_classes=10).to('cuda' if torch.cuda.is_available() else 'cpu')
lambda1, lambda2, lambda3 = 1.0, 1.0, 0.1

trained_global_model = federated_training_process(num_rounds=50, local_datasets=local_datasets, global_model=global_model, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, apply_privacy=True)
```

### Few-Shot Training
To perform few-shot training with the trained global model:
```python
n_way, k_shot, q_query = 5, 5, 15
few_shot_data = [torch.rand((100, 3, 1024)) for _ in range(n_way)]

trained_few_shot_model = few_shot_training(trained_global_model, few_shot_data, num_rounds=10, n_way=n_way, k_shot=k_shot, q_query=q_query)
```

### Saving and Loading the Model
To save the trained model:
```python
torch.save(trained_few_shot_model.state_dict(), 'few_shot_trained_model.pth')
```
To load the trained model:
```python
model = GlobalModel(input_dim=3, feature_dim=128, num_classes=10)
model.load_state_dict(torch.load('few_shot_trained_model.pth'))
```

## Key Components

### PointNet++
The `PointNetPP` class implements a simplified version of PointNet++ for feature extraction:
```python
class PointNetPP(nn.Module):
    # Initialization and forward methods here
```

### Attention Layer
The `AttentionLayer` class implements an attention mechanism:
```python
class AttentionLayer(nn.Module):
    # Initialization and forward methods here
```

### Loss Functions
The following functions calculate the different loss components:
- `calculate_embedding_loss`
- `calculate_learnable_loss`
- `calculate_overall_loss`

### Differential Privacy
The `differential_privacy_update` function applies differential privacy updates to the model:
```python
def differential_privacy_update(model, noise_multiplier=0.1):
    # Implementation here
```

### Data Augmentation
The following functions handle data augmentation using Mixup:
- `mixup_data`
- `mixup_criterion`

### Federated Training Process
The `federated_training_process` function performs federated training:
```python
def federated_training_process(num_rounds, local_datasets, global_model, lambda1, lambda2, lambda3, apply_privacy=False):
    # Implementation here
```

### Few-Shot Learning
The `ProtoNet` class and `few_shot_training` function implement few-shot learning using Prototypical Networks:
```python
class ProtoNet(MetaTemplate):
    # Initialization and methods here
```
```python
def few_shot_training(global_model, data, num_rounds, n_way, k_shot, q_query):
    # Implementation here
```

## Example
The following example demonstrates how to use the provided framework:
```python
# Main execution
local_datasets = load_data()
global_model = GlobalModel(input_dim=3, feature_dim=128, num_classes=10).to('cuda' if torch.cuda.is_available() else 'cpu')
lambda1, lambda2, lambda3 = 1.0, 1.0, 0.1

# Federated training
trained_global_model = federated_training_process(num_rounds=50, local_datasets=local_datasets, global_model=global_model, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, apply_privacy=True)

# Few-shot learning
n_way, k_shot, q_query = 5, 5, 15
few_shot_data = [torch.rand((100, 3, 1024)) for _ in range(n_way)]
trained_few_shot_model = few_shot_training(trained_global_model, few_shot_data, num_rounds=10, n_way=n_way, k_shot=k_shot, q_query=q_query)

# Save the trained model state
torch.save(trained_few_shot_model.state_dict(), 'few_shot_trained_model.pth')
```
