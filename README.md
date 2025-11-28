# cse291-group11-final-project

Medical Imaging GAN with Federated Learning

# Setup

## Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Download dataset
```bash
python get_dataset.py
```

## Preprocess dataset
```bash
python preprocess_dataset.py
```

- Images are converted to grayscale.  
- Each image is center-cropped to a square.  
- Cropped images are resized to 128x128 
- The processed image is saved as a **.png**, and its array + metadata as a **.pkl**.

# Training

## Option 1: Centralized Training (Baseline)

### Train cGAN from scratch
```bash
python pretrain_GAN.py
```

### Fine-tune pretrained BigGAN
```bash
python finetune_BigGAN.py
```

## Option 2: Federated Learning

Train a GAN in a federated setting, simulating multiple hospitals with private data:

### Basic Usage (IID data distribution)
```bash
python federated_learning.py --num_clients 5 --num_rounds 10 --local_epochs 3
```

### Non-IID data distribution (simulates data heterogeneity)
```bash
python federated_learning.py --num_clients 5 --num_rounds 10 --local_epochs 3 --partition non_iid --alpha 0.5
```

### Full Options
```bash
python federated_learning.py \
  --config config.yaml \
  --num_clients 5 \
  --num_rounds 10 \
  --local_epochs 3 \
  --batch_size 32 \
  --partition iid \
  --alpha 0.5 \
  --save_dir models/federated \
  --seed 42
```

### Parameters:
- `--num_clients`: Number of federated clients (simulated hospitals)
- `--num_rounds`: Number of federated learning rounds
- `--local_epochs`: Local training epochs per round
- `--batch_size`: Batch size for training
- `--partition`: Data partitioning strategy (`iid` or `non_iid`)
- `--alpha`: Dirichlet parameter for non-IID partitioning (lower = more skewed, e.g., 0.1 is very skewed, 1.0 is more uniform)
- `--save_dir`: Directory to save trained models
- `--seed`: Random seed for reproducibility

# Model Outputs

- **Centralized models**: Saved in `models/` directory
- **Federated models**: Saved in `models/federated/` directory
- **Training curves**: PNG files showing loss over time