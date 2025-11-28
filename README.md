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

# Automated Experiments
## Run All Experiments Automatically

To run all experiments and evaluations in one go:

```bash
./run_experiments.sh
```

This script will:
1. Train centralized baseline (Experiment 1)
2. Train 8 different federated configurations (Experiments 2-9)
3. Evaluate all models with FID and Inception Score
4. Generate a summary table of results

**Note:** This will take several hours to complete all experiments.

### What Gets Run:

- **E1**: Centralized baseline (20 epochs)
- **E2**: Federated IID (5 clients, 10 rounds, 3 local epochs)
- **E3**: Non-IID α=1.0 - Mild heterogeneity
- **E4**: Non-IID α=0.5 - Moderate heterogeneity
- **E5**: Non-IID α=0.1 - High heterogeneity
- **E6**: 3 clients (fewer, larger institutions)
- **E7**: 10 clients (many smaller institutions)
- **E8**: 20 rounds × 1 epoch (communication study)
- **E9**: 5 rounds × 6 epochs (computation study)

Results saved to:
- `results/experiments/` - Model checkpoints and training curves
- `results/metrics/` - FID and Inception Score evaluations

## Evaluate Individual Models

To evaluate a specific trained model:

```bash
python evaluate_metrics.py \
  --model models/federated/federated_G_round_final.pth \
  --data_dir dataset \
  --num_samples 1000 \
  --output_dir results/metrics
```

### Options:
- `--model`: Path to generator checkpoint (.pth file)
- `--data_dir`: Path to dataset (default: dataset)
- `--num_samples`: Number of samples to generate for evaluation (default: 1000)
- `--batch_size`: Batch size for computation (default: 50)
- `--device`: Device to use (cuda/mps/cpu)
- `--output_dir`: Where to save metrics (default: results/metrics)

## Visualize Results

After running experiments, generate comparison plots:

```bash
python plot_comparison.py
```

This creates:
- `results/plots/fid_comparison.png` - FID scores across all experiments
- `results/plots/inception_comparison.png` - Inception scores comparison
- `results/plots/combined_metrics.png` - Both metrics together
- `results/plots/training_curves_all.png` - Training loss over rounds
- `results/plots/heterogeneity_study.png` - Impact of data heterogeneity
- `results/plots/client_scaling.png` - Impact of number of clients
- `results/plots/results_report.md` - Markdown summary report

### Custom plot directories:

```bash
python plot_comparison.py \
  --metrics_dir results/metrics \
  --results_dir results/experiments \
  --output_dir results/plots
```

# Evaluation Metrics

## FID Score (Fréchet Inception Distance)
- Measures similarity between real and generated image distributions
- **Lower is better**
- Typical range: 20-200 (lower = more realistic images)

## Inception Score
- Measures quality and diversity of generated images
- **Higher is better**
- Typical range: 1-10 (higher = better quality and diversity)