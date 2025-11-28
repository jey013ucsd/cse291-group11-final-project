#!/bin/bash

################################################################################
# Automated Experiment Runner for Federated Learning GAN
# Runs all experiments and evaluates models with FID/IS scores
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RESULTS_DIR="results/experiments"
METRICS_DIR="results/metrics"
NUM_EVAL_SAMPLES=1000
BATCH_SIZE=32

# Create directories
mkdir -p ${RESULTS_DIR}
mkdir -p ${METRICS_DIR}

# Function to print colored output
print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local exp_dir="${RESULTS_DIR}/${exp_name}"
    shift
    local args="$@"
    
    print_header "Running Experiment: ${exp_name}"
    echo "Arguments: ${args}"
    echo "Output directory: ${exp_dir}"
    echo ""
    
    # Run the experiment
    python federated_learning.py ${args} --save_dir ${exp_dir}
    
    if [ $? -eq 0 ]; then
        print_success "Experiment ${exp_name} completed successfully"
    else
        print_error "Experiment ${exp_name} failed"
        return 1
    fi
    
    echo ""
}

# Function to evaluate model
evaluate_model() {
    local exp_name=$1
    local model_name=$2
    local exp_dir="${RESULTS_DIR}/${exp_name}"
    
    if [ ! -f "${exp_dir}/${model_name}" ]; then
        print_warning "Model not found: ${exp_dir}/${model_name}"
        return 1
    fi
    
    print_header "Evaluating: ${exp_name}/${model_name}"
    
    python evaluate_metrics.py \
        --model ${exp_dir}/${model_name} \
        --data_dir dataset \
        --num_samples ${NUM_EVAL_SAMPLES} \
        --batch_size 50 \
        --output_dir ${METRICS_DIR}/${exp_name}
    
    if [ $? -eq 0 ]; then
        print_success "Evaluation of ${exp_name} completed"
    else
        print_error "Evaluation of ${exp_name} failed"
    fi
    
    echo ""
}

# Start timestamp
START_TIME=$(date +%s)
print_header "Starting Automated Experiments"
echo "Start time: $(date)"
echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

################################################################################
# EXPERIMENT 1: Centralized Training (Baseline)
################################################################################

print_header "EXPERIMENT 1: Centralized Training (Baseline)"
echo "Training GAN on all data without federated learning..."
echo ""

python pretrain_GAN.py

if [ $? -eq 0 ]; then
    print_success "Centralized baseline training completed"
    # Copy to results directory for consistency
    mkdir -p ${RESULTS_DIR}/e1_centralized
    cp models/cgan_scratch_G_128_20epochs.pth ${RESULTS_DIR}/e1_centralized/
    cp models/cgan_scratch_D_128_20epochs.pth ${RESULTS_DIR}/e1_centralized/
    cp models/training_loss_curve.png ${RESULTS_DIR}/e1_centralized/
else
    print_error "Centralized baseline training failed"
fi

echo ""

################################################################################
# EXPERIMENT 2: Federated IID
################################################################################

run_experiment "e2_federated_iid" \
    --num_clients 5 \
    --num_rounds 10 \
    --local_epochs 3 \
    --batch_size ${BATCH_SIZE} \
    --partition iid \
    --seed 42

################################################################################
# EXPERIMENT 3: Federated Non-IID (Mild Heterogeneity)
################################################################################

run_experiment "e3_noniid_alpha1.0" \
    --num_clients 5 \
    --num_rounds 10 \
    --local_epochs 3 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 1.0 \
    --seed 42

################################################################################
# EXPERIMENT 4: Federated Non-IID (Moderate Heterogeneity) - Main Result
################################################################################

run_experiment "e4_noniid_alpha0.5" \
    --num_clients 5 \
    --num_rounds 10 \
    --local_epochs 3 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.5 \
    --seed 42

################################################################################
# EXPERIMENT 5: Federated Non-IID (High Heterogeneity)
################################################################################

run_experiment "e5_noniid_alpha0.1" \
    --num_clients 5 \
    --num_rounds 10 \
    --local_epochs 3 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.1 \
    --seed 42

################################################################################
# EXPERIMENT 6: Few Large Clients
################################################################################

run_experiment "e6_clients_3" \
    --num_clients 3 \
    --num_rounds 15 \
    --local_epochs 5 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.5 \
    --seed 42

################################################################################
# EXPERIMENT 7: Many Small Clients
################################################################################

run_experiment "e7_clients_10" \
    --num_clients 10 \
    --num_rounds 10 \
    --local_epochs 2 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.5 \
    --seed 42

################################################################################
# EXPERIMENT 8: Communication Efficiency - More Rounds, Less Local Work
################################################################################

run_experiment "e8_rounds_20_epochs_1" \
    --num_clients 5 \
    --num_rounds 20 \
    --local_epochs 1 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.5 \
    --seed 42

################################################################################
# EXPERIMENT 9: Communication Efficiency - Fewer Rounds, More Local Work
################################################################################

run_experiment "e9_rounds_5_epochs_6" \
    --num_clients 5 \
    --num_rounds 5 \
    --local_epochs 6 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.5 \
    --seed 42

################################################################################
# EVALUATION PHASE: Calculate FID and Inception Scores
################################################################################

print_header "EVALUATION PHASE: Calculating FID and Inception Scores"
echo "This may take some time..."
echo ""

# Evaluate centralized baseline
evaluate_model "e1_centralized" "cgan_scratch_G_128_20epochs.pth"

# Evaluate all federated experiments (using final models)
evaluate_model "e2_federated_iid" "federated_G_round_final.pth"
evaluate_model "e3_noniid_alpha1.0" "federated_G_round_final.pth"
evaluate_model "e4_noniid_alpha0.5" "federated_G_round_final.pth"
evaluate_model "e5_noniid_alpha0.1" "federated_G_round_final.pth"
evaluate_model "e6_clients_3" "federated_G_round_final.pth"
evaluate_model "e7_clients_10" "federated_G_round_final.pth"
evaluate_model "e8_rounds_20_epochs_1" "federated_G_round_final.pth"
evaluate_model "e9_rounds_5_epochs_6" "federated_G_round_final.pth"

################################################################################
# SUMMARY AND COMPARISON
################################################################################

print_header "EXPERIMENT SUMMARY"

echo "All experiments completed!"
echo ""
echo "Results directory structure:"
echo "  ${RESULTS_DIR}/"
echo "    ├── e1_centralized/          (Baseline)"
echo "    ├── e2_federated_iid/        (IID partitioning)"
echo "    ├── e3_noniid_alpha1.0/      (Mild heterogeneity)"
echo "    ├── e4_noniid_alpha0.5/      (Moderate heterogeneity)"
echo "    ├── e5_noniid_alpha0.1/      (High heterogeneity)"
echo "    ├── e6_clients_3/            (Few large clients)"
echo "    ├── e7_clients_10/           (Many small clients)"
echo "    ├── e8_rounds_20_epochs_1/   (More rounds, less local)"
echo "    └── e9_rounds_5_epochs_6/    (Fewer rounds, more local)"
echo ""

echo "Metrics saved to: ${METRICS_DIR}/"
echo ""

# Create summary table
print_header "METRICS SUMMARY TABLE"

printf "%-30s | %-12s | %-20s\n" "Experiment" "FID Score" "Inception Score"
echo "--------------------------------------------------------------------------------"

for exp_dir in ${METRICS_DIR}/*/; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename $exp_dir)
        json_file=$(find $exp_dir -name "*.json" | head -n 1)
        
        if [ -f "$json_file" ]; then
            fid=$(python -c "import json; print(f\"{json.load(open('$json_file'))['fid_score']:.4f}\")" 2>/dev/null || echo "N/A")
            is_mean=$(python -c "import json; d=json.load(open('$json_file')); print(f\"{d['inception_score_mean']:.4f}\")" 2>/dev/null || echo "N/A")
            is_std=$(python -c "import json; d=json.load(open('$json_file')); print(f\"{d['inception_score_std']:.4f}\")" 2>/dev/null || echo "N/A")
            
            printf "%-30s | %-12s | %s ± %s\n" "$exp_name" "$fid" "$is_mean" "$is_std"
        fi
    fi
done

echo ""

# End timestamp
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

print_header "ALL EXPERIMENTS COMPLETE!"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "End time: $(date)"
echo ""

print_success "Next steps:"
echo "  1. Review training curves in ${RESULTS_DIR}/*/federated_training_curves.png"
echo "  2. Compare metrics in ${METRICS_DIR}/"
echo "  3. Generate comparison plots with: python plot_comparison.py"
echo ""

