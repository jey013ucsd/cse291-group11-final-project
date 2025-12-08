#!/bin/bash

set -e  # Exit on error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

RESULTS_DIR="results_new/experiments"
METRICS_DIR="results_new/metrics"
NUM_EVAL_SAMPLES=1000
BATCH_SIZE=32

mkdir -p ${RESULTS_DIR}
mkdir -p ${METRICS_DIR}

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

run_experiment() {
    local exp_name=$1
    local exp_dir="${RESULTS_DIR}/${exp_name}"
    shift
    local args="$@"
    
    print_header "Running Experiment: ${exp_name}"
    echo "Arguments: ${args}"
    echo "Output directory: ${exp_dir}"
    echo ""
    
    python federated_learning.py ${args} --save_dir ${exp_dir}
    
    if [ $? -eq 0 ]; then
        print_success "Experiment ${exp_name} completed successfully"
    else
        print_error "Experiment ${exp_name} failed"
        return 1
    fi
    
    echo ""
}

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

print_header "EXPERIMENT 1: Centralized Training (Baseline)"
echo "Training GAN on all data without federated learning..."
echo "Training for 125 epochs (matching original milestone report)"
echo ""


mkdir -p ${RESULTS_DIR}/e1_centralized
if [ -f "models/cgan_scratch_G_128_125epochs.pth" ]; then
    cp models/cgan_scratch_G_128_125epochs.pth ${RESULTS_DIR}/e1_centralized/
    cp models/cgan_scratch_D_128_125epochs.pth ${RESULTS_DIR}/e1_centralized/
    [ -f "models/training_loss_curve.png" ] && cp models/training_loss_curve.png ${RESULTS_DIR}/e1_centralized/
    print_success "Centralized baseline copied (125 epochs)"
else
    print_error "Centralized model not found. Please run pretrain_GAN.py first to train 125 epochs"
fi

echo ""

run_experiment "e2_federated_iid" \
    --num_clients 5 \
    --num_rounds 25 \
    --local_epochs 5 \
    --batch_size ${BATCH_SIZE} \
    --partition iid \
    --seed 42

run_experiment "e3_noniid_alpha1.0" \
    --num_clients 5 \
    --num_rounds 25 \
    --local_epochs 5 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 1.0 \
    --seed 42

run_experiment "e4_noniid_alpha0.5" \
    --num_clients 5 \
    --num_rounds 25 \
    --local_epochs 5 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.5 \
    --seed 42

run_experiment "e5_noniid_alpha0.1" \
    --num_clients 5 \
    --num_rounds 25 \
    --local_epochs 5 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.1 \
    --seed 42

run_experiment "e6_clients_3" \
    --num_clients 3 \
    --num_rounds 21 \
    --local_epochs 6 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.5 \
    --seed 42

run_experiment "e7_clients_10" \
    --num_clients 10 \
    --num_rounds 30 \
    --local_epochs 4 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.5 \
    --seed 42


run_experiment "e8_rounds_60_epochs_2" \
    --num_clients 5 \
    --num_rounds 60 \
    --local_epochs 2 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.5 \
    --seed 42


run_experiment "e9_rounds_20_epochs_6" \
    --num_clients 5 \
    --num_rounds 20 \
    --local_epochs 6 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.5 \
    --seed 42

run_experiment "e10_noniid_alpha0.9" \
    --num_clients 5 \
    --num_rounds 25 \
    --local_epochs 5 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.9 \
    --seed 42

run_experiment "e11_noniid_alpha0.8" \
    --num_clients 5 \
    --num_rounds 25 \
    --local_epochs 5 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.8 \
    --seed 42

run_experiment "e12_noniid_alpha0.7" \
    --num_clients 5 \
    --num_rounds 25 \
    --local_epochs 5 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.7 \
    --seed 42

run_experiment "e13_noniid_alpha0.6" \
    --num_clients 5 \
    --num_rounds 25 \
    --local_epochs 5 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.6 \
    --seed 42

run_experiment "e14_noniid_alpha0.4" \
    --num_clients 5 \
    --num_rounds 25 \
    --local_epochs 5 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.4 \
    --seed 150

run_experiment "e15_noniid_alpha0.3" \
    --num_clients 5 \
    --num_rounds 25 \
    --local_epochs 5 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.3 \
    --seed 42

run_experiment "e16_noniid_alpha0.2" \
    --num_clients 5 \
    --num_rounds 25 \
    --local_epochs 5 \
    --batch_size ${BATCH_SIZE} \
    --partition non_iid \
    --alpha 0.2 \
    --seed 42


evaluate_model "e1_centralized" "cgan_scratch_G_128_125epochs.pth"
evaluate_model "e2_federated_iid" "federated_G_round_final.pth"
evaluate_model "e3_noniid_alpha1.0" "federated_G_round_final.pth"
evaluate_model "e4_noniid_alpha0.5" "federated_G_round_final.pth"
evaluate_model "e5_noniid_alpha0.1" "federated_G_round_final.pth"
evaluate_model "e6_clients_3" "federated_G_round_final.pth"
evaluate_model "e7_clients_10" "federated_G_round_final.pth"
evaluate_model "e8_rounds_60_epochs_2" "federated_G_round_final.pth"
evaluate_model "e9_rounds_20_epochs_6" "federated_G_round_final.pth"

print_header "GENERATING SAMPLE IMAGES"
echo "Creating visual samples from all trained models..."
echo ""

for exp_dir in ${RESULTS_DIR}/*/; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename $exp_dir)
        
        if [ -f "${exp_dir}cgan_scratch_G_128_125epochs.pth" ]; then
            model_path="${exp_dir}cgan_scratch_G_128_125epochs.pth"
        elif [ -f "${exp_dir}federated_G_round_final.pth" ]; then
            model_path="${exp_dir}federated_G_round_final.pth"
        else
            continue
        fi
        
        echo "Generating samples for ${exp_name}..."
        python generate_samples.py \
            --model "${model_path}" \
            --output "${exp_dir}generated_samples.png" \
            --num_samples 8 \
            --seed 42
    fi
done

print_success "Sample image generation completed"
echo ""

print_header "METRICS SUMMARY TABLE"

printf "%-30s | %-12s | %-20s | %-15s\n" "Experiment" "FID Score" "Inception Score" "Total Epochs"
echo "--------------------------------------------------------------------------------------------"

for exp_dir in ${METRICS_DIR}/*/; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename $exp_dir)
        json_file=$(find $exp_dir -name "*.json" | head -n 1)
        
        if [ -f "$json_file" ]; then
            fid=$(python -c "import json; print(f\"{json.load(open('$json_file'))['fid_score']:.4f}\")" 2>/dev/null || echo "N/A")
            is_mean=$(python -c "import json; d=json.load(open('$json_file')); print(f\"{d['inception_score_mean']:.4f}\")" 2>/dev/null || echo "N/A")
            is_std=$(python -c "import json; d=json.load(open('$json_file')); print(f\"{d['inception_score_std']:.4f}\")" 2>/dev/null || echo "N/A")
            
            # Calculate total epochs
            case $exp_name in
                "e1_centralized") epochs="125" ;;
                "e2_federated_iid") epochs="125" ;;
                "e3_noniid_alpha1.0") epochs="125" ;;
                "e4_noniid_alpha0.5") epochs="125" ;;
                "e5_noniid_alpha0.1") epochs="125" ;;
                "e6_clients_3") epochs="126" ;;
                "e7_clients_10") epochs="120" ;;
                "e8_rounds_60_epochs_2") epochs="120" ;;
                "e9_rounds_20_epochs_6") epochs="120" ;;
                *) epochs="N/A" ;;
            esac
            
            printf "%-30s | %-12s | %s ± %s | %-15s\n" "$exp_name" "$fid" "$is_mean" "$is_std" "$epochs"
        fi
    fi
done

echo ""


