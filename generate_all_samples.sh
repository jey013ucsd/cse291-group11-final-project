#!/bin/bash
# used claude to help generate script eval for all cases

set -e  # Exit on error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

RESULTS_DIR="results_new/experiments"
NUM_SAMPLES=8

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

generate_samples() {
    local exp_name=$1
    local model_name=$2
    local exp_dir="${RESULTS_DIR}/${exp_name}"
    local model_path="${exp_dir}/${model_name}"
    local output_path="${exp_dir}/generated_samples.png"
    
    if [ ! -f "${model_path}" ]; then
        print_warning "Model not found: ${model_path}"
        return 1
    fi
    
    echo "Generating samples from: ${exp_name}/${model_name}"
    
    python generate_samples.py \
        --model "${model_path}" \
        --output "${output_path}" \
        --num_samples ${NUM_SAMPLES} \
        --seed 42
    
    if [ $? -eq 0 ]; then
        print_success "Generated samples for ${exp_name}"
    else
        print_error "Failed to generate samples for ${exp_name}"
        return 1
    fi
    
    echo ""
}

print_header "Experiment 1: Centralized Baseline (125 epochs)"
generate_samples "e1_centralized" "cgan_scratch_G_128_125epochs.pth"

print_header "Experiment 2: Federated IID"
generate_samples "e2_federated_iid" "federated_G_round_final.pth"

print_header "Experiment 3: Non-IID (α=1.0)"
generate_samples "e3_noniid_alpha1.0" "federated_G_round_final.pth"

print_header "Experiment 4: Non-IID (α=0.5)"
generate_samples "e4_noniid_alpha0.5" "federated_G_round_final.pth"

print_header "Experiment 5: Non-IID (α=0.1)"
generate_samples "e5_noniid_alpha0.1" "federated_G_round_final.pth"

print_header "Experiment 6: 3 Clients"
generate_samples "e6_clients_3" "federated_G_round_final.pth"

print_header "Experiment 7: 10 Clients"
generate_samples "e7_clients_10" "federated_G_round_final.pth"

print_header "Experiment 8: 60 Rounds, 2 Local Epochs"
generate_samples "e8_rounds_60_epochs_2" "federated_G_round_final.pth"

print_header "Experiment 9: 20 Rounds, 6 Local Epochs"
generate_samples "e9_rounds_20_epochs_6" "federated_G_round_final.pth"

print_header "GENERATION COMPLETE!"

echo "Generated sample images saved to:"
echo ""
for exp_dir in ${RESULTS_DIR}/*/; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename $exp_dir)
        normal_file="${exp_dir}generated_samples_normal.png"
        pneumonia_file="${exp_dir}generated_samples_pneumonia.png"
        if [ -f "$normal_file" ] && [ -f "$pneumonia_file" ]; then
            echo "${exp_name}/generated_samples_normal.png"
            echo "${exp_name}/generated_samples_pneumonia.png"
        else
            echo "${exp_name}/ (files not found)"
        fi
    fi
done
