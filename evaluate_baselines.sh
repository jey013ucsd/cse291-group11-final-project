#!/bin/bash

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

MODELS_DIR="models"
METRICS_DIR="results_new/metrics_baselines"
DATA_DIR="dataset"
NUM_SAMPLES=500

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

mkdir -p ${METRICS_DIR}

print_header "Evaluating Unconditional Baselines"
echo "Number of samples: ${NUM_SAMPLES}"
echo "Metric: KID (Kernel Inception Distance)"
echo ""

if [ -f "${MODELS_DIR}/unconditional_gan_G_128_125epochs.pth" ]; then
    print_header "Evaluating Unconditional GAN"
    python evaluate_baselines.py \
        --model "${MODELS_DIR}/unconditional_gan_G_128_125epochs.pth" \
        --model_type unconditional_gan \
        --data_dir "${DATA_DIR}" \
        --num_samples ${NUM_SAMPLES} \
        --output_dir "${METRICS_DIR}"
    print_success "Unconditional GAN evaluation complete"
else
    print_warning "Unconditional GAN model not found at ${MODELS_DIR}/unconditional_gan_G_128_125epochs.pth"
    echo "  Run: python pretrain_unconditional_GAN.py"
fi

echo ""


if [ -f "${MODELS_DIR}/vae_128_125epochs.pth" ]; then
    print_header "Evaluating VAE"
    python evaluate_baselines.py \
        --model "${MODELS_DIR}/vae_128_125epochs.pth" \
        --model_type vae \
        --data_dir "${DATA_DIR}" \
        --num_samples ${NUM_SAMPLES} \
        --output_dir "${METRICS_DIR}"
    print_success "VAE evaluation complete"
else
    print_warning "VAE model not found at ${MODELS_DIR}/vae_128_125epochs.pth"
    echo "  Run: python pretrain_VAE.py"
fi


echo ""
print_header "BASELINE COMPARISON BY CLASS"
echo ""
printf "%-25s | %-12s | %-12s | %-12s\n" "Model" "KID NORMAL" "KID PNEUM" "KID AVG"
echo "--------------------------------------------------------------------------------"

# Unconditional GAN
if [ -f "${METRICS_DIR}/metrics_unconditional_gan_unconditional_gan_G_128_125epochs.json" ]; then
    kid_normal=$(python3 -c "import json; print(f\"{json.load(open('${METRICS_DIR}/metrics_unconditional_gan_unconditional_gan_G_128_125epochs.json'))['kid_normal']:.4f}\")")
    kid_pneum=$(python3 -c "import json; print(f\"{json.load(open('${METRICS_DIR}/metrics_unconditional_gan_unconditional_gan_G_128_125epochs.json'))['kid_pneumonia']:.4f}\")")
    kid_avg=$(python3 -c "import json; print(f\"{json.load(open('${METRICS_DIR}/metrics_unconditional_gan_unconditional_gan_G_128_125epochs.json'))['kid_combined']:.4f}\")")
    printf "%-25s | %-12s | %-12s | %-12s\n" "Unconditional GAN" "$kid_normal" "$kid_pneum" "$kid_avg"
fi

# VAE
if [ -f "${METRICS_DIR}/metrics_vae_vae_128_125epochs.json" ]; then
    kid_normal=$(python3 -c "import json; print(f\"{json.load(open('${METRICS_DIR}/metrics_vae_vae_128_125epochs.json'))['kid_normal']:.4f}\")")
    kid_pneum=$(python3 -c "import json; print(f\"{json.load(open('${METRICS_DIR}/metrics_vae_vae_128_125epochs.json'))['kid_pneumonia']:.4f}\")")
    kid_avg=$(python3 -c "import json; print(f\"{json.load(open('${METRICS_DIR}/metrics_vae_vae_128_125epochs.json'))['kid_combined']:.4f}\")")
    printf "%-25s | %-12s | %-12s | %-12s\n" "VAE" "$kid_normal" "$kid_pneum" "$kid_avg"
fi

