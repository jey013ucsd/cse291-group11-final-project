#!/bin/bash

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

RESULTS_DIR="results_new/experiments"
METRICS_DIR="results_new/metrics"
DATA_DIR="dataset"
NUM_SAMPLES=500 # per class

print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

mkdir -p ${METRICS_DIR}

print_header "Evaluating by Class: NORMAL vs PNEUMONIA"
echo "Samples per class: ${NUM_SAMPLES}"
echo ""

python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e1_centralized/cgan_scratch_G_128_125epochs.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e1_centralized"

python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e2_federated_iid/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e2_federated_iid"

python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e3_noniid_alpha1.0/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e3_noniid_alpha1.0"

python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e4_noniid_alpha0.5/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e4_noniid_alpha0.5"

python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e5_noniid_alpha0.1/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e5_noniid_alpha0.1"

python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e6_clients_3/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e6_clients_3"

python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e7_clients_10/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e7_clients_10"

python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e8_rounds_60_epochs_2/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e8_rounds_60_epochs_2"

python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e9_rounds_20_epochs_6/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e9_rounds_20_epochs_6"

python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e10_noniid_alpha0.9/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e10_noniid_alpha0.9"

python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e11_noniid_alpha0.8/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e11_noniid_alpha0.8"

python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e12_noniid_alpha0.7/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e12_noniid_alpha0.7"

python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e13_noniid_alpha0.6/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e13_noniid_alpha0.6"

python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e14_noniid_alpha0.4/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e14_noniid_alpha0.4"


python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e15_noniid_alpha0.3/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e15_noniid_alpha0.3"


python evaluate_by_class.py \
    --model "${RESULTS_DIR}/e16_noniid_alpha0.2/federated_G_round_final.pth" \
    --data_dir "${DATA_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --output_dir "${METRICS_DIR}/e16_noniid_alpha0.2"

echo ""
printf "%-30s | %-12s | %-12s | %-12s\n" "Experiment" "FID NORMAL" "FID PNEUM" "FID AVG"
echo "----------------------------------------------------------------------------------------"

for exp in "e1_centralized" "e2_federated_iid" "e3_noniid_alpha1.0" "e4_noniid_alpha0.5" "e5_noniid_alpha0.1" \
           "e6_clients_3" "e7_clients_10" "e8_rounds_60_epochs_2" "e9_rounds_20_epochs_6" \
           "e10_noniid_alpha0.9" "e11_noniid_alpha0.8" "e12_noniid_alpha0.7" \
           "e13_noniid_alpha0.6" "e14_noniid_alpha0.4" "e15_noniid_alpha0.3" "e16_noniid_alpha0.2"; do
    json_file="${METRICS_DIR}/${exp}/metrics_by_class_"*.json
    if ls ${json_file} 1> /dev/null 2>&1; then
        json_file=$(ls ${json_file} | head -n 1)
        fid_normal=$(python3 -c "import json; print(f\"{json.load(open('$json_file'))['fid_normal']:.2f}\")")
        fid_pneum=$(python3 -c "import json; print(f\"{json.load(open('$json_file'))['fid_pneumonia']:.2f}\")")
        fid_avg=$(python3 -c "import json; print(f\"{json.load(open('$json_file'))['fid_combined']:.2f}\")")
        printf "%-30s | %-12s | %-12s | %-12s\n" "$exp" "$fid_normal" "$fid_pneum" "$fid_avg"
    fi
done

echo ""
printf "%-30s | %-12s | %-12s | %-12s\n" "Experiment" "KID NORMAL" "KID PNEUM" "KID AVG"
echo "----------------------------------------------------------------------------------------"

for exp in "e1_centralized" "e2_federated_iid" "e3_noniid_alpha1.0" "e4_noniid_alpha0.5" "e5_noniid_alpha0.1" \
           "e6_clients_3" "e7_clients_10" "e8_rounds_60_epochs_2" "e9_rounds_20_epochs_6" \
           "e10_noniid_alpha0.9" "e11_noniid_alpha0.8" "e12_noniid_alpha0.7" \
           "e13_noniid_alpha0.6" "e14_noniid_alpha0.4" "e15_noniid_alpha0.3" "e16_noniid_alpha0.2"; do
    json_file="${METRICS_DIR}/${exp}/metrics_by_class_"*.json
    if ls ${json_file} 1> /dev/null 2>&1; then
        json_file=$(ls ${json_file} | head -n 1)
        kid_normal=$(python3 -c "import json; d=json.load(open('$json_file')); print(f\"{d.get('kid_normal', 'N/A'):.4f}\" if 'kid_normal' in d else 'N/A')")
        kid_pneum=$(python3 -c "import json; d=json.load(open('$json_file')); print(f\"{d.get('kid_pneumonia', 'N/A'):.4f}\" if 'kid_pneumonia' in d else 'N/A')")
        kid_avg=$(python3 -c "import json; d=json.load(open('$json_file')); print(f\"{d.get('kid_combined', 'N/A'):.4f}\" if 'kid_combined' in d else 'N/A')")
        printf "%-30s | %-12s | %-12s | %-12s\n" "$exp" "$kid_normal" "$kid_pneum" "$kid_avg"
    fi
done

echo ""
printf "%-30s | %-12s | %-12s | %-12s\n" "Experiment" "IS NORMAL" "IS PNEUM" "IS AVG"
echo "----------------------------------------------------------------------------------------"

for exp in "e1_centralized" "e2_federated_iid" "e3_noniid_alpha1.0" "e4_noniid_alpha0.5" "e5_noniid_alpha0.1" \
           "e6_clients_3" "e7_clients_10" "e8_rounds_60_epochs_2" "e9_rounds_20_epochs_6" \
           "e10_noniid_alpha0.9" "e11_noniid_alpha0.8" "e12_noniid_alpha0.7" \
           "e13_noniid_alpha0.6" "e14_noniid_alpha0.4" "e15_noniid_alpha0.3" "e16_noniid_alpha0.2"; do
    json_file="${METRICS_DIR}/${exp}/metrics_by_class_"*.json
    if ls ${json_file} 1> /dev/null 2>&1; then
        json_file=$(ls ${json_file} | head -n 1)
        is_normal=$(python3 -c "import json; print(f\"{json.load(open('$json_file'))['is_mean_normal']:.3f}\")")
        is_pneum=$(python3 -c "import json; print(f\"{json.load(open('$json_file'))['is_mean_pneumonia']:.3f}\")")
        is_avg=$(python3 -c "import json; print(f\"{json.load(open('$json_file'))['is_mean_combined']:.3f}\")")
        printf "%-30s | %-12s | %-12s | %-12s\n" "$exp" "$is_normal" "$is_pneum" "$is_avg"
    fi
done
