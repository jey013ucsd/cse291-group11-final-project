#!/bin/bash

################################################################################
# Quick Test Script - Verify Setup Before Running Full Experiments
# Runs a minimal experiment to check if everything works
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Quick Test - Federated Learning Setup Verification${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Check if dataset exists
echo "1. Checking dataset..."
if [ -d "dataset/train/NORMAL" ] && [ -d "dataset/train/PNEUMONIA" ]; then
    NORMAL_COUNT=$(ls dataset/train/NORMAL/*.pkl 2>/dev/null | wc -l)
    PNEUMONIA_COUNT=$(ls dataset/train/PNEUMONIA/*.pkl 2>/dev/null | wc -l)
    
    if [ $NORMAL_COUNT -gt 0 ] && [ $PNEUMONIA_COUNT -gt 0 ]; then
        echo -e "${GREEN}✓ Dataset found: ${NORMAL_COUNT} NORMAL, ${PNEUMONIA_COUNT} PNEUMONIA samples${NC}"
    else
        echo -e "${RED}✗ Dataset PKL files not found. Run: python get_dataset.py && python preprocess_dataset.py${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Dataset directory not found. Run: python get_dataset.py && python preprocess_dataset.py${NC}"
    exit 1
fi

echo ""

# Run quick federated learning test
echo "2. Running quick federated learning test (2 clients, 2 rounds, 1 epoch)..."
echo "   This should take 2-5 minutes..."
echo ""

python federated_learning.py \
    --num_clients 2 \
    --num_rounds 2 \
    --local_epochs 1 \
    --batch_size 16 \
    --partition iid \
    --save_dir results/test_quick \
    --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Federated learning test passed!${NC}"
else
    echo ""
    echo -e "${RED}✗ Federated learning test failed${NC}"
    exit 1
fi

echo ""

# Test evaluation script
echo "3. Testing evaluation script (100 samples)..."
echo "   This should take 1-2 minutes..."
echo ""

python evaluate_metrics.py \
    --model results/test_quick/federated_G_round_final.pth \
    --data_dir dataset \
    --num_samples 100 \
    --batch_size 50 \
    --output_dir results/test_quick/metrics

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Evaluation test passed!${NC}"
else
    echo ""
    echo -e "${RED}✗ Evaluation test failed${NC}"
    exit 1
fi

echo ""

# Test plotting script
echo "4. Testing plotting script..."
echo ""

python plot_comparison.py \
    --metrics_dir results/test_quick/metrics \
    --results_dir results/test_quick \
    --output_dir results/test_quick/plots

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Plotting test passed!${NC}"
else
    echo ""
    echo -e "${RED}✗ Plotting test failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${GREEN}✓ All tests passed! Your setup is working correctly.${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo "You can now run the full experiment suite with:"
echo "  ./run_experiments.sh"
echo ""
echo "Test results saved to: results/test_quick/"
echo ""

