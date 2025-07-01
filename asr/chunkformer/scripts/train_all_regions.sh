#!/bin/bash

# Chunkformer Training Script for All VIMD Regions
# =================================================
# This script trains Chunkformer models on all regions of the VIMD dataset
# Usage: bash scripts/train_all_regions.sh

set -e  # Exit on any error

# Configuration
CONFIG_FILE="configs/config_vimd.yaml"
REGIONS=("All" "Central" "South" "North")
LOG_LEVEL="INFO"
DEVICE="cuda"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            print_status "CUDA is available"
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        else
            print_warning "nvidia-smi command failed"
            return 1
        fi
    else
        print_warning "nvidia-smi not found"
        return 1
    fi
}

# Function to check Python environment
check_environment() {
    print_status "Checking Python environment..."
    
    # Check Python version
    python_version=$(python --version 2>&1)
    print_status "Python version: $python_version"
    
    # Check if required packages are installed
    required_packages=("torch" "torchaudio" "datasets" "wandb" "numpy" "soundfile" "jiwer" "yaml")
    
    for package in "${required_packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            print_status "✓ $package is installed"
        else
            print_error "✗ $package is NOT installed"
            return 1
        fi
    done
    
    # Check WeNet
    if python -c "import wenet" 2>/dev/null; then
        print_status "✓ WeNet is installed"
    else
        print_error "✗ WeNet is NOT installed"
        return 1
    fi
}

# Function to train a single region
train_region() {
    local region=$1
    print_status "Starting training for region: $region"
    
    # Create a timestamp for this run
    timestamp=$(date '+%Y%m%d_%H%M%S')
    log_file="logs/chunkformer_${region,,}_${timestamp}.log"
    
    # Run training
    python train_chunkformer.py \
        --config "$CONFIG_FILE" \
        --region "$region" \
        --log-level "$LOG_LEVEL" \
        --device "$DEVICE" 2>&1 | tee "$log_file"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_success "Training completed for region: $region"
        print_status "Log file: $log_file"
    else
        print_error "Training failed for region: $region"
        print_error "Check log file: $log_file"
        return 1
    fi
}

# Function to run dry run for all regions
dry_run_all() {
    print_status "Running dry run for all regions..."
    
    for region in "${REGIONS[@]}"; do
        print_status "Dry run for region: $region"
        python train_chunkformer.py \
            --config "$CONFIG_FILE" \
            --region "$region" \
            --dry-run \
            --log-level "$LOG_LEVEL"
        
        if [ $? -eq 0 ]; then
            print_success "Dry run completed for region: $region"
        else
            print_error "Dry run failed for region: $region"
            return 1
        fi
    done
}

# Function to show usage
show_usage() {
    cat << EOF
Chunkformer Training Script for VIMD Dataset

Usage: $0 [OPTIONS]

Options:
    --help, -h          Show this help message
    --dry-run           Run data preparation only (no training)
    --region REGION     Train only specific region (All, Central, South, North)
    --config FILE       Use custom config file (default: configs/config_vimd.yaml)
    --device DEVICE     Use specific device (default: cuda)
    --log-level LEVEL   Set log level (DEBUG, INFO, WARNING, ERROR)
    --no-eval           Skip evaluation after training
    --continue-on-error Continue training other regions even if one fails

Examples:
    $0                                  # Train all regions
    $0 --region Central                 # Train only Central region
    $0 --dry-run                        # Run dry run for all regions
    $0 --config custom_config.yaml     # Use custom config
    $0 --log-level DEBUG                # Enable debug logging

EOF
}

# Parse command line arguments
DRY_RUN=false
SINGLE_REGION=""
NO_EVAL=false
CONTINUE_ON_ERROR=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --region)
            SINGLE_REGION="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --no-eval)
            NO_EVAL=true
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "Chunkformer Training for VIMD Dataset"
    print_status "====================================="
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    print_status "Configuration file: $CONFIG_FILE"
    print_status "Device: $DEVICE"
    print_status "Log level: $LOG_LEVEL"
    
    # Check environment
    print_status "Checking environment..."
    if ! check_environment; then
        print_error "Environment check failed"
        exit 1
    fi
    
    # Check CUDA if using GPU
    if [ "$DEVICE" = "cuda" ]; then
        if ! check_cuda; then
            print_warning "CUDA check failed, but continuing..."
        fi
    fi
    
    # Create logs directory
    mkdir -p logs
    
    # Handle dry run
    if [ "$DRY_RUN" = true ]; then
        if [ -n "$SINGLE_REGION" ]; then
            print_status "Running dry run for region: $SINGLE_REGION"
            python train_chunkformer.py \
                --config "$CONFIG_FILE" \
                --region "$SINGLE_REGION" \
                --dry-run \
                --log-level "$LOG_LEVEL"
        else
            dry_run_all
        fi
        print_success "Dry run completed!"
        exit 0
    fi
    
    # Handle single region training
    if [ -n "$SINGLE_REGION" ]; then
        # Validate region
        if [[ ! " ${REGIONS[@]} " =~ " ${SINGLE_REGION} " ]]; then
            print_error "Invalid region: $SINGLE_REGION"
            print_error "Valid regions: ${REGIONS[*]}"
            exit 1
        fi
        
        print_status "Training single region: $SINGLE_REGION"
        if train_region "$SINGLE_REGION"; then
            print_success "Training completed successfully for region: $SINGLE_REGION"
        else
            print_error "Training failed for region: $SINGLE_REGION"
            exit 1
        fi
        exit 0
    fi
    
    # Train all regions
    print_status "Training all regions: ${REGIONS[*]}"
    
    failed_regions=()
    successful_regions=()
    
    for region in "${REGIONS[@]}"; do
        print_status "Starting training for region: $region ($(date))"
        
        if train_region "$region"; then
            successful_regions+=("$region")
            print_success "Completed training for region: $region"
        else
            failed_regions+=("$region")
            print_error "Failed training for region: $region"
            
            if [ "$CONTINUE_ON_ERROR" = false ]; then
                print_error "Stopping due to training failure"
                break
            else
                print_warning "Continuing with next region..."
            fi
        fi
        
        print_status "Progress: Completed ${#successful_regions[@]}/${#REGIONS[@]} regions"
    done
    
    # Final summary
    print_status "Training Summary"
    print_status "==============="
    
    if [ ${#successful_regions[@]} -gt 0 ]; then
        print_success "Successful regions (${#successful_regions[@]}): ${successful_regions[*]}"
    fi
    
    if [ ${#failed_regions[@]} -gt 0 ]; then
        print_error "Failed regions (${#failed_regions[@]}): ${failed_regions[*]}"
        exit 1
    else
        print_success "All regions trained successfully!"
    fi
}

# Run main function
main "$@"