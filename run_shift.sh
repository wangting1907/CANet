#!/bin/bash
DEVICE="0"
CONFIG_FILE="config_spectral.yaml"
ENERGY_START=0
ENERGY_END=66

echo "Starting Spectral Alignment Workflow with CANet..."
echo "Configuration: $CONFIG_FILE"
echo "------------------------------------------------"

for ENERGY_INDEX in $(seq $ENERGY_START $ENERGY_END); do
  
  echo "[$(date +'%H:%M:%S')] Processing ENERGY_INDEX: $ENERGY_INDEX"
  CUDA_VISIBLE_DEVICES=$DEVICE python -u train_spectral_alignment.py \
    --config $CONFIG_FILE \
    --energy_index $ENERGY_INDEX
  if [ $? -ne 0 ]; then
    echo "Error occurred at ENERGY_INDEX $ENERGY_INDEX. Terminating workflow."
    exit 1
  fi

done

echo "------------------------------------------------"
echo "Success: All $ENERGY_END ENERGY_INDEX values processed!"
