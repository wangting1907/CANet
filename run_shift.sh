#!/bin/bash
DEVICE="0"

# Define the range of energy indices
ENERGY_START=0
ENERGY_END=66

# Paths and directories
TXT_DIR="./Data/XANES_00089/AC3_C4p6_3DXANES/AC3_C4p6_3DXANES_TOMO-XANES.txt"
HIGHEST_PROJ="./Data/XANES_00089/highest_energy/alignment_8346.mat"
RESULT_DIR="./Data/XANES_00089/alignment_result/"


# Iterate through each energy index in the range [1, 65]
for ENERGY_INDEX in $(seq $ENERGY_START $ENERGY_END); do
  LOG_DIR="./runs/scaling=$SCALING,/energy=$ENERGY_INDEX"
  
  # Print the current energy index for debugging/logging
  echo "Running for ENERGY_INDEX=$ENERGY_INDEX"

  # Run the Python script with the current ENERGY_INDEX
  CUDA_VISIBLE_DEVICES=$DEVICE python -u diff_energy_alignment_shift_scaling.py \
    --txt_dir $TXT_DIR \
    --energy_index $ENERGY_INDEX \
    --highest_proj $HIGHEST_PROJ \
    --cuda \
    --result_dir $RESULT_DIR \
    --log_dir $LOG_DIR \
    --scaling
done
# Notify when all runs are complete
echo "All ENERGY_INDEX values processed!"

