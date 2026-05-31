#!/bin/bash


VIDEOS_PATH="/root/data1/yizeli/VideoX-Fun/examples/wan2.1/vbench_videos/gt_sft/"


FAILED_DIMS=()

for dim in subject_consistency background_consistency temporal_flickering motion_smoothness dynamic_degree aesthetic_quality imaging_quality object_class multiple_objects human_action color spatial_relationship scene appearance_style temporal_style overall_consistency; do
  echo "=== Evaluating: $dim ==="
  if ! vbench evaluate --videos_path $VIDEOS_PATH --dimension $dim; then
    echo "!!! FAILED: $dim !!!"
    FAILED_DIMS+=("$dim")
  fi
done

echo ""
echo "================================"
if [ ${#FAILED_DIMS[@]} -eq 0 ]; then
  echo "All dimensions evaluated successfully!"
else
  echo "Failed dimensions (${#FAILED_DIMS[@]}):"
  for d in "${FAILED_DIMS[@]}"; do
    echo "  - $d"
  done
fi