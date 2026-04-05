docker run -d --name sst-resemblyzer \
  --gpus '"device=4,5,6,7"' \
  -v /home/joregan/spoken-sentence-transformers/workspace:/workspace \
  -e DATA_DIR=/workspace/training-data \
  -e TARGETS_DIR=/workspace/resemblyzer-targets \
  -e OUTPUT_DIR=/workspace/wavlm-resemblyzer \
  -e OSR_AUDIO_DIR=/workspace/osr-audio \
  -e OSR_SEGMENTS_DIR=/workspace/osr-segments \
  -e OSR_DATASET_DIR=/workspace/osr-dataset \
  sst \
  bash experiment/bootstrap_resemblyzer.sh