docker run -d --gpus '"device=0,1,2,3"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/results:/results \
  -e DATASET_DIR=/data/training-data \
  -e TARGETS_DIR=/data/training-data-targets \
  -e OUTPUT_DIR=/models/wavlm-multiaxis \
  -e NPROC=4 \
  sst bash /workspace/experiment/run_train_multiaxis.sh

docker logs 0e627e4ac0bd183efc05dbfebb0278d8363a71a98519dc47a9ea22bcf15f7610

docker run -d --gpus '"device=4,5,6,7"' --ipc=host \
  -v /data:/data -v /models:/models \
  -e DATASET_DIR=/data/training-data \
  -e TARGETS_DIR=/data/training-data-targets \
  -e OUTPUT_DIR=/models/wavlm-semantic \
  -e NPROC=4 \
  sst bash /workspace/experiment/run_train_semantic.sh

220bab284a3d140fec0bd8daad9a94cbb5783ee474618e53a0b374395f862af6