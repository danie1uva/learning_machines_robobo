#!/usr/bin/env zsh

set -xe

export WANDB_API_KEY=""

# Build the Docker image
docker build --platform linux/amd64 --tag learning_machines .

# Run the container, passing the WandB API key as an environment variable
docker run -t --rm --platform linux/amd64 \
  -p 45100:45100 \
  -p 45101:45101 \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -v "$(pwd)/results:/root/results" \
  learning_machines "$@"

# Adjust file ownership
# sudo chown "$USER":"$USER" ./results -R

