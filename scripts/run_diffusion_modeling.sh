#!/bin/bash
docker run -it --rm \
  -v "$(pwd):/workspaces/scenic-ins" \
  --user $(id -u):$(id -g) \
  scenic-ins \
  python -m src.run_diffusion_modeling