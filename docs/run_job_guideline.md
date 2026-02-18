# Guideline for job running

## Environment
- /home/aiops/zhaojx/venv/aigc

## GPU allocation
- If running with less than 8 GPU, please avoid using GPU0

## Wandb
- Always enable wandb log in, current venv should already log into the wandb, otherwise please contact human supervisor
- The key is stored in .netrc_wandb 

## Monitor
- Monitor for at least 10 steps to make sure the job does not break immediately