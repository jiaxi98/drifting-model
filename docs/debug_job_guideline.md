# Job Debugging Guideline

This file is a practical checklist for debugging broken jobs in this repo.

## Checklist
- Locate the code block where the problem occurs, print in stdout for complete info.
- Explain the errors in details, e.g. if it is a memory issue (usually the case)
  - State the total memory of the pod
  - State which big tensor cost the memory
  - Explain why this tensor is so big via explicit calculation
  - Explain how to fix