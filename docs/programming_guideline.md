
# Guideline
- Do not introduce extra variable if not necessary
- Whenever you plan to implement a function, first search carefully the whole codebase to check if it has been implemented elsewhere
- Error handling is a good habit, but if it becomes too messy, we should simplify it, e.g. too many `try`, too many if else condition
for example, the following is useless, as the execution will return similar message:
```
try:
    from torchmetrics.image.fid import FrechetInceptionDistance  # noqa: F401
except ImportError:
    print("torchmetrics is not installed; disabling FID logging.")
```