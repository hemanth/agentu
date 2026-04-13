---
description: Bump version, commit, push, and publish to PyPI
---

1. Update `version` in `pyproject.toml`
// turbo
2. Run `cd /Users/hemanth/labs/agentu && git add -A`
3. Commit with message: `chore: bump version to X.Y.Z`
4. Push to remote: `git push`
// turbo
5. Clean old dist: `rm -rf dist/`
// turbo
6. Build: `python3 -m build`
7. Publish to PyPI: `twine upload dist/*`
// turbo
8. Verify: `pip3 index versions agentu | head -1`
