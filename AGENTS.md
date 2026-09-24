# Working on CL

- Read EXPERIMENTS.md section 4 before preparing a new server or rerunning experiments.
- Code-only repository: models/, data/, results/, legacy asset paths and credentials are ignored. Never force-add them.
- Run python3 scripts/prepare_workspace.py after cloning. Fetch public prerequisites with scripts/fetch_assets.py.
- Mainline Python is 3.12. Pin dependencies using config/environments/; historical Python 3.13 tools use separate environments.
- Use TTCL_WORKSPACE and TTCL_PYTHON for mainline path overrides. Do not assume /home/fengboyu/cl exists on a new server.
- Preserve official scoring, train/test separation, declared budgets, failure records and frozen-run hashes.
- New trajectory data requires new reviewed annotation targets and input-content bindings. Never reuse labels based only on an old history ID.
- Historical checkpoints and datasets are deliberately absent from Git. Follow the dependency order instead of inventing missing results.
- Preserve vendored LICENSE files and config/upstreams.json. Upstream nested Git repositories have been removed intentionally.
- Check staged files with python3 scripts/check_repository.py before any push.
