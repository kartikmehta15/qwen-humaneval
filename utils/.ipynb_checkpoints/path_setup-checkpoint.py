import sys
from pathlib import Path

def add_repo_root_to_sys_path():
    """
    Add the repo root (parent of `src` and `notebooks`) to sys.path
    so notebooks can import project modules.
    """
    if "__file__" in globals():
        repo_root = Path(__file__).resolve().parents[1]
    else:
        # Jupyter notebooks: use working dir
        repo_root = Path.cwd().parent

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print("Repo root:", repo_root)
    return repo_root
