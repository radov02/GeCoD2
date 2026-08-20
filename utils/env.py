import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Load the project .env into os.environ, log in to HF if a token is set."""
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=False)
        except ImportError:
            # no python-dotenv installed, parse the key=value lines by hand
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass
