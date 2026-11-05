import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Org-roam directory
# Reads from ORG_ROAM_DIR in .env, defaults to ~/org/org-roam
ORG_ROAM_DIR = (
    Path(os.getenv("ORG_ROAM_DIR", str(Path.home() / "org" / "org-roam")))
    .expanduser()
    .resolve()
)

# Chroma vector database directory
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
