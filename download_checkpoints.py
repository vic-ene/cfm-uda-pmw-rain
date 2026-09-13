# Requires: pip install -U huggingface_hub

import os
import shutil
import zipfile
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, EntryNotFoundError

# ---- Settings ----
REPO_ID = "vicene/cfm-uda-pmw-rain"
REPO_TYPE = "dataset"
FILENAME = "huggingface_downloads.zip"
ROOT = os.path.abspath(".")                           # where the script is run from
EXTRACT_DIR = os.path.join(ROOT, "huggingface_downloads")
DELETE_ZIP_AFTER = True                             # True = free disk space after extracting

# Public repo: no token needed.
# Private repo: uses the token from `hf auth login` (or HF_TOKEN if set).
TOKEN = os.environ.get("HF_TOKEN") or None

if os.path.isdir(EXTRACT_DIR):
    print("Already exists, skipping:", EXTRACT_DIR)
else:
    # 1. Download the zip into ROOT (resumes if interrupted, skips if already there)
    print("Downloading...")
    try:
        zip_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type=REPO_TYPE,
            local_dir=ROOT,
            token=TOKEN,
        )
    except GatedRepoError:
        raise SystemExit("Repo is gated: request access on the repo page, then run `hf auth login`.")
    except RepositoryNotFoundError:
        raise SystemExit("Repo not found: the name is wrong, or it's private and you're not "
                         "logged in with an account that has access (run `hf auth login`).")
    except EntryNotFoundError:
        raise SystemExit(f"'{FILENAME}' is not in the repo. Check the file name on the repo page.")
    print("Downloaded:", zip_path)

    # 2. Extract into a temp folder, then rename only once complete
    tmp_dir = EXTRACT_DIR + ".part"
    shutil.rmtree(tmp_dir, ignore_errors=True)   # clean up any interrupted attempt
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)
    os.replace(tmp_dir, EXTRACT_DIR)
    print("Extracted to:", EXTRACT_DIR)

    # 3. Optionally delete the zip
    if DELETE_ZIP_AFTER:
        os.remove(zip_path)
        print("Deleted zip to save space")