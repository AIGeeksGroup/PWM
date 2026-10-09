import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "111c3fab7fb020d1e261a68be6ec78a3fecc8d5b"
UPSTREAM = "https://github.com/stdstu12/YUME.git"
SAMPLES = ("sample_5b.py", "sample_5b_natsom.py", "sample_5b_sft.py", "sample_5b_som.py", "sample_5b_nft.py")


def setup(destination, source_repo=None):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; use a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".yume-", dir=destination.parent) as tmp:
        checkout = Path(tmp) / "YUME"
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        source = str(Path(source_repo).resolve()) if source_repo else UPSTREAM
        subprocess.run(["git", "-C", str(checkout), "fetch", "--depth", "1", source, REVISION], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"], check=True)
        for name in SAMPLES:
            shutil.copy2(ROOT / "sample" / name, checkout / "fastvideo/sample" / name)
        (checkout / ".pwm-overlay").write_text(REVISION + "\n")
        os.rename(checkout, destination)
    print(f"Yume ready at {destination}")
    print("Install PWM requirements; the launchers add this checkout to PYTHONPATH.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=ROOT / "third_party/YUME")
    p.add_argument("--source-repo", type=Path, help="Optional local git mirror containing the pinned commit")
    a = p.parse_args()
    setup(a.destination, a.source_repo)
