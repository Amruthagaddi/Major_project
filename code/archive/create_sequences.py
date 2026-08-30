"""
Legacy: moved to code/archive/. Use code/build_dataset.py instead.
"""

from pathlib import Path
import shutil

src = Path(__file__).with_name("../create_sequences.py").resolve()
if src.exists():
    shutil.move(str(src), str(Path(__file__).parent / "create_sequences_legacy.py"))
