"""
Execute Jupyter Notebook programmatically to verify all cells and embed real outputs.
"""

import sys
import io
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

notebook_path = "Tesla_Autopilot_Object_Detection.ipynb"
print(f"[*] Executing notebook: {notebook_path}...")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

try:
    ep.preprocess(nb, {"metadata": {"path": "."}})
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[OK] Notebook executed successfully with all outputs populated!")
except Exception as e:
    print(f"[ERROR] Notebook execution failed: {e}")
    sys.exit(1)
