"""
OptiQuant-GPU: High-Throughput Regime-Adaptive Quantitative Portfolio Engine
Main Streamlit Application Entrypoint.
"""

import sys
from pathlib import Path

# Add root directory to python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from optiquant.ui.dashboard import main

if __name__ == "__main__":
    main()
