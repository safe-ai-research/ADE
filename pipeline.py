# Compatibility import for existing scripts

"""
The refactored processing order is implemented in ade/pipeline.py
Keeping this small anchor means existing callers using "from pipeline import run_pipeline" do not need to change immediately
"""

from ade.pipeline import run_pipeline

__all__ = ["run_pipeline"]
