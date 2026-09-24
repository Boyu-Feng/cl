"""Portable checkout/interpreter paths shared by new-server entry points.

TTCL_WORKSPACE overrides checkout discovery. TTCL_PYTHON selects an executable
(path relative to the checkout, absolute path, or name on PATH).
"""
from ttcl.experience_evolution.core import python_executable, workspace_root

WORKSPACE = workspace_root()
PYTHON = python_executable(WORKSPACE)
