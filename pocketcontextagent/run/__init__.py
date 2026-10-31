"""Run scripts for mini-SWE-agent."""

import sys

import benchmarks

sys.modules[f"{__name__}.benchmarks"] = benchmarks
