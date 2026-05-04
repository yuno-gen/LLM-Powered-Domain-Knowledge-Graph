"""Shared pytest fixtures.

Adds the package root to ``sys.path`` so the tests can be invoked as
either ``pytest vertexiq/tests`` (from project root) or ``pytest tests``
(from inside the package directory) without extra config.
"""

from __future__ import annotations

import os
import sys

# Insert the parent directory of the ``vertexiq`` package onto sys.path
# so ``import vertexiq`` resolves regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)
for path in (_PROJECT_ROOT, _PACKAGE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
