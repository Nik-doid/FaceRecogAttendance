"""Utility helpers shared across the service (image ops, misc).

Kept intentionally small: heavy logic lives in ``app.ai``, ``app.storage`` and the
worker modules. Put cross-cutting, framework-agnostic helpers here.
"""

from __future__ import annotations
