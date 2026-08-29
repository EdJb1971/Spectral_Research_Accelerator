"""Compatibility import for the executable extension now used by TG12.2d.

The domain remains declared outside ``src``; this module keeps the original TG8.1 import path
while exercising the production extension rather than a test-only declaration.
"""

from extensions.argo_float import ARGO, ARGO_PHRASES, DOMAIN_NAME, ONBOARDED

__all__ = ["ARGO", "ARGO_PHRASES", "DOMAIN_NAME", "ONBOARDED"]
