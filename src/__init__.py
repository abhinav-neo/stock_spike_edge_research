"""Stock spike edge research package."""

# NumPy 2.x removed the historical ``np.math`` alias. The V4 walk-forward
# module uses only ``erfc`` from the standard library through that alias, so
# restore it explicitly for compatibility across NumPy versions.
import math

import numpy as np

if not hasattr(np, "math"):
    np.math = math  # type: ignore[attr-defined]
