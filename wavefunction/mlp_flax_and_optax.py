# ─────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────────────────────
import time
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import flax.linen as nn
from flax.training import train_state
