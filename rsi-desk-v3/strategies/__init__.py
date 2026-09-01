"""Import every strategy module here so it registers itself."""

from strategies.base import REGISTRY, Strategy, get, names, register  # noqa: F401
from strategies import rsi_reversion  # noqa: F401,E402
from strategies import trend  # noqa: F401,E402

# Add your own here:
# from strategies import my_rule  # noqa: F401
