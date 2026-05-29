import logging

from .base import BaseMILModel

logger = logging.getLogger(__name__)

try:
    from .lightgbm import LightGBMMILModel
except ImportError:
    logger.info("lightgbm is not installed; LightGBMMILModel unavailable")
