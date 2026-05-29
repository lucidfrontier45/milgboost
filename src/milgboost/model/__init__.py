import logging

from .base import BaseMILModel

logger = logging.getLogger(__name__)

try:
    from .lightgbm import LightGBMMILModel
except ImportError:
    logger.info("lightgbm is not installed; LightGBMMILModel unavailable")

try:
    from .xgboost import XGBoostMILModel
except ImportError:
    logger.info("xgboost is not installed; XGBoostMILModel unavailable")
