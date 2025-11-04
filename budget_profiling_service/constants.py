"""
Constantes partagées pour le service de profilage budgétaire.

Ce module contient les constantes configurables utilisées dans le service.
"""
import os

# Analysis Configuration
DEFAULT_MONTHS_ANALYSIS = int(os.getenv("DEFAULT_MONTHS_ANALYSIS", "12"))
MAX_MONTHS_ANALYSIS = int(os.getenv("MAX_MONTHS_ANALYSIS", "24"))
MIN_MONTHS_FOR_PROFILING = int(os.getenv("MIN_MONTHS_FOR_PROFILING", "3"))

# Fixed Charge Detection
FIXED_CHARGE_MIN_OCCURRENCES = int(os.getenv("FIXED_CHARGE_MIN_OCCURRENCES", "3"))
FIXED_CHARGE_AMOUNT_VARIANCE = float(os.getenv("FIXED_CHARGE_AMOUNT_VARIANCE", "0.05"))  # 5%

# Outlier Detection
OUTLIER_STD_MULTIPLIER = float(os.getenv("OUTLIER_STD_MULTIPLIER", "2.0"))
MIN_SAMPLES_FOR_OUTLIER = int(os.getenv("MIN_SAMPLES_FOR_OUTLIER", "6"))

# Savings Goals
MIN_SAVINGS_RATE = float(os.getenv("MIN_SAVINGS_RATE", "0.05"))  # 5%
RECOMMENDED_SAVINGS_RATE = float(os.getenv("RECOMMENDED_SAVINGS_RATE", "0.20"))  # 20%
HIGH_SAVINGS_RATE = float(os.getenv("HIGH_SAVINGS_RATE", "0.30"))  # 30%

# Database Configuration
DB_QUERY_TIMEOUT = float(os.getenv("DB_QUERY_TIMEOUT", "30.0"))

# Retry Configuration
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_MIN_WAIT_SECONDS = float(os.getenv("RETRY_MIN_WAIT_SECONDS", "1"))
RETRY_MAX_WAIT_SECONDS = float(os.getenv("RETRY_MAX_WAIT_SECONDS", "5"))

# Segmentation Thresholds
LOW_INCOME_THRESHOLD = int(os.getenv("LOW_INCOME_THRESHOLD", "2000"))
MEDIUM_INCOME_THRESHOLD = int(os.getenv("MEDIUM_INCOME_THRESHOLD", "4000"))
HIGH_INCOME_THRESHOLD = int(os.getenv("HIGH_INCOME_THRESHOLD", "7000"))
