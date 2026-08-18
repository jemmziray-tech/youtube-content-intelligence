import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.cleaner import DataCleaner
from src.feature_engineering import FeatureEngineer

def test_duration_parsing():
    cleaner = DataCleaner()
    assert cleaner._parse_duration("PT15M33S") == 933.0
    assert cleaner._parse_duration("PT1H") == 3600.0
    assert cleaner._parse_duration("P1DT2H") == 93600.0
    assert np.isnan(cleaner._parse_duration(np.nan))
    assert np.isnan(cleaner._parse_duration("invalid_string"))

def test_safe_divide():
    engineer = FeatureEngineer()
    a = np.array([10, 20, 30])
    b = np.array([2, 0, 5])
    result = engineer._safe_divide(a, b)
    assert np.array_equal(result, np.array([5.0, 0.0, 6.0]))
