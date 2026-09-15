# -*- coding: utf-8 -*-
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cwru.config import CLASSES  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def records():
    from cwru.data.audit import audit_all
    return audit_all()


@pytest.fixture(scope="session")
def splits(records):
    """直接使用代码中写死的 A/B/C 三套固定划分。"""
    from cwru.data.split import build_fixed_splits
    return build_fixed_splits(records)


@pytest.fixture(scope="session")
def manifest_101():
    """A/no_overlap/101DE 实验清单（新键名）。"""
    from cwru.config import MANIFESTS_DIR
    path = os.path.join(MANIFESTS_DIR, "experiment_A_101DE_no_overlap.json")
    if not os.path.exists(path):
        pytest.skip("缺少实验清单，请先运行 prepare")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


CLASS_NAMES = CLASSES
