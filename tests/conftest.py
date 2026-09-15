# -*- coding: utf-8 -*-
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cwru.config import CLASSES  # noqa: E402


@pytest.fixture(scope="session")
def records():
    from cwru.data.audit import audit_all
    return audit_all()


@pytest.fixture(scope="session")
def splits(records):
    from cwru.data.split import build_split
    return build_split(records)


@pytest.fixture(scope="session")
def manifest_101():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "artifacts", "manifests", "experiment_101DE.json"), encoding="utf-8") as f:
        return json_load(f)


def json_load(f):
    import json
    return json.load(f)


CLASS_NAMES = CLASSES
