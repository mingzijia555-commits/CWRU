# -*- coding: utf-8 -*-
"""数据层测试：文件识别、特殊映射、降采样、三套划分、窗口（含 50% 重叠）、归一化与类别权重。"""
import numpy as np
import pytest

from cwru.config import (GROUP4_TARGETS, SPLIT_NAMES, SPLIT_SETS,
                         SPLIT_TARGETS_101, SPLIT_TARGETS_109, WINDOW_PLANS)
from cwru.data.audit import identify_variables, parse_filename
from cwru.data.signals import load_channel, make_windows
from cwru.data.split import validate_splits


class TestFileIdentification:
    def test_total_109(self, records):
        assert len(records) == 109

    def test_set_sizes(self, records):
        assert sum(1 for r in records if r.in_set_101) == 101
        assert all(r.in_set_109 for r in records)

    def test_fault_composition(self, records):
        by_fault = {}
        for r in records:
            by_fault[r.fault] = by_fault.get(r.fault, 0) + 1
        assert by_fault == {"Normal": 4, "IR": 28, "OR": 49, "B": 28}
        mil28 = [r for r in records if r.diameter_mil >= 28.0]
        assert len(mil28) == 8
        assert all(r.end == "Drive" for r in mil28)

    def test_parse_filename(self):
        info = parse_filename("12k_Drive_End_OR007@3_2_146.mat")
        assert info == {"end": "Drive", "fault": "OR", "diameter_mil": 7.0,
                        "or_clock": 3, "load": 2, "file_id": 146}
        info = parse_filename("normal_3_100.mat")
        assert info["end"] == "Normal" and info["load"] == 3 and info["file_id"] == 100


class TestSpecialMappings:
    def test_normal_2_99_uses_x099(self):
        import scipy.io as sio
        from cwru.config import DATA_DIR
        mat = sio.loadmat(f"{DATA_DIR}/normal_2_99.mat")
        vs = identify_variables("normal_2_99.mat", mat)
        assert vs == {"DE": "X099_DE_time", "FE": "X099_FE_time"}

    def test_fan_ir014_1_276_uses_x275(self):
        import scipy.io as sio
        from cwru.config import DATA_DIR
        mat = sio.loadmat(f"{DATA_DIR}/12k_Fan_End_IR014_1_276.mat")
        vs = identify_variables("12k_Fan_End_IR014_1_276.mat", mat)
        assert vs == {"DE": "X275_DE_time", "FE": "X275_FE_time"}

    def test_28mil_unique_de(self, records):
        for r in records:
            if r.diameter_mil >= 28.0:
                assert r.var_de.endswith("_DE_time")
                assert r.var_fe == ""
                assert r.n_de > 100_000


class TestResampling:
    def test_normal_48k_to_12k(self, records):
        import math
        normal = [r for r in records if r.end == "Normal"]
        for r in normal:
            sig = load_channel(r, "DE")
            assert len(sig) == math.ceil(r.n_de / 4), r.filename


class TestWindows:
    def test_no_overlap_shape(self):
        sig = np.arange(2500, dtype=np.float64)
        w = make_windows(sig, 1024, 1024)
        assert w.shape == (2, 1024)
        assert w.dtype == np.float32
        assert np.allclose(w[1], np.arange(1024, 2048))

    def test_overlap50_shape_and_step(self):
        sig = np.arange(4096, dtype=np.float64)
        w = make_windows(sig, 1024, 512)
        # (4096-1024)//512 + 1 = 7 个窗口，相邻窗口起点相差 512
        assert w.shape == (7, 1024)
        assert np.allclose(w[0][:512], sig[:512])
        assert np.allclose(w[1][:512], sig[512:1024])          # 步长 512
        assert np.allclose(w[0][512:], sig[512:1024])           # 与下一窗重叠 512

    def test_overlap_window_count_nearly_double(self):
        sig = np.arange(12000, dtype=np.float64)
        assert len(make_windows(sig, 1024, 1024)) == 11
        assert len(make_windows(sig, 1024, 512)) == 22

    def test_rejects_illegal_stride(self):
        with pytest.raises(ValueError):
            make_windows(np.arange(4096.0), 1024, 0)
        with pytest.raises(ValueError):
            make_windows(np.arange(4096.0), 1024, 2048)


class TestSplitSets:
    def test_three_sets_counts(self, splits, records):
        for name in SPLIT_SETS:
            s101, s109 = splits[name]["101"], splits[name]["109"]
            assert len(s101) == 101, name
            assert len(s109) == 109, name
            c101 = {k: sum(1 for v in s101.values() if v == k) for k in SPLIT_NAMES}
            c109 = {k: sum(1 for v in s109.values() if v == k) for k in SPLIT_NAMES}
            assert (c101["train"], c101["val"], c101["test"]) == SPLIT_TARGETS_101
            assert (c109["train"], c109["val"], c109["test"]) == SPLIT_TARGETS_109

    def test_common_consistent_and_28mil_only_109(self, splits, records):
        by_name = {r.filename: r for r in records}
        for name in SPLIT_SETS:
            s101, s109 = splits[name]["101"], splits[name]["109"]
            for f, v in s101.items():
                assert s109[f] == v, f
            for f in set(s109) - set(s101):
                assert by_name[f].diameter_mil >= 28.0, f
            assert len(set(s109) - set(s101)) == 8

    def test_set_balance(self, splits, records):
        by_name = {r.filename: r for r in records}
        for name in SPLIT_SETS:
            s101 = splits[name]["101"]
            for s in SPLIT_NAMES:
                members = [by_name[f] for f, v in s101.items() if v == s]
                ends = {r.end for r in members}
                faults = {r.fault for r in members}
                assert {"Drive", "Fan"} <= ends, (name, s, ends)
                assert {"Normal", "IR", "OR", "B"} <= faults, (name, s, faults)
            train_conditions = {
                (by_name[f].end, by_name[f].fault, int(by_name[f].diameter_mil), by_name[f].or_clock)
                for f, part in s101.items() if part == "train"
            }
            for part in ("val", "test"):
                members = [by_name[f] for f, value in s101.items() if value == part]
                assert sum(r.fault == "Normal" for r in members) == 1
                assert sum(r.fault == "IR" for r in members) >= 3
                assert sum(r.fault == "B" for r in members) >= 3
                assert sum(r.fault == "OR" for r in members) >= 5
                assert min(sum(r.end == end for r in members) for end in ("Drive", "Fan")) >= 5
                assert min(sum(r.load == load for r in members) for load in range(4)) >= 2
                assert min(sum(r.fault != "Normal" and int(r.diameter_mil) == d for r in members)
                           for d in (7, 14, 21)) >= 3
                assert min(sum(r.fault == "OR" and r.or_clock == clock for r in members)
                           for clock in (3, 6, 12)) >= 1
                assert all((r.end, r.fault, int(r.diameter_mil), r.or_clock) in train_conditions
                           for r in members)

    def test_validation_passes(self, splits, records):
        report = validate_splits(splits, records)
        assert report["ok"], report["issues"]

    def test_test_sets_low_repeat(self, splits):
        for set_id in ("101", "109"):
            test_sets = [{f for f, s in splits[n][set_id].items() if s == "test"}
                         for n in SPLIT_SETS]
            for i in range(3):
                for j in range(i + 1, 3):
                    assert not test_sets[i] & test_sets[j]

    def test_normal_and_28mil_split(self, splits, records):
        by_name = {r.filename: r for r in records}
        for name in SPLIT_SETS:
            s101 = splits[name]["101"]
            normal = {s: 0 for s in SPLIT_NAMES}
            for f, v in s101.items():
                if by_name[f].fault == "Normal":
                    normal[v] += 1
            assert tuple(normal[s] for s in SPLIT_NAMES) == GROUP4_TARGETS


class TestExperimentArrays:
    def test_shapes_and_labels(self, manifest_101):
        assert manifest_101["window_len"] == 1024
        assert manifest_101["stride"] == 1024
        assert manifest_101["sr"] == 12000
        assert manifest_101["split_set"] == "A"
        n_train = sum(m["n_windows"] for m in manifest_101["files"] if m["split"] == "train")
        assert sum(manifest_101["train_class_counts"]) == n_train

    def test_class_weights_formula(self, manifest_101):
        counts = manifest_101["train_class_counts"]
        w = [1.0 / max(c, 1) ** 0.5 for c in counts]
        mean = sum(w) / len(w)
        w = [x / mean for x in w]
        assert np.allclose(manifest_101["class_weights"], w, atol=1e-5)

    def test_overlap_manifest_stride(self):
        import json
        import os
        from cwru.config import MANIFESTS_DIR
        path = os.path.join(MANIFESTS_DIR, "experiment_A_101DE_overlap50.json")
        if not os.path.exists(path):
            pytest.skip("缺少 50% 重叠实验清单，请先运行 prepare")
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["stride"] == WINDOW_PLANS["overlap50"]["stride"] == 512

    def test_norm_from_train_only(self, manifest_101):
        import os
        from cwru.config import CACHE_DIR
        path = os.path.join(CACHE_DIR, "A_101DE_no_overlap.npz")
        if not os.path.exists(path):
            pytest.skip("缺少窗口缓存，请先运行 prepare")
        cache = np.load(path)
        X, file_idx = cache["X"], cache["file_idx"]
        train_files = {m["index"] for m in manifest_101["files"] if m["split"] == "train"}
        sel_train = np.isin(file_idx, list(train_files))
        mean_tr = float(X[sel_train][:, 0, :].mean())
        std_tr = float(X[sel_train][:, 0, :].std())
        assert abs(mean_tr) < 1e-3
        assert abs(std_tr - 1.0) < 1e-3
        assert manifest_101["norm"][0]["std"] > 0

    def test_normal_regression_mask(self, manifest_101):
        import os
        from cwru.config import CACHE_DIR
        path = os.path.join(CACHE_DIR, "A_101DE_no_overlap.npz")
        if not os.path.exists(path):
            pytest.skip("缺少窗口缓存，请先运行 prepare")
        cache = np.load(path)
        file_idx, y_reg, mask = cache["file_idx"], cache["y_reg"], cache["reg_mask"]
        normal_idx = {m["index"] for m in manifest_101["files"] if m["fault"] == "Normal"}
        sel = np.isin(file_idx, list(normal_idx))
        assert sel.any()
        assert not mask[sel].any()
        assert np.allclose(y_reg[sel], 0.0)
        assert mask[~sel].all()
