# -*- coding: utf-8 -*-
"""数据层测试：文件识别、特殊映射、降采样、划分、窗口、归一化与类别权重。"""
import numpy as np
import pytest

from cwru.config import CLASSES, SPLIT_NAMES
from cwru.data.audit import identify_variables, parse_filename
from cwru.data.signals import load_channel, make_windows
from cwru.data.split import build_split


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
        # 28 mil 只有驱动端 8 个文件
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
            # resample_poly(1,4) 输出长度 = ceil(n/4)
            assert len(sig) == math.ceil(r.n_de / 4), r.filename

    def test_windows_shape(self):
        sig = np.arange(2500, dtype=np.float64)
        w = make_windows(sig, 1024, 1024)
        assert w.shape == (2, 1024)
        assert w.dtype == np.float32
        assert np.allclose(w[1], np.arange(1024, 2048))


class TestSplit:
    def test_counts_and_no_overlap(self, splits, records):
        s101 = splits["101"]
        assert len(s101) == 101
        assert len(set(s101.keys())) == 101
        counts = {k: sum(1 for v in s101.values() if v == k) for k in SPLIT_NAMES}
        assert counts == {"train": 71, "val": 15, "test": 15}
        records_by_name = {r.filename: r for r in records}
        normal_counts = {}
        for name, split in s101.items():
            if records_by_name[name].fault == "Normal":
                normal_counts[split] = normal_counts.get(split, 0) + 1
        assert normal_counts == {"train": 2, "val": 1, "test": 1}

    def test_109_consistent_common(self, splits):
        s101, s109 = splits["101"], splits["109"]
        assert len(s109) == 109
        for f, v in s101.items():
            assert s109[f] == v, f
        extra = set(s109) - set(s101)
        assert len(extra) == 8  # B028 + IR028
        counts = {k: sum(1 for f in extra if s109[f] == k) for k in SPLIT_NAMES}
        assert counts == {"train": 4, "val": 2, "test": 2}

    def test_deterministic(self, records):
        a = build_split(records)
        b = build_split(records)
        assert a == b


class TestExperimentArrays:
    def test_shapes_and_labels(self, manifest_101):
        assert manifest_101["window_len"] == 1024
        assert manifest_101["stride"] == 1024
        assert manifest_101["sr"] == 12000
        n_train = sum(m["n_windows"] for m in manifest_101["files"] if m["split"] == "train")
        counts = manifest_101["train_class_counts"]
        assert sum(counts) == n_train

    def test_class_weights_formula(self, manifest_101):
        import math
        counts = manifest_101["train_class_counts"]
        w = [1.0 / math.sqrt(c) for c in counts]
        mean = sum(w) / len(w)
        w = [x / mean for x in w]
        assert np.allclose(manifest_101["class_weights"], w, atol=1e-5)

    def test_norm_from_train_only(self, manifest_101):
        cache = np.load(_cache_path("101DE.npz"))
        X, file_idx = cache["X"], cache["file_idx"]
        train_files = {m["index"] for m in manifest_101["files"] if m["split"] == "train"}
        sel_train = np.isin(file_idx, list(train_files))
        sel_other = ~sel_train
        # 归一化基于训练集：训练集通道均值≈0、方差≈1
        mean_tr = float(X[sel_train][:, 0, :].mean())
        std_tr = float(X[sel_train][:, 0, :].std())
        assert abs(mean_tr) < 1e-3
        assert abs(std_tr - 1.0) < 1e-3
        # 验证/测试集未参与归一化参数计算：其统计量与 0/1 有偏差
        mean_va = float(X[sel_other][:, 0, :].mean())
        std_va = float(X[sel_other][:, 0, :].std())
        assert abs(mean_va) > 1e-6 or abs(std_va - 1.0) > 1e-6
        # manifest 记录的参数为正的标准差
        assert manifest_101["norm"][0]["std"] > 0

    def test_normal_regression_mask(self, manifest_101):
        cache = np.load(_cache_path("101DE.npz"))
        file_idx, y_reg, mask = cache["file_idx"], cache["y_reg"], cache["reg_mask"]
        normal_idx = {m["index"] for m in manifest_101["files"] if m["fault"] == "Normal"}
        sel = np.isin(file_idx, list(normal_idx))
        assert sel.any()
        assert not mask[sel].any()
        assert np.allclose(y_reg[sel], 0.0)
        # 故障窗口都有回归标签
        assert mask[~sel].all()


def _cache_path(name):
    import os
    from cwru.config import CACHE_DIR
    return os.path.join(CACHE_DIR, name)
