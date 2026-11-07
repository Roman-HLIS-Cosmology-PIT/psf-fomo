"""YAML helpers: NumPy values are written as plain Python values."""
import numpy as np

from psf_fomo.io import read_yaml, to_plain, write_yaml


def test_numpy_values_round_trip(tmp_path):
    data = {"abzp": np.float64(25.775), "n": np.int64(7), "ok": np.bool_(True),
            "name": np.str_("WFI11"), "arr": np.array([1.5, 2.5]), "pair": (1, np.float32(2.5)),
            "nested": {"ids": [np.int32(3), np.int64(4)], "deep": {"x": np.float32(0.5)}},
            "plain": [1, "a", None, 2.0]}
    write_yaml(tmp_path / "x.yaml", data)
    back = read_yaml(tmp_path / "x.yaml")
    assert back == {"abzp": 25.775, "n": 7, "ok": True, "name": "WFI11", "arr": [1.5, 2.5],
                    "pair": [1, 2.5], "nested": {"ids": [3, 4], "deep": {"x": 0.5}},
                    "plain": [1, "a", None, 2.0]}
    for v in (back["abzp"], back["n"], back["ok"]):
        assert type(v) in (float, int, bool)


def test_to_plain_types():
    out = to_plain({"a": np.float64(1.0), "b": (np.int16(2),)})
    assert type(out["a"]) is float and out["b"] == [2] and type(out["b"][0]) is int
