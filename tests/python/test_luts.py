"""(a) LUT tables regenerate identically to the committed headers."""
from __future__ import annotations

import math

import numpy as np

from llaccel import luts


def test_cpp_header_matches_committed(tmp_path, repo_root):
    out = tmp_path / "luts.h"
    luts.write_cpp_header(out)
    committed = (repo_root / "include" / "llaccel" / "luts.h").read_text()
    assert out.read_text() == committed


def test_sv_header_matches_committed_if_present(tmp_path, repo_root):
    sv = repo_root / "rtl" / "llaccel_luts.svh"
    if not sv.exists():
        return
    out = tmp_path / "luts.svh"
    luts.write_sv_header(out)
    assert out.read_text() == sv.read_text()


def test_table_definitions():
    sig, ei, ef = luts.sigmoid_lut(), luts.exp_int_lut(), luts.exp_frac_lut()
    assert sig.shape == (257,) and ei.shape == (16,) and ef.shape == (256,)
    assert sig.dtype == np.uint16 and ei.dtype == np.uint16 and ef.dtype == np.uint16
    assert sig[128] == 32768  # sigmoid(0) * 65536
    assert sig[256] == 65514  # NUMERICS.md: sigma(8) * 65536
    assert np.all(np.diff(sig.astype(np.int64)) >= 0)  # monotone (interpolation diff >= 0)
    assert ei[0] == 65535 and ef[0] == 65535
    for i in range(16):
        assert ei[i] == round(math.exp(-i) * 65535)
    for f in range(256):
        assert ef[f] == round(math.exp(-f / 256) * 65535)
