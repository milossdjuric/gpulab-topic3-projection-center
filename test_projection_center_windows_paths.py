"""Regression test: CppBackend's binary-path helpers must resolve to
forward_search.exe/forward_search_resample.exe on Windows, not the
extension-less Linux names -- meson builds .exe on Windows, and
Path(...).exists() checking the Linux name would always be False there,
making --backend cpp report "binary not found" even after a successful
build."""
import sys

sys.path.insert(0, "projection-center/src")
from projection_center_searching.backends import _cpp_binary_path, _cpp_resample_binary_path


def test_windows_paths_have_exe_suffix(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert _cpp_binary_path().name == "forward_search.exe"
    assert _cpp_resample_binary_path().name == "forward_search_resample.exe"
    print("test_windows_paths_have_exe_suffix: PASS")


def test_linux_paths_have_no_suffix(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert _cpp_binary_path().name == "forward_search"
    assert _cpp_resample_binary_path().name == "forward_search_resample"
    print("test_linux_paths_have_no_suffix: PASS")


if __name__ == "__main__":
    class _FakeMonkeypatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    mp = _FakeMonkeypatch()
    orig_platform = sys.platform
    try:
        test_windows_paths_have_exe_suffix(mp)
        test_linux_paths_have_no_suffix(mp)
    finally:
        sys.platform = orig_platform
