# conftest.py
# SPDX-License-Identifier: Apache-2.0
import os
import pytest

def _get_test_scope():
    scope = os.getenv("XPU_KERNEL_TEST_SCOPE", "").strip().lower()
    if scope:
        return scope
    if os.getenv("XPU_KERNEL_PYTEST_PROFILER", "").strip().upper() == "MINI":
        return "mini"
    return "full"

def _resolve_scope_params(module, func_name, scope):
    if scope.startswith("ondemand:"):
        profile_name = scope.split(":", 1)[1].lstrip(":")
        try:
            from tests.test_scope_profiles import ONDEMAND_PROFILES
        except ImportError:
            return {}
        profile = ONDEMAND_PROFILES.get(profile_name, {})
        module_file = getattr(module, "__file__", "")
        for path_key, funcs in profile.items():
            if module_file.endswith(path_key):
                entry = funcs.get(func_name, funcs.get("default", None))
                return entry
        return None

    scope_key = scope
    if scope_key == "ci":
        return {}
    if scope_key == "mini":
        legacy = getattr(module, "MINI_PYTEST_PARAMS", {})
        entry = legacy.get(func_name)
        if entry is not None:
            return entry
        entry = legacy.get("default")
        if entry is not None:
            return entry
    return {}

def _apply_param_overrides(metafunc, profile):
    new_markers = []
    for mark in metafunc.definition.own_markers:
        if (mark.name == "parametrize" and mark.args[0] in profile):
            param_name = mark.args[0]
            split_names = [n.strip() for n in param_name.split(",")]
            if all(n in metafunc.fixturenames for n in split_names):
                new_mark = pytest.mark.parametrize(param_name, profile[param_name])
                new_markers.append(new_mark.mark)
                continue
        new_markers.append(mark)
    metafunc.definition.own_markers = new_markers

def _skip_test(metafunc, reason):
    new_markers = []
    for m in metafunc.definition.own_markers:
        if m.name == "parametrize" and m.args:
            param_name = m.args[0]
            original_values = m.args[1]
            single = [original_values[0]] if original_values else [None]
            new_markers.append(pytest.mark.parametrize(param_name, single).mark)
        else:
            new_markers.append(m)
    metafunc.definition.own_markers = new_markers

    skip_mark = pytest.mark.skip(reason=reason).mark
    func = metafunc.function
    existing = list(getattr(func, "pytestmark", []))
    existing.append(skip_mark)
    func.pytestmark = existing

def pytest_generate_tests(metafunc):
    scope = _get_test_scope()
    if scope == "full":
        return

    module = metafunc.module
    func_name = metafunc.function.__name__

    if scope == "mini" and getattr(module, "SKIP_IN_MINI_SCOPE", False):
        _skip_test(metafunc, "Skipped in mini scope (SKIP_IN_MINI_SCOPE=True)")
        return

    profile = _resolve_scope_params(module, func_name, scope)
    if profile is None:
        _skip_test(metafunc, f"Skipped in {scope} scope")
        return
    if not profile:
        return

    _apply_param_overrides(metafunc, profile)

@pytest.fixture
def reset_default_device():
    import torch
    original_device = torch.get_default_device()
    yield
    torch.set_default_device(original_device)