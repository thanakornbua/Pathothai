"""
Diagnostics and Preflight Checks for Pathothai Pipeline

Run this script to proactively detect common environment, dependency, and runtime issues
before starting training. It prints a human-readable summary and also returns a JSON dict
from run_all_checks(). Can be used as a module or CLI.
"""
from __future__ import annotations

import os
import sys
import json
import platform
import importlib.util
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, Optional, Iterable, Tuple, Callable, List, TypeAlias

DIAG_VERSION = "1.2.2"

# Optional tqdm progress bar (graceful fallback if not installed)
try:  # noqa: SIM105
    from tqdm import tqdm as _tqdm

    def _progress(iterable: Iterable, total: Optional[int] = None, desc: Optional[str] = None):
        return _tqdm(iterable, total=total, desc=desc)

except Exception:  # pragma: no cover - fallback when tqdm is missing
    def _progress(iterable: Iterable, total: Optional[int] = None, desc: Optional[str] = None):
        # No-op fallback: just return the iterable unchanged
        return iterable


def _bool(x: bool) -> str:
    return "Yes" if x else "No"


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def check_python_env() -> Dict[str, Any]:
    info = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "arch": platform.machine(),
    }
    return {"ok": True, "info": info}


def check_torch_cuda() -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "info": {}}
    try:
        import torch
    except Exception as e:
        return {"ok": False, "error": f"PyTorch import failed: {e}"}

    cuda_ok = torch.cuda.is_available()
    out["info"]["torch_version"] = torch.__version__
    out["info"]["cuda_available"] = cuda_ok

    if cuda_ok:
        try:
            dev_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory
            out["info"]["device_name"] = dev_name
            out["info"]["total_vram_gb"] = round(total_mem / 1e9, 2)
            # AMP bf16 support
            bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            out["info"]["bf16_supported"] = bf16
        except Exception as e:
            out["ok"] = False
            out["error"] = f"CUDA query failed: {e}"
    return out


def check_triton_compile(try_on_windows: bool = False) -> Dict[str, Any]:
    """Detect Triton availability and try a minimal torch.compile probe on CPU to ensure API exists.
    We don't need GPU for API presence; GPU compile will be probed by model forward tests if desired.
    """
    out: Dict[str, Any] = {"ok": True, "info": {}}
    try:
        import torch
    except Exception as e:
        return {"ok": False, "error": f"PyTorch import failed: {e}"}

    # Triton/Inductor is not supported on Windows today; skip probe and do not fail diagnostics
    try:
        import sys as _sys, os as _os
        is_windows = _sys.platform.startswith("win") or _os.name == "nt"
    except Exception:
        is_windows = False

    if is_windows:
        # Reflect module presence accurately, and optionally try a CPU-only torch.compile probe
        has_triton = _find_spec("triton")
        out["info"]["has_triton_module"] = bool(has_triton)
        out["info"]["has_torch_compile"] = hasattr(torch, "compile")
        out["info"]["skipped_on_windows"] = not bool(try_on_windows)
        out["info"]["note"] = "Attempting CPU torch.compile probe on Windows" if try_on_windows else "Skipping torch.compile probe on Windows"
        if not try_on_windows or not hasattr(torch, "compile"):
            out["info"]["compile_probe"] = False
            return out
        # Try a minimal CPU probe; attempt multiple backends to maximize chances
        try:
            class Tiny(torch.nn.Module):
                def forward(self, x):
                    return x.relu()
            m = Tiny()
            success = False
            errors: list[str] = []
            for backend in [None, "inductor", "eager"]:
                try:
                    cm = torch.compile(m, backend=backend) if backend else torch.compile(m)
                    with torch.no_grad():
                        _ = cm(torch.randn(2, 3))
                    out["info"]["compile_probe"] = True
                    out["info"]["compile_backend_used"] = backend or "default"
                    success = True
                    break
                except Exception as _e:
                    errors.append(f"backend={backend or 'default'}: {_e}")
            if not success:
                out["info"]["compile_probe"] = False
                out.setdefault("warnings", []).append("; ".join(errors))
        except Exception as e:
            out["info"]["compile_probe"] = False
            out.setdefault("warnings", []).append(f"torch.compile probe (Windows) failed: {e}")
        return out

    has_triton = _find_spec("triton")
    out["info"]["has_triton_module"] = has_triton
    out["info"]["has_torch_compile"] = hasattr(torch, "compile")

    # Quick compile probe if API exists (CPU-only); attempt multiple backends
    if hasattr(torch, "compile"):
        try:
            class Tiny(torch.nn.Module):
                def forward(self, x):
                    return x.relu()
            m = Tiny()
            success = False
            errors: list[str] = []
            for backend in [None, "inductor", "eager"]:
                try:
                    cm = torch.compile(m, backend=backend) if backend else torch.compile(m)
                    with torch.no_grad():
                        _ = cm(torch.randn(2, 3))
                    out["info"]["compile_probe"] = True
                    out["info"]["compile_backend_used"] = backend or "default"
                    success = True
                    break
                except Exception as _e:
                    errors.append(f"backend={backend or 'default'}: {_e}")
            if not success:
                # Non-Windows: capture failure but keep diagnostics non-fatal
                out["ok"] = True
                out["info"]["compile_probe"] = False
                out.setdefault("warnings", []).append("; ".join(errors))
        except Exception as e:
            out["ok"] = True
            out["info"]["compile_probe"] = False
            out.setdefault("warnings", []).append(f"torch.compile probe failed: {e}")
    else:
        out["info"]["compile_probe"] = False
    return out


def check_optional_deps() -> Dict[str, Any]:
    """
    Checks for optional dependencies using importlib.util.find_spec,
    which detects if the module can be imported (i.e., installed in the environment).
    This does not directly check if the package is installed via pip, but if the module is importable.
    """
    deps = {
        "openslide": _find_spec("openslide") or _find_spec("openslide_python"),
        "monai": _find_spec("monai"),
        "wandb": _find_spec("wandb"),
        "tensorboard": _find_spec("torch.utils.tensorboard") or _find_spec("tensorboard"),
        # pip name: pytorch-grad-cam, module: pytorch_grad_cam (fallback: pytorch_gradcam)
        "pytorch-gradcam": _find_spec("pytorch_grad_cam") or _find_spec("pytorch_gradcam"),
        "sklearn": _find_spec("sklearn"),
        "opencv": _find_spec("cv2"),
        "triton": _find_spec("triton"),
    }
    return {"ok": True, "info": deps}


def check_data_paths(config: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        data_dir = Path(getattr(config, "DATA_DIR", getattr(config, "data", {}).get("data_dir", "data")))
    except Exception:
        data_dir = Path("data")
    ann_dir = Path(getattr(config, "ANNOTATIONS_DIR", getattr(config, "data", {}).get("annotations_dir", "Annotations")))
    svs_dir = Path(getattr(config, "SVS_DIR", data_dir / "SVS"))
    info["data_dir_exists"] = data_dir.exists()
    info["annotations_dir_exists"] = ann_dir.exists()
    info["svs_dir_exists"] = svs_dir.exists()
    info["svs_count"] = len(list(svs_dir.glob("*.svs"))) if svs_dir.exists() else 0
    return {"ok": info["data_dir_exists"], "info": info}


def check_numpy_casting() -> Dict[str, Any]:
    """Verify that bf16 tensors are safely convertible via float32 cast before NumPy conversion."""
    out: Dict[str, Any] = {"ok": True, "info": {}}
    try:
        import torch
        t = torch.ones(2, dtype=torch.bfloat16)
        # direct .numpy() would fail; emulate helper behavior
        _ = t.detach().cpu().to(torch.float32).numpy()
        out["info"]["bf16_to_numpy_safe"] = True
    except Exception as e:
        out["ok"] = False
        out["error"] = f"bf16 numpy conversion failed: {e}"
    return out


def tiny_forward_probes(config: Any) -> Dict[str, Any]:
    """Run tiny forward passes on models to catch missing Triton/compile or shape/device issues early."""
    out: Dict[str, Any] = {"ok": True, "info": {}, "warnings": []}
    try:
        import torch
        from scripts.train import AttentionMIL, SegmentationUNet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # AttentionMIL probe (classification)
        try:
            mil = AttentionMIL(backbone='resnet50', num_classes=getattr(config, 'NUM_CLASSES', 2)).to(device)
            x = torch.randn(1, 1, 3, getattr(config, 'PATCH_SIZE', 256), getattr(config, 'PATCH_SIZE', 256), device=device)
            with torch.no_grad():
                _ = mil(x)
            out["info"]["attention_mil_forward"] = True
        except Exception as e:
            out["ok"] = False
            out["info"]["attention_mil_forward"] = False
            out.setdefault("errors", []).append(f"AttentionMIL forward failed: {e}")

        # Segmentation U-Net probe
        try:
            seg = SegmentationUNet().to(device)
            h = getattr(config, 'PATCH_SIZE_SEG', getattr(config, 'PATCH_SIZE', 256))
            w = getattr(config, 'PATCH_SIZE_SEG', getattr(config, 'PATCH_SIZE', 256))
            x = torch.randn(1, 3, h, w, device=device)
            with torch.no_grad():
                _ = seg(x)
            out["info"]["seg_unet_forward"] = True
        except Exception as e:
            out["ok"] = False
            out["info"]["seg_unet_forward"] = False
            out.setdefault("errors", []).append(f"SegmentationUNet forward failed: {e}")
    except Exception as e:
        out["ok"] = False
        out.setdefault("errors", []).append(f"Model probe setup failed: {e}")
    return out


def check_gradcam_smoke(config: Optional[Any]) -> Dict[str, Any]:
    """Run a minimal Grad-CAM/explainability smoke test.

    This is intentionally tolerant: it will not fail the overall diagnostics if
    the optional dependency is missing or no checkpoint is found. Instead it
    records availability and whether a tiny explain run executed without error.
    """
    out: Dict[str, Any] = {"ok": True, "info": {}}
    try:
        # Detect explain function and optional dependency
        has_pkg = _find_spec("pytorch_grad_cam") or _find_spec("pytorch_gradcam")
        out["info"]["has_pytorch_gradcam"] = bool(has_pkg)

        try:
            from scripts.train import explain_predictions
            has_explain = callable(explain_predictions)
        except Exception:
            has_explain = False
        out["info"]["has_explain_fn"] = has_explain

        # Resolve checkpoint candidates
        ckpt_dir = None
        try:
            ckpt_dir = Path(getattr(config, "CHECKPOINT_DIR", "checkpoints")) if config is not None else Path("checkpoints")
        except Exception:
            ckpt_dir = Path("checkpoints")

        candidates = [
            ckpt_dir / "phase2_fold0_best.pth",
            ckpt_dir / "phase1_fold0_best.pth",
            Path("checkpoints") / "best_model.pth",
            Path("checkpoints") / "latest_checkpoint.pth",
        ]
        ckpt_path = next((p for p in candidates if p.exists()), None)
        out["info"]["checkpoint"] = str(ckpt_path) if ckpt_path else None

        # Quick exits: no function or no checkpoint
        if not has_explain:
            out["info"]["executed"] = False
            out["info"]["note"] = "explain_predictions not available"
            return out
        if ckpt_path is None:
            out["info"]["executed"] = False
            out["info"]["note"] = "no checkpoint found"
            return out

        # Safety: disable torch.compile for explain step if Triton is missing
        try:
            if config is not None and getattr(config, "USE_TORCH_COMPILE", False) and not _find_spec("triton"):
                setattr(config, "USE_TORCH_COMPILE", False)
                out["info"]["compile_disabled_for_explain"] = True
            # Provide minimal defaults expected by explain if missing
            if config is not None and not hasattr(config, "NUM_CLASSES"):
                try:
                    setattr(config, "NUM_CLASSES", 2)
                    out["info"]["defaulted_NUM_CLASSES"] = 2
                except Exception:
                    pass
        except Exception:
            pass

        # Execute a tiny explain run
        # Suppress noisy prints during smoke run to keep diagnostics clean
        from contextlib import contextmanager
        import sys as _sys
        import os as _os

        @contextmanager
        def _suppress_output():  # noqa: N802
            try:
                null = open(_os.devnull, 'w')
                _old_out, _old_err = _sys.stdout, _sys.stderr
                _sys.stdout, _sys.stderr = null, null
                yield
            finally:
                try:
                    _sys.stdout, _sys.stderr = _old_out, _old_err
                    null.close()
                except Exception:
                    pass

        try:
            with _suppress_output():
                ckpt_for_explain = str(ckpt_path)
                _extracted_tmp_path = None
                # If checkpoint is a training bundle, extract state_dict to a temp file
                try:
                    import torch as _torch
                    import tempfile as _tempfile
                    obj = _torch.load(ckpt_path, map_location="cpu")
                    sd = None
                    if isinstance(obj, dict):
                        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
                            sd = obj["model_state_dict"]
                        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
                            sd = obj["state_dict"]
                    if sd is not None:
                        # Strip common prefixes such as 'module.' or 'model.' often present in trainer checkpoints
                        def _strip_prefixes(d: dict) -> dict:
                            prefixes = ("model.module.", "module.", "model.", "net.")
                            new_d = {}
                            for k, v in d.items():
                                nk = k
                                changed = True
                                while changed:
                                    changed = False
                                    for p in prefixes:
                                        if nk.startswith(p):
                                            nk = nk[len(p):]
                                            changed = True
                                new_d[nk] = v
                            return new_d

                        sd = _strip_prefixes(sd)
                        tmp = _tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
                        tmp.close()
                        _torch.save(sd, tmp.name)
                        ckpt_for_explain = tmp.name
                        _extracted_tmp_path = tmp.name
                        out["info"]["checkpoint_extracted_state_dict"] = True
                        out["info"]["checkpoint_extracted_path"] = ckpt_for_explain
                except Exception:
                    # If load/extract fails, proceed with original path
                    pass

                explain_predictions(config, ckpt_for_explain, fold=0, num_samples=1)  # type: ignore[misc]
                # Cleanup any temporary checkpoint file
                try:
                    if _extracted_tmp_path and _os.path.exists(_extracted_tmp_path):
                        _os.remove(_extracted_tmp_path)
                except Exception:
                    pass
            out["info"]["executed"] = True
            out["info"]["note"] = "explain ran (overlays may be skipped if optional deps missing)"
        except Exception as e:
            # Do not fail overall diagnostics; record the error
            out["ok"] = True  # keep tolerant
            out.setdefault("errors", []).append(f"explain_predictions error: {e}")
            out["info"]["executed"] = False
    except Exception as e:
        # Unexpected wrapper error; keep tolerant but record
        out["ok"] = True
        out.setdefault("errors", []).append(f"gradcam smoke setup failed: {e}")
    return out


def run_all_checks(config: Optional[Any] = None, use_tqdm: bool = True, try_compile_probe_on_windows: bool = True) -> Dict[str, Any]:
    """Run all diagnostics checks with optional tqdm progress bars.

    Parameters:
    - config: optional configuration object to enable data/model-specific checks
    - use_tqdm: whether to display a tqdm progress bar (falls back to no-op if tqdm missing)
    """
    results: Dict[str, Any] = {"version": DIAG_VERSION}

    # Build task list as (name, callable) to iterate with progress
    Task: TypeAlias = Tuple[str, Callable[[], Dict[str, Any]]]
    tasks: List[Task] = [
        ("python_env", check_python_env),
        ("torch_cuda", check_torch_cuda),
        ("triton_compile", lambda: check_triton_compile(try_on_windows=try_compile_probe_on_windows)),
        ("optional_deps", check_optional_deps),
        ("numpy_casting", check_numpy_casting),
    ]

    if config is not None:
        tasks.extend([
            ("data_paths", lambda: check_data_paths(config)),
            ("tiny_forwards", lambda: tiny_forward_probes(config)),
            ("gradcam_smoke", lambda: check_gradcam_smoke(config)),
        ])

    iterable = _progress(tasks, total=len(tasks), desc="Diagnostics") if use_tqdm else tasks

    def _run_loop(loop_tasks: List[Task]):
        for name, fn in loop_tasks:
            try:
                results[name] = fn()
            except Exception as e:
                # Ensure a well-formed section even on unexpected errors
                results[name] = {"ok": False, "error": f"{name} failed: {e}"}

    try:
        for name, fn in iterable:
            try:
                results[name] = fn()
            except Exception as e:
                results[name] = {"ok": False, "error": f"{name} failed: {e}"}
    except AttributeError:
        # Some tqdm builds may raise AttributeError (e.g., missing disp). Fallback to plain loop.
        _run_loop(tasks)
    finally:
        try:
            if hasattr(iterable, "close"):
                iterable.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    # Overall status: consider only dict sections that carry an 'ok' flag
    ok = True
    for section in results.values():
        if isinstance(section, dict) and 'ok' in section:
            ok = ok and bool(section.get('ok', True))
    results["overall_ok"] = ok
    return results


def _print_human(results: Dict[str, Any]) -> None:
    print("Diagnostics Summary")
    print("=" * 20)
    py = results.get("python_env", {})
    print(f"Python: {py.get('info', {}).get('python', 'n/a')}")
    tc = results.get("torch_cuda", {})
    tci = tc.get('info', {})
    print(f"PyTorch: {tci.get('torch_version', 'n/a')} | CUDA: {_bool(tci.get('cuda_available', False))}")
    if tci.get('cuda_available', False):
        print(f"  GPU: {tci.get('device_name', 'n/a')} | VRAM: {tci.get('total_vram_gb', 'n/a')} GB | bf16: {_bool(tci.get('bf16_supported', False))}")
    tr = results.get("triton_compile", {})
    tri = tr.get('info', {})
    print(f"torch.compile: {_bool(tri.get('has_torch_compile', False))} | Triton: {_bool(tri.get('has_triton_module', False))} | compile_probe: {_bool(tri.get('compile_probe', False))}")
    deps = results.get("optional_deps", {}).get("info", {})
    print("Optional deps:")
    for k, v in deps.items():
        print(f"  - {k}: {_bool(bool(v))}")
    npcast = results.get("numpy_casting", {})
    # Use ASCII arrow for Windows console compatibility
    print(f"bf16->NumPy safe path: {_bool(npcast.get('ok', False))}")
    if "data_paths" in results:
        dp = results["data_paths"]["info"]
        print(f"Data paths: data={_bool(dp.get('data_dir_exists', False))}, ann={_bool(dp.get('annotations_dir_exists', False))}, svs={_bool(dp.get('svs_dir_exists', False))} (count={dp.get('svs_count', 0)})")
    if "tiny_forwards" in results:
        tf = results["tiny_forwards"]
        print("Tiny forward probes:")
        print(f"  AttentionMIL: {_bool(tf.get('info', {}).get('attention_mil_forward', False))}")
        print(f"  SegmentationUNet: {_bool(tf.get('info', {}).get('seg_unet_forward', False))}")
    if "gradcam_smoke" in results:
        gs = results["gradcam_smoke"].get("info", {})
        print("Grad-CAM smoke:")
        print(f"  explain fn: {_bool(gs.get('has_explain_fn', False))} | pkg: {_bool(gs.get('has_pytorch_gradcam', False))}")
        ck = gs.get('checkpoint', None)
        print(f"  checkpoint: {ck if ck else 'none'} | executed: {_bool(gs.get('executed', False))}")
    print(f"Overall OK: {_bool(results.get('overall_ok', False))}")


if __name__ == "__main__":
    # Attempt to import notebook config if available
    config = None
    try:
        from scripts.train import Config as TrainConfig
        config = TrainConfig()
    except Exception:
        pass
    res = run_all_checks(config)
    _print_human(res)
    # Save JSON
    try:
        out_dir = Path("output") / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"Saved diagnostics to {out_dir / 'diagnostics.json'}")
    except Exception as e:
        print(f"Could not save diagnostics: {e}")
