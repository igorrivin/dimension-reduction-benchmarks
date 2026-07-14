"""Profile FP32 versus BF16-autocast training for the large-data autoencoder.

This is deliberately a training-step microbenchmark, not a quality benchmark.  It
reproduces the two autoencoder shapes used by the large arXiv and Wikipedia runs and
uses their batch size, FP32 parameters, FP32 reconstruction loss, and AdamW settings.
Synthetic resident CUDA tensors isolate the numerical training work from data loading,
validation, epoch selection, and final embedding extraction.

Run in the RAPIDS environment on a CUDA device::

    python experiments/profile_ae_precision.py
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CASES = (
    {
        "dataset": "arxiv",
        "input_features": 384,
        "hidden_dimensions": (256, 128, 32),
    },
    {
        "dataset": "wikipedia",
        "input_features": 1024,
        "hidden_dimensions": (512, 256, 64),
    },
)
TARGET_DIMENSIONS = (2, 8)
PRECISIONS = ("fp32", "bf16_autocast")


def _build_autoencoder(torch, n_features, target_dimension, hidden_dimensions):
    """Build the architecture used by the archived large-data harness."""

    class Autoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            encoder = []
            previous = n_features
            for width in hidden_dimensions:
                encoder.extend((torch.nn.Linear(previous, width), torch.nn.SiLU()))
                previous = width
            encoder.append(torch.nn.Linear(previous, target_dimension))
            self.encoder = torch.nn.Sequential(*encoder)

            decoder = []
            previous = target_dimension
            for width in reversed(hidden_dimensions):
                decoder.extend((torch.nn.Linear(previous, width), torch.nn.SiLU()))
                previous = width
            decoder.append(torch.nn.Linear(previous, n_features))
            self.decoder = torch.nn.Sequential(*decoder)

        def forward(self, batch):
            return self.decoder(self.encoder(batch))

    return Autoencoder()


def _distribution(values):
    samples = np.asarray(values, dtype=np.float64)
    q1, median, q3 = np.percentile(samples, (25.0, 50.0, 75.0))
    return {
        "n": int(samples.size),
        "mean_ms": float(np.mean(samples)),
        "sample_sd_ms": float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0,
        "median_ms": float(median),
        "q1_ms": float(q1),
        "q3_ms": float(q3),
        "iqr_ms": float(q3 - q1),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
    }


def _scalar_distribution(values):
    samples = np.asarray(values, dtype=np.float64)
    q1, median, q3 = np.percentile(samples, (25.0, 50.0, 75.0))
    return {
        "n": int(samples.size),
        "mean": float(np.mean(samples)),
        "sample_sd": float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }


def _state_dtypes(optimizer):
    dtypes = set()
    for state in optimizer.state.values():
        for value in state.values():
            if hasattr(value, "dtype"):
                dtypes.add(str(value.dtype))
    return sorted(dtypes)


def _benchmark_precision(
    torch,
    *,
    case,
    target_dimension,
    precision,
    batch_size,
    warmup_steps,
    measured_steps,
    learning_rate,
    weight_decay,
    seed,
):
    use_bf16 = precision == "bf16_autocast"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")

    # Generate the same data and initialization for both precision modes.  Keeping the
    # batch resident on the GPU matches the fast path selected by the large harness.
    batch = torch.randn(
        batch_size,
        case["input_features"],
        device=device,
        dtype=torch.float32,
    )
    model = _build_autoencoder(
        torch,
        case["input_features"],
        target_dimension,
        case["hidden_dimensions"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    model.train()

    def training_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            prediction = model(batch)
            # This is intentionally outside autocast's reduced-precision arithmetic:
            # it is the exact FP32 loss expression used by the large-data harness.
            loss = torch.mean((prediction.float() - batch.float()) ** 2)
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(warmup_steps):
        loss = training_step()
    torch.cuda.synchronize()

    event_rows = []
    losses = []
    wall_started = time.perf_counter()
    for _ in range(measured_steps):
        total_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        optimizer_end = torch.cuda.Event(enable_timing=True)

        total_start.record()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            prediction = model(batch)
            loss = torch.mean((prediction.float() - batch.float()) ** 2)
        forward_end.record()
        loss.backward()
        backward_end.record()
        optimizer.step()
        optimizer_end.record()
        losses.append(loss.detach())
        event_rows.append((total_start, forward_end, backward_end, optimizer_end))

    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started

    timings = {
        "total_step": [],
        "forward_and_loss": [],
        "backward": [],
        "forward_backward": [],
        "optimizer_step": [],
    }
    for total_start, forward_end, backward_end, optimizer_end in event_rows:
        timings["total_step"].append(total_start.elapsed_time(optimizer_end))
        timings["forward_and_loss"].append(total_start.elapsed_time(forward_end))
        timings["backward"].append(forward_end.elapsed_time(backward_end))
        timings["forward_backward"].append(total_start.elapsed_time(backward_end))
        timings["optimizer_step"].append(backward_end.elapsed_time(optimizer_end))

    loss_values = torch.stack(losses).cpu().numpy()
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    gradient_dtypes = sorted(
        {
            str(parameter.grad.dtype)
            for parameter in model.parameters()
            if parameter.grad is not None
        }
    )
    result = {
        "precision": precision,
        "autocast_enabled": use_bf16,
        "autocast_dtype": "torch.bfloat16" if use_bf16 else None,
        "parameter_dtypes": parameter_dtypes,
        "gradient_dtypes": gradient_dtypes,
        "optimizer_state_dtypes": _state_dtypes(optimizer),
        "loss_dtype": str(loss.dtype),
        "loss_first": float(loss_values[0]),
        "loss_last": float(loss_values[-1]),
        "loss_finite": bool(np.isfinite(loss_values).all()),
        "timings": {name: _distribution(values) for name, values in timings.items()},
        "wall_clock": {
            "total_seconds": wall_seconds,
            "mean_ms_per_step": 1000.0 * wall_seconds / measured_steps,
        },
    }
    median_ms = result["timings"]["total_step"]["median_ms"]
    result["median_examples_per_second"] = 1000.0 * batch_size / median_ms

    return result


def _speedups(fp32, bf16):
    speedups = {}
    for name in fp32["timings"]:
        fp32_median = fp32["timings"][name]["median_ms"]
        bf16_median = bf16["timings"][name]["median_ms"]
        speedups[name] = fp32_median / bf16_median
    speedups["throughput"] = bf16["median_examples_per_second"] / fp32["median_examples_per_second"]
    return speedups


def _environment(torch):
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_index": device,
        "device_name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_device_memory_bytes": properties.total_memory,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32_768)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measured-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/large-ae-precision/microprofile.json"),
    )
    args = parser.parse_args(argv)
    if min(args.batch_size, args.warmup_steps, args.measured_steps) < 1:
        parser.error("batch size, warm-up steps, and measured steps must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("this microbenchmark requires an available CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")

    environment = _environment(torch)
    print(
        f"CUDA device: {environment['device_name']}; torch {environment['torch']}; "
        f"BF16 supported: {environment['bf16_supported']}",
        flush=True,
    )
    cases = []
    for case_index, case in enumerate(CASES):
        for target_dimension in TARGET_DIMENSIONS:
            print(
                f"Profiling {case['dataset']} {case['input_features']} -> "
                f"{target_dimension} ...",
                flush=True,
            )
            precision_results = {}
            for precision in PRECISIONS:
                precision_results[precision] = _benchmark_precision(
                    torch,
                    case=case,
                    target_dimension=target_dimension,
                    precision=precision,
                    batch_size=args.batch_size,
                    warmup_steps=args.warmup_steps,
                    measured_steps=args.measured_steps,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    seed=args.seed + case_index * 100 + target_dimension,
                )
                median = precision_results[precision]["timings"]["total_step"]["median_ms"]
                print(f"  {precision}: {median:.3f} ms/step (median)", flush=True)
            speedups = _speedups(precision_results["fp32"], precision_results["bf16_autocast"])
            print(f"  BF16 speedup: {speedups['total_step']:.3f}x", flush=True)
            cases.append(
                {
                    **case,
                    "hidden_dimensions": list(case["hidden_dimensions"]),
                    "target_dimension": target_dimension,
                    "parameter_count": sum(
                        parameter.numel()
                        for parameter in _build_autoencoder(
                            torch,
                            case["input_features"],
                            target_dimension,
                            case["hidden_dimensions"],
                        ).parameters()
                    ),
                    "precision_results": precision_results,
                    "speedup_fp32_over_bf16": speedups,
                }
            )

    total_speedups = [row["speedup_fp32_over_bf16"]["total_step"] for row in cases]
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "large-autoencoder CUDA training-step precision microprofile",
        "scope": (
            "Synthetic resident-device training steps only; this does not measure "
            "validation quality or end-to-end fit time."
        ),
        "protocol": {
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "input_dtype": "torch.float32",
            "parameter_dtype": "torch.float32",
            "loss": "mean((prediction.float() - batch.float()) ** 2)",
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_scaler": False,
            "activation": "SiLU",
            "data": "deterministic synthetic standard-normal resident CUDA batch",
            "timing": "CUDA events; one device synchronization after all measured steps",
            "seed": args.seed,
            "precision_order": list(PRECISIONS),
        },
        "environment": environment,
        "cases": cases,
        "summary": {
            "total_step_speedups": _scalar_distribution(total_speedups),
            "minimum_total_step_speedup": min(total_speedups),
            "maximum_total_step_speedup": max(total_speedups),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
