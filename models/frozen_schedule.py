"""Utilities for strict frozen-schedule checkpoint transfer."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def _is_schedule_key(key: str) -> bool:
    return ".schedule." in key or key.startswith("schedule.")


def _format_key_sample(keys: list[str], limit: int = 3) -> str:
    sample = ", ".join(repr(key) for key in keys[:limit])
    if len(keys) > limit:
        sample += ", ..."
    return sample if sample else "<none>"


def _format_collision_sample(
    collisions: list[tuple[str, str, str]],
    limit: int = 2,
) -> str:
    entries = []
    for source_key, target_key, existing_source_key in collisions[:limit]:
        entries.append(
            f"{source_key!r} -> {target_key!r} (already mapped by {existing_source_key!r})"
        )
    suffix = ", ..." if len(collisions) > limit else ""
    return ", ".join(entries) + suffix if entries else "<none>"


def _format_shape_mismatch_sample(
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]],
    limit: int = 2,
) -> str:
    entries = []
    for target_key, source_shape, target_shape in shape_mismatches[:limit]:
        entries.append(
            f"{target_key!r}: source_shape={source_shape}, target_shape={target_shape}"
        )
    suffix = ", ..." if len(shape_mismatches) > limit else ""
    return ", ".join(entries) + suffix if entries else "<none>"


def build_verified_schedule_state(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    ckpt_path: str | None = None,
) -> dict[str, torch.Tensor]:
    """Map schedule parameters from checkpoint -> target model with strict checks.

    Mapping uses progressive suffix matching so that wrapped checkpoint keys
    (e.g. ``backbone.blocks.0.attn_qkv.schedule.alpha_raw``) can be
    transferred to backbone keys (e.g.
    ``blocks.0.attn_qkv.schedule.alpha_raw``).
    """

    ckpt_label = ckpt_path if ckpt_path is not None else "<checkpoint>"
    source_schedule_keys = sorted(k for k in source_state if _is_schedule_key(k))
    target_schedule_keys = sorted(k for k in target_state if _is_schedule_key(k))

    if not source_schedule_keys:
        raise RuntimeError(
            f"[frozen_schedule] No schedule parameters found in checkpoint {ckpt_label}"
        )
    if not target_schedule_keys:
        raise RuntimeError(
            "[frozen_schedule] Target model has no schedule parameters to load."
        )

    target_schedule_key_set = set(target_schedule_keys)
    schedule_state: dict[str, torch.Tensor] = {}
    mapped_source_by_target: dict[str, str] = {}

    unmapped_source_keys: list[str] = []
    target_collisions: list[tuple[str, str, str]] = []
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for source_key in source_schedule_keys:
        parts = source_key.split(".")
        matched_target_key = None
        for start_idx in range(len(parts)):
            candidate = ".".join(parts[start_idx:])
            if candidate in target_schedule_key_set:
                matched_target_key = candidate
                break

        if matched_target_key is None:
            unmapped_source_keys.append(source_key)
            continue

        if matched_target_key in mapped_source_by_target:
            target_collisions.append(
                (
                    source_key,
                    matched_target_key,
                    mapped_source_by_target[matched_target_key],
                )
            )
            continue

        source_tensor = source_state[source_key]
        target_tensor = target_state[matched_target_key]
        source_shape = tuple(source_tensor.shape)
        target_shape = tuple(target_tensor.shape)
        if source_shape != target_shape:
            shape_mismatches.append((matched_target_key, source_shape, target_shape))
            continue

        mapped_source_by_target[matched_target_key] = source_key
        schedule_state[matched_target_key] = source_tensor

    missing_target_keys = sorted(target_schedule_key_set - set(schedule_state))

    errors = []
    if unmapped_source_keys:
        errors.append(
            f"{len(unmapped_source_keys)} source schedule key(s) were not matched "
            f"(sample: {_format_key_sample(unmapped_source_keys)})."
        )
    if target_collisions:
        errors.append(
            f"{len(target_collisions)} source schedule key(s) collided onto an already "
            f"mapped target key (sample: {_format_collision_sample(target_collisions)})."
        )
    if shape_mismatches:
        errors.append(
            f"{len(shape_mismatches)} matched schedule key(s) had tensor-shape "
            f"mismatches (sample: {_format_shape_mismatch_sample(shape_mismatches)})."
        )
    if missing_target_keys:
        errors.append(
            f"{len(missing_target_keys)} target schedule key(s) were missing after "
            f"transfer (sample: {_format_key_sample(missing_target_keys)})."
        )

    if errors:
        raise RuntimeError(
            "[frozen_schedule] Incomplete schedule transfer; refusing to freeze "
            f"schedules. Checkpoint: {ckpt_label}. " + " ".join(errors)
        )

    return schedule_state
