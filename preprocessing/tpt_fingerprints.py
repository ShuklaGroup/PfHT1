"""Build OF-to-IF TPT paths and tICA occupancy fingerprints for PfHT1."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import pickle
from pathlib import Path

import numpy as np


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def microstate_gate_statistics(features, dtrajs, n_states, gate_columns, boxes):
    counts = np.zeros(n_states, dtype=np.int64)
    gate_sums = np.zeros((n_states, 2), dtype=np.float64)
    box_counts = {name: np.zeros(n_states, dtype=np.int64) for name in boxes}

    if len(features) != len(dtrajs):
        raise ValueError("Feature and discrete-trajectory counts differ")
    for trajectory_index, (values, labels) in enumerate(zip(features, dtrajs)):
        values = np.asarray(values)
        labels = np.asarray(labels, dtype=np.int64)
        if len(values) != len(labels):
            raise ValueError(f"Frame mismatch in trajectory {trajectory_index}")
        valid = (labels >= 0) & (labels < n_states)
        labels = labels[valid]
        gates = values[valid][:, gate_columns].astype(np.float64, copy=False)
        counts += np.bincount(labels, minlength=n_states)
        for column in range(2):
            gate_sums[:, column] += np.bincount(
                labels, weights=gates[:, column], minlength=n_states
            )
        for name, limits in boxes.items():
            inside = (
                (gates[:, 0] >= limits["egate"][0])
                & (gates[:, 0] <= limits["egate"][1])
                & (gates[:, 1] >= limits["igate"][0])
                & (gates[:, 1] <= limits["igate"][1])
            )
            box_counts[name] += np.bincount(labels[inside], minlength=n_states)

    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0)
        raise ValueError(f"MSM states without assigned frames: {missing[:20].tolist()}")
    means = gate_sums / counts[:, None]
    fractions = {name: values / counts for name, values in box_counts.items()}
    return counts, means, fractions


def select_endpoints(counts, fractions, minimum_frames, minimum_fraction):
    selected = {
        name: np.flatnonzero(
            (counts >= minimum_frames) & (fractions[name] >= minimum_fraction)
        ).astype(np.int32)
        for name in ("OF", "IF")
    }
    overlap = np.intersect1d(selected["OF"], selected["IF"])
    selected["OF"] = np.setdiff1d(selected["OF"], overlap)
    selected["IF"] = np.setdiff1d(selected["IF"], overlap)
    return selected, overlap


def widest_path(residual, sources, targets):
    targets = set(map(int, targets))
    best = np.zeros(len(residual), dtype=np.float64)
    parent = np.full(len(residual), -1, dtype=np.int32)
    visited = np.zeros(len(residual), dtype=bool)
    queue = []
    for source in sources:
        best[source] = np.inf
        heapq.heappush(queue, (-np.inf, int(source)))

    while queue:
        negative_capacity, node = heapq.heappop(queue)
        capacity = -negative_capacity
        if visited[node]:
            continue
        visited[node] = True
        if node in targets:
            path = [node]
            while parent[path[-1]] != -1:
                path.append(int(parent[path[-1]]))
            return path[::-1], float(capacity)
        for nxt in np.flatnonzero(residual[node] > 0):
            candidate = min(capacity, residual[node, nxt])
            if candidate > best[nxt]:
                best[nxt] = candidate
                parent[nxt] = node
                heapq.heappush(queue, (-candidate, int(nxt)))
    return None, 0.0


def decompose_flux(net_flux, sources, targets, maximum_paths):
    residual = np.asarray(net_flux, dtype=np.float64).copy()
    paths, fluxes = [], []
    for _ in range(maximum_paths):
        path, capacity = widest_path(residual, sources, targets)
        if path is None or capacity <= 0:
            break
        paths.append(path)
        fluxes.append(capacity)
        for left, right in zip(path[:-1], path[1:]):
            residual[left, right] = max(0.0, residual[left, right] - capacity)
    return paths, np.asarray(fluxes, dtype=np.float64)


def pack_paths(paths):
    lengths = np.asarray([len(path) for path in paths], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    states = np.concatenate([np.asarray(path, dtype=np.int32) for path in paths])
    return offsets, states


def aligned_tica(dtrajs, projections, n_states):
    if len(dtrajs) != len(projections):
        raise ValueError("Discrete-trajectory and tICA trajectory counts differ")
    states, coordinates = [], []
    for index, (labels, values) in enumerate(zip(dtrajs, projections)):
        labels = np.asarray(labels, dtype=np.int32)
        values = np.asarray(values)
        if len(labels) != len(values):
            raise ValueError(f"tICA frame mismatch in trajectory {index}")
        valid = (labels >= 0) & (labels < n_states)
        states.append(labels[valid])
        coordinates.append(values[valid, :2].astype(np.float32, copy=False))
    return np.concatenate(states), np.concatenate(coordinates)


def state_histograms(states, coordinates, n_states, grid, percentiles, cap):
    indices, edges = [], []
    for dimension in range(2):
        low, high = np.percentile(coordinates[:, dimension], percentiles)
        edge = np.linspace(low, high, grid + 1)
        index = np.searchsorted(edge, coordinates[:, dimension], side="right") - 1
        edges.append(edge)
        indices.append(np.clip(index, 0, grid - 1))
    flat_bins = indices[0] * grid + indices[1]
    number_bins = grid * grid
    global_counts = np.bincount(flat_bins, minlength=number_bins).astype(float)
    reference = np.median(global_counts[global_counts > 0])
    inverse_density = np.minimum(reference / (global_counts + 1.0), cap)
    combined = states.astype(np.int64) * number_bins + flat_bins
    histograms = np.bincount(
        combined,
        weights=inverse_density[flat_bins],
        minlength=n_states * number_bins,
    ).reshape(n_states, number_bins)
    return histograms, edges


def gate_order(paths, gate_means, egate_threshold, igate_threshold):
    codes = []
    for path in paths:
        states = np.asarray(path, dtype=np.int32)
        egate_crossing = np.flatnonzero(gate_means[states, 0] <= egate_threshold)
        igate_crossing = np.flatnonzero(gate_means[states, 1] >= igate_threshold)
        first_e = int(egate_crossing[0]) if len(egate_crossing) else None
        first_i = int(igate_crossing[0]) if len(igate_crossing) else None
        if first_e is None or first_i is None or first_e == first_i:
            codes.append(2)
        elif first_e < first_i:
            codes.append(0)
        else:
            codes.append(1)
    return np.asarray(codes, dtype=np.int8)


def make_fingerprints(paths, state_histograms_array):
    fingerprints = np.empty((len(paths), state_histograms_array.shape[1]), dtype=np.float32)
    hashes = []
    for index, path in enumerate(paths):
        vector = state_histograms_array[np.asarray(path, dtype=np.int32)].sum(axis=0)
        if vector.sum() <= 0:
            raise ValueError(f"Path {index} has an empty tICA fingerprint")
        fingerprints[index] = vector / vector.sum()
        hashes.append(hashlib.sha256(",".join(map(str, path)).encode("ascii")).hexdigest())
    return fingerprints, np.asarray(hashes)


def prepare_system(name, paths, analysis, output_directory):
    msm = load_pickle(paths["msm"])
    dtrajs = load_pickle(paths["dtrajs"])
    features = load_pickle(paths["features"])
    tica = load_pickle(paths["tica"])
    n_states = len(msm.stationary_distribution)

    counts, gate_means, fractions = microstate_gate_statistics(
        features,
        dtrajs,
        n_states,
        paths["gate_columns"],
        analysis["endpoint_boxes_nm"],
    )
    endpoints, overlap = select_endpoints(
        counts,
        fractions,
        analysis["minimum_assigned_frames"],
        analysis["endpoint_frame_fraction"],
    )
    minimum_states = analysis["minimum_endpoint_states"]
    if min(len(endpoints["OF"]), len(endpoints["IF"])) < minimum_states:
        raise ValueError(f"{name}: endpoint support is below {minimum_states} states")

    stationary = np.asarray(msm.stationary_distribution, dtype=np.float64)
    endpoint_record = {
        "OF_states": endpoints["OF"].tolist(),
        "IF_states": endpoints["IF"].tolist(),
        "removed_overlap_states": overlap.tolist(),
        "OF_stationary_probability": float(stationary[endpoints["OF"]].sum()),
        "IF_stationary_probability": float(stationary[endpoints["IF"]].sum()),
    }
    write_json(output_directory / "endpoints" / f"{name}.json", endpoint_record)

    source_states = endpoints["OF"].astype(int).tolist()
    target_states = endpoints["IF"].astype(int).tolist()
    reactive = msm.reactive_flux(source_states, target_states)
    candidate_paths, candidate_flux = decompose_flux(
        reactive.net_flux,
        endpoints["OF"],
        endpoints["IF"],
        analysis["maximum_candidate_paths"],
    )
    total_flux = float(reactive.total_flux)
    cumulative = np.cumsum(candidate_flux) / total_flux
    reached = np.flatnonzero(cumulative >= analysis["target_flux_fraction"])
    first_reach = int(reached[0] + 1) if len(reached) else None
    return {
        "msm": msm,
        "dtrajs": dtrajs,
        "tica": tica,
        "n_states": n_states,
        "gate_means": gate_means,
        "endpoints": endpoints,
        "endpoint_record": endpoint_record,
        "candidate_paths": candidate_paths,
        "candidate_flux": candidate_flux,
        "total_flux": total_flux,
        "cumulative": cumulative,
        "first_reach": first_reach,
        "forward_committor": np.asarray(reactive.forward_committor),
        "backward_committor": np.asarray(reactive.backward_committor),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    base = config_path.parent
    analysis = config["analysis"]
    output_directory = resolve(base, config["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    systems = {}
    for name, values in config["systems"].items():
        inputs = {
            key: resolve(base, values[key]) for key in ("msm", "dtrajs", "features", "tica")
        }
        inputs["gate_columns"] = [values["egate_feature_index"], values["igate_feature_index"]]
        systems[name] = prepare_system(name, inputs, analysis, output_directory)
        print(
            f"{name}: OF={len(systems[name]['endpoints']['OF'])}, "
            f"IF={len(systems[name]['endpoints']['IF'])}, "
            f"candidate paths={len(systems[name]['candidate_paths'])}",
            flush=True,
        )

    maximum_paths = analysis["maximum_candidate_paths"]
    first_reach = [record["first_reach"] for record in systems.values()]
    if any(value is None or value >= maximum_paths for value in first_reach):
        common_count = maximum_paths
    else:
        rounding = analysis["path_count_rounding"]
        common_count = int(np.ceil(max(first_reach) / rounding) * rounding)
    if any(len(record["candidate_paths"]) < common_count for record in systems.values()):
        raise RuntimeError("A reactive network was exhausted before the shared path count")

    summary = {"common_path_count": common_count, "systems": {}}
    for name, record in systems.items():
        paths = record["candidate_paths"][:common_count]
        flux = record["candidate_flux"][:common_count]
        offsets, packed_states = pack_paths(paths)
        states, coordinates = aligned_tica(record["dtrajs"], record["tica"], record["n_states"])
        histograms, edges = state_histograms(
            states,
            coordinates,
            record["n_states"],
            analysis["fingerprint_grid"],
            analysis["tica_percentiles"],
            analysis["inverse_density_cap"],
        )
        fingerprints, sequence_hash = make_fingerprints(paths, histograms)
        route_code = gate_order(
            paths,
            record["gate_means"],
            analysis["egate_midpoint_nm"],
            analysis["igate_midpoint_nm"],
        )

        (output_directory / "tpt").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_directory / "tpt" / f"{name}_paths.npz",
            path_state_offsets=offsets,
            path_states=packed_states,
            path_flux=flux,
            cumulative_flux_fraction=np.cumsum(flux) / record["total_flux"],
            total_reactive_flux=record["total_flux"],
            source_states=record["endpoints"]["OF"],
            target_states=record["endpoints"]["IF"],
            forward_committor=record["forward_committor"],
            backward_committor=record["backward_committor"],
        )
        (output_directory / "fingerprints").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_directory / "fingerprints" / f"{name}.npz",
            fingerprints=fingerprints,
            sequence_hash=sequence_hash,
            path_flux=flux,
            route_code=route_code,
            path_state_offsets=offsets,
            path_states=packed_states,
            total_reactive_flux=record["total_flux"],
            number_states=record["n_states"],
            tic1_edges=edges[0],
            tic2_edges=edges[1],
        )
        summary["systems"][name] = {
            "captured_flux_fraction": float(flux.sum() / record["total_flux"]),
            "first_path_reaching_target": record["first_reach"],
            "fingerprint_shape": list(fingerprints.shape),
        }
    write_json(output_directory / "summary.json", summary)
    print(f"Shared path count: {common_count}", flush=True)


if __name__ == "__main__":
    main()
