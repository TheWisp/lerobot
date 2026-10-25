"""Research-only exact CUDA traversal of SciPy's prepared cKDTree."""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl
from scipy.spatial import cKDTree


@triton.jit
def _query_kernel(
    query,
    points,
    indices,
    split_dim,
    split_value,
    lesser,
    greater,
    parent,
    start,
    end,
    result,
    count: tl.constexpr,
    leaf_size: tl.constexpr,
    block: tl.constexpr,
):
    lane = tl.program_id(0) * block + tl.arange(0, block)
    valid_query = lane < count
    qx = tl.load(query + 3 * lane, mask=valid_query, other=0)
    qy = tl.load(query + 3 * lane + 1, mask=valid_query, other=0)
    qz = tl.load(query + 3 * lane + 2, mask=valid_query, other=0)
    node = tl.where(valid_query, 0, -1)
    previous = tl.full((block,), -1, tl.int32)
    best = tl.full((block,), float("inf"), tl.float64)
    leaf_offset = tl.arange(0, leaf_size)

    while tl.sum((node >= 0).to(tl.int32), 0) > 0:
        active = node >= 0
        safe_node = tl.maximum(node, 0)
        dimension = tl.load(split_dim + safe_node, mask=active, other=-1)
        is_leaf = active & (dimension < 0)
        parent_node = tl.load(parent + safe_node, mask=active, other=-1)

        leaf_start = tl.load(start + safe_node, mask=is_leaf, other=0)
        leaf_end = tl.load(end + safe_node, mask=is_leaf, other=0)
        positions = leaf_start[:, None] + leaf_offset[None, :]
        valid_point = is_leaf[:, None] & (positions < leaf_end[:, None])
        point_index = tl.load(indices + positions, mask=valid_point, other=0)
        px = tl.load(points + 3 * point_index, mask=valid_point, other=0)
        py = tl.load(points + 3 * point_index + 1, mask=valid_point, other=0)
        pz = tl.load(points + 3 * point_index + 2, mask=valid_point, other=0)
        dx = px - qx[:, None]
        dy = py - qy[:, None]
        dz = pz - qz[:, None]
        distance = dx * dx + dy * dy + dz * dz
        best = tl.minimum(best, tl.min(tl.where(valid_point, distance, float("inf")), 1))

        internal = active & ~is_leaf
        split = tl.load(split_value + safe_node, mask=internal, other=0)
        left = tl.load(lesser + safe_node, mask=internal, other=-1)
        right = tl.load(greater + safe_node, mask=internal, other=-1)
        coordinate = tl.where(dimension == 0, qx, tl.where(dimension == 1, qy, qz))
        near = tl.where(coordinate <= split, left, right)
        far = tl.where(coordinate <= split, right, left)
        entering = previous == parent_node
        after_near = previous == near
        split_distance = coordinate - split
        far_possible = split_distance * split_distance <= best
        next_internal = tl.where(entering, near, tl.where(after_near & far_possible, far, parent_node))
        next_node = tl.where(is_leaf, parent_node, next_internal)
        previous = tl.where(active, node, previous)
        node = tl.where(active, next_node, -1)

    tl.store(result + lane, tl.sqrt(best), mask=valid_query)


class ExactGpuKdTree:
    """Transfer the same cKDTree splits and leaves once, then query on CUDA."""

    def __init__(self, tree: cKDTree):
        nodes = []

        def visit(node, parent_index):
            index = len(nodes)
            nodes.append((node, parent_index, -1, -1))
            if node.split_dim >= 0:
                left = visit(node.lesser, index)
                right = visit(node.greater, index)
                nodes[index] = (node, parent_index, left, right)
            return index

        visit(tree.tree, -1)

        def device_array(values, dtype):
            return torch.as_tensor(np.asarray(values, dtype=dtype), device="cuda").contiguous()

        self.points = device_array(tree.data, np.float64).reshape(-1)
        self.indices = device_array(tree.indices, np.int32)
        self.dimension = device_array([node.split_dim for node, _, _, _ in nodes], np.int32)
        self.split = device_array([node.split for node, _, _, _ in nodes], np.float64)
        self.lesser = device_array([left for _, _, left, _ in nodes], np.int32)
        self.greater = device_array([right for _, _, _, right in nodes], np.int32)
        self.parent = device_array([parent_index for _, parent_index, _, _ in nodes], np.int32)
        self.start = device_array([node.start_idx for node, _, _, _ in nodes], np.int32)
        self.end = device_array([node.end_idx for node, _, _, _ in nodes], np.int32)
        # cKDTree may keep more than its requested leafsize when identical
        # coordinates cannot be split. Cover the actual largest leaf.
        largest_leaf = max(node.end_idx - node.start_idx for node, _, _, _ in nodes if node.split_dim < 0)
        self.leaf_size = triton.next_power_of_2(largest_leaf)

    def query(self, points: torch.Tensor) -> torch.Tensor:
        if points.device.type != "cuda" or points.dtype != torch.float64 or points.shape[-1] != 3:
            raise ValueError("queries must be CUDA float64 XYZ points")
        flat = points.reshape(-1, 3).contiguous()
        result = torch.empty(len(flat), dtype=torch.float64, device=flat.device)
        block = 32
        _query_kernel[(triton.cdiv(len(flat), block),)](
            flat,
            self.points,
            self.indices,
            self.dimension,
            self.split,
            self.lesser,
            self.greater,
            self.parent,
            self.start,
            self.end,
            result,
            len(flat),
            self.leaf_size,
            block,
        )
        return result.reshape(points.shape[:-1])
