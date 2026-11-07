"""Display measured group support with the sealed demo's persistent wireframe.

The drawing code in group_wireframe.py is copied unchanged from show-and-servo
80d90b783. Its triangle/quad links are illustration, not new tracking constraints.
This adapter maps signed track IDs into the dense IDs that renderer expects.
"""

import numpy as np

from .group_wireframe import Wireframe


class GroupOverlay:
    def __init__(self):
        self.wire = Wireframe(joins=False)
        self.identities = {}

    def update(self, group):
        ids = group.get("display_ids", group.get("ids", []))
        pixels = group.get("display_pixels", group.get("pixels", []))
        xyz = group.get("display_camera_xyz", group.get("camera_xyz", []))
        if not (len(ids) == len(pixels) == len(xyz)):
            raise ValueError("Group display IDs, pixels and measurements disagree")
        tracks = np.array([self.identities.setdefault(i, len(self.identities)) for i in ids], dtype=int)
        # Current output supplies actual scene-group IDs. Historical recordings
        # retain their original two-set overlay for faithful playback.
        if "display_group_ids" in group:
            groups = np.asarray(group["display_group_ids"], dtype=int)
            if groups.shape != tracks.shape:
                raise ValueError("Group display membership and point IDs disagree")
        else:
            own = set(group.get("object_ids", []))
            groups = np.array([0 if i in own else 1 for i in ids], dtype=int)
        mesh = self.wire.update(
            tracks, np.asarray(pixels).reshape(-1, 2), np.asarray(xyz).reshape(-1, 3), groups
        )
        return {
            **{k: v for k, v in group.items() if not k.endswith("camera_xyz")},
            "links": mesh.links.tolist(),
        }


class SceneGroupOverlay:
    """Draw persistent links from scene point IDs, independent of object poses."""

    def __init__(self):
        self.wire = Wireframe(joins=False)
        self.identities = {}

    def update(self, records, *, frame_size=None):
        tracks, pixels, xyz, membership = [], [], [], []
        offsets = []
        for record in records:
            ids = record.get("ids", [])
            uv = record.get("pixels", [])
            points = record.get("camera_xyz", [])
            if not (len(ids) == len(uv) == len(points)):
                raise ValueError("Scene group display IDs, pixels and measurements disagree")
            offsets.append((len(tracks), len(tracks) + len(ids)))
            tracks.extend(self.identities.setdefault(int(i), len(self.identities)) for i in ids)
            pixels.extend(uv)
            xyz.extend(points)
            membership.extend([int(record["id"])] * len(ids))
        mesh = self.wire.update(
            np.asarray(tracks, dtype=int),
            np.asarray(pixels, dtype=float).reshape(-1, 2),
            np.asarray(xyz, dtype=float).reshape(-1, 3),
            np.asarray(membership, dtype=int),
        )
        # Sparse co-moving points can belong to the same group while being far
        # apart in the image. Keep their shared color, but draw only local mesh
        # edges. This is a display limit, never a group-membership rule.
        max_span = min(frame_size) / 10 if frame_size is not None else None
        uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
        result = []
        for record, (start, stop) in zip(records, offsets, strict=True):
            links = [
                [int(a - start), int(b - start)]
                for a, b in mesh.links
                if start <= a < stop
                and start <= b < stop
                and (max_span is None or np.linalg.norm(uv[a] - uv[b]) <= max_span)
            ]
            result.append({**record, "links": links})
        return result
