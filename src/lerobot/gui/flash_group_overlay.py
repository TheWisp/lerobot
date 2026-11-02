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
        # Own-object points and borrowed surroundings are distinct evidence
        # sets: never imply a measured surface connecting them across empty space.
        own = set(group.get("object_ids", []))
        groups = np.array([0 if i in own else 1 for i in ids], dtype=int)
        mesh = self.wire.update(
            tracks, np.asarray(pixels).reshape(-1, 2), np.asarray(xyz).reshape(-1, 3), groups
        )
        return {
            **{k: v for k, v in group.items() if not k.endswith("camera_xyz")},
            "links": mesh.links.tolist(),
        }
