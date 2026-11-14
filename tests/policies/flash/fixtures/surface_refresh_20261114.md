# Surface refresh regression inputs

Lossless NPZ captures from the shared recorded replay. Camera coordinates are
metres; intrinsics, poses, masks and recorded RGB/depth are retained unchanged.
Do not regenerate these by running the current algorithm and blessing its output.

- `ring_empty_refresh_20261114.npz`: accepted source frame 290, later fit frame 340.
  SHA256 `5c084db3a2eb61338ea7829bad8e7e37fc69a3ddfee38dc88e36acc09f0f91bf`.
- `dowel_empty_refresh_20261114.npz`: accepted source frame 165, later fit frame 166.
- `green_toy_boundary_20261113.npz`: frozen current/retained measurement, stored map,
  proposed pose and limits from the boundary-depth experiment. It is a rejection
  reproduction, not a ground-truth pose label.

Ring and dowel reproduce a frame accepted without refreshing any exact map
coordinates. Their later real small_gicp fit must still be able to return a
supported pose; independent 100 mm displacements must fail. Boundary tests also
use displaced and rotated controls and verify uncertainty in actual preparation.

RGB is necessary for the library appearance consumer; depth remains lossless.
These three files intentionally exceed the normal 1 MiB added-file limit (1.6,
1.6 and 4.4 MiB) and are the only new fixture exceptions in the commit hook.

SHA256 (all inputs):

- `ring_empty_refresh_20261114.npz`: `5c084db3a2eb61338ea7829bad8e7e37fc69a3ddfee38dc88e36acc09f0f91bf`
- `dowel_empty_refresh_20261114.npz`: `978d8b86617dfebc457b1ed8091505a4d0fc72dea38879b9eee815cd97c39d1d`
- `green_toy_boundary_20261113.npz`: `d9402391492f1c46e359f7dc93db34de187e1dd4beda29300d0237058dfce2df`
