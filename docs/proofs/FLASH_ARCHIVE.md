# FLASH research evidence archive

The FLASH integration retains its regression tests, fixtures, executable
reproducers, recorded NumPy inputs, and the compact UI review from 2026-11-22.
Generated reports and repeated images or videos that are not referenced by
those checks or maintained code have been removed from the landing tree.
This is artifact curation; it does not change the tracker or its acceptance
criteria, and it does not resolve the known test failures.

The complete pre-curation snapshot is preserved at
[`4190305f512ca83114ac2d34c4c87f645635af54`](https://github.com/TheWisp/lerobot/tree/4190305f512ca83114ac2d34c4c87f645635af54/docs/proofs).
Links in retained reports point to that exact snapshot when their generated
artifacts are archived. Measurements and screenshots in those reports describe
the dated runs recorded there; they are not fresh integration validation.

To inspect the complete evidence locally without changing a working checkout:

```bash
archive_dir=$(mktemp -d)
git archive 4190305f512ca83114ac2d34c4c87f645635af54 docs/proofs | tar -x -C "$archive_dir"
```

Some proof directories also contain executable checks or required inputs.
For example, `tests/policies/flash/test_native_acceleration_candidates.py`
loads `flash-native-acceleration/candidates.py`, and maintained benchmark
helpers read the point-reacquisition audit and three-point demo-frame data.
Those dependencies remain local. The 2026-11-22 UI review retains its capture
provenance and negative control alongside the images.

The original research branch remains available. The landing stack will be
rebuilt from main using the curated file contents, so it does not introduce
then delete the archived blobs through the research branch's ancestry.
