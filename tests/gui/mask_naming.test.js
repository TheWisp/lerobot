// The client must derive the same mask column name the writer used, or it asks
// for a column the dataset does not have, gets nothing back, and draws no
// masks — which reads as "this episode was never segmented" rather than as a
// naming bug. So the expectations come from the PYTHON helpers
// (mask_naming_fixture.json) and are asserted here against the JS pair.
//
// Regenerate with tests/gui/regen_mask_naming_fixture.py after changing the
// naming; a hand-edited fixture would let the two drift apart silently.

const fs = require('fs');
const path = require('path');
const assert = require('assert');

const src = fs.readFileSync(
    path.join(__dirname, '..', '..', 'src', 'lerobot', 'gui', 'static', 'masks.js'),
    'utf8',
);
global.window = {};
new Function(src)();
const { _maskKeyFor, _camKeyFor } = global.window.MaskOverlay;

const fixture = JSON.parse(
    fs.readFileSync(path.join(__dirname, 'mask_naming_fixture.json'), 'utf8'),
);

assert.ok(fixture.cases.length > 0, 'empty fixture would make every assertion vacuous');

for (const c of fixture.cases) {
    assert.strictEqual(
        _maskKeyFor(c.camera),
        c.mask_key,
        `_maskKeyFor(${JSON.stringify(c.camera)}) disagrees with mask_feature_of`,
    );
    assert.strictEqual(
        _camKeyFor(c.mask_key, c.camera_keys),
        c.camera_roundtrip,
        `_camKeyFor(${JSON.stringify(c.mask_key)}) disagrees with camera_feature_of`,
    );
}

// A key that names no mask column must come back untouched, so a caller can
// pass any feature key without first testing it.
for (const p of fixture.inverse_passthrough) {
    assert.strictEqual(
        _camKeyFor(p.key, []),
        p.expected,
        `_camKeyFor(${JSON.stringify(p.key)}) should pass through unchanged`,
    );
}

// The complement: a fixture that only ever round-tripped would pass against a
// pair of identity functions. At least one case must actually change the key.
assert.ok(
    fixture.cases.some((c) => c.mask_key !== c.camera),
    'no case changes the key, so identity functions would satisfy this file',
);

console.log(`mask naming: ${fixture.cases.length} cases + ${fixture.inverse_passthrough.length} passthrough OK`);
