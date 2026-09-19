// What a segmentation step's prompt rows start from when the panel leaves
// saved-effects mode. Pure, and its own module so the decision can be tested
// under node (see tests/gui/mask_seed.test.js) rather than through a browser,
// a dataset and a model load. Loaded as a plain <script> (exposes
// window.MaskSeed) and as a CommonJS module in the test.
(function (root, factory) {
    if (typeof module !== 'undefined' && module.exports) module.exports = factory();
    else root.MaskSeed = factory();
})(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    var BLANK = { key: 'none', params: {} };

    function named(objects) {
        return (objects || []).filter(function (o) { return o && String(o.name || '').trim(); });
    }

    /**
     * Choose the rows a segmentation step opens with.
     *
     * `prev` is what the operator had typed before saved-effects mode took the
     * rows; `recipe` is the dataset's saved mask recipe ({labels, treatments,
     * background}) or null.
     *
     * The rule, in order:
     *   1. The operator's own prompts win. Seeding over them would throw away
     *      typing, which is worse than any convenience it buys.
     *   2. Otherwise, if the dataset already names what is in this scene, start
     *      from that vocabulary — re-running a segmenter on a masked episode
     *      almost always means "those objects again", and retyping five
     *      specific names is the cost of not doing this.
     *   3. Otherwise, one blank row.
     *
     * The background comes with the vocabulary in case 2 on purpose: a save
     * writes the panel's background treatment, so opening at `none` over a
     * dataset saved with `blur` would quietly downgrade it on the next save.
     *
     * Returns a fresh {objects, background} — never aliases its inputs, so a
     * later edit to the rows cannot reach back into the saved recipe.
     */
    function seedForStep(prev, recipe) {
        var own = named(prev);
        if (own.length) {
            return {
                objects: (prev || []).map(function (o) { return Object.assign({}, o); }),
                background: Object.assign({}, (prev && prev.background) || BLANK),
                source: 'operator',
            };
        }
        var labels = (recipe && recipe.labels) || [];
        if (labels.length) {
            var treatments = (recipe && recipe.treatments) || {};
            return {
                objects: labels.map(function (name) {
                    return {
                        name: name,
                        sign: '+',
                        treatment: Object.assign({}, treatments[name] || BLANK),
                    };
                }),
                background: Object.assign({}, (recipe && recipe.background) || BLANK),
                source: 'saved',
            };
        }
        return {
            objects: [{ name: '', sign: '+', treatment: Object.assign({}, BLANK) }],
            background: Object.assign({}, BLANK),
            source: 'blank',
        };
    }

    return { seedForStep: seedForStep };
});
