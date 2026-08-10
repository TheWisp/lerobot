// How a run's training image is described in the run detail card.
//
// A tag is not an identity: `lerobot-training:dev-local` is rebuilt whenever
// the trainer changes, so runs recording only the tag cannot be told apart.
// The orchestrator resolves the tag, build date and git revision on the host
// at launch and stores them on the run; this renders them.
//
// Isolated so it is unit-testable in node. Loaded as a plain <script>
// (exposes window.TrainingImageIdentity) and as a CommonJS module in the test.
(function (root, factory) {
    if (typeof module !== 'undefined' && module.exports) module.exports = factory();
    else root.TrainingImageIdentity = factory();
})(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    // Minute precision: the image is rebuilt several times a day, and a bare
    // date would collapse those rebuilds back into looking identical.
    // An unparsable value passes through — the raw string beats an empty cell.
    function formatBuiltAt(iso) {
        if (!iso) return '';
        var d = new Date(iso);
        if (isNaN(d.getTime())) return String(iso);
        var pad = function (n) { return String(n).padStart(2, '0'); };
        return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) +
            ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
    }

    // Parts are appended only when present, so a run predating the recording
    // renders as the bare tag — indistinguishable from the old output.
    function text(args) {
        args = args || {};
        // `__image_resolved__` first: a run that did not override the image has
        // no `__image__` and read "(default)", which the moving pin makes a lie.
        var parts = [args['__image_resolved__'] || args['__image__'] || '(default)'];
        var built = formatBuiltAt(args['__image_created__']);
        if (built) parts.push('built ' + built);
        var revision = args['__image_revision__'];
        if (revision) parts.push(String(revision).slice(0, 7));
        return parts.join(' · ');
    }

    return { text: text, formatBuiltAt: formatBuiltAt };
});
