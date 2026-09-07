// Bit maths without bitwise operators.
//
// JavaScript's &, | and ~ coerce to *32-bit* integers, so `value &
// Math.pow(2, 40)` is 0 and every flag past bit 30 would be silently invisible
// and untickable -- while the stored contract allows 63. Division and modulo
// stay exact to 2^53, which is also where JSON stops carrying integers
// faithfully, so this is as far as the browser can go regardless.
//
// Its own module because three files now need it and each temptation to reach
// for `>> bit & 1` instead has cost a real defect: a mask lane past 31 drew
// from another lane's bit, and a flag edit at bit 35 wrapped onto bit 3.
//
// Loaded as a plain <script> (exposes window.Bitset) and as a CommonJS module
// in the node tests.
(function (root, factory) {
    if (typeof module !== "undefined" && module.exports) module.exports = factory();
    else root.Bitset = factory();
})(typeof self !== "undefined" ? self : this, function () {
    "use strict";

    const MAX_JS_BIT = 52; // Number.MAX_SAFE_INTEGER is 2^53 - 1

    /** Whether `bit` is set in `value`. Bits above the safe range read false. */
    function bitIsSet(value, bit) {
        if (bit > MAX_JS_BIT) return false;
        return Math.floor(Math.round(value) / Math.pow(2, bit)) % 2 === 1;
    }

    /** The bit positions set in `mask`, ascending. */
    function bitsOfMask(mask) {
        const bits = [];
        for (let b = 0; b <= MAX_JS_BIT; b++) {
            if (Math.pow(2, b) > mask) break;
            if (bitIsSet(mask, b)) bits.push(b);
        }
        return bits;
    }

    /** `value` with `setMask`'s bits set and `clearMask`'s cleared. */
    function withBits(value, setMask, clearMask) {
        let v = Math.round(value);
        for (const b of bitsOfMask(setMask)) if (!bitIsSet(v, b)) v += Math.pow(2, b);
        for (const b of bitsOfMask(clearMask)) if (bitIsSet(v, b)) v -= Math.pow(2, b);
        return v;
    }

    return { MAX_JS_BIT, bitIsSet, bitsOfMask, withBits };
});
