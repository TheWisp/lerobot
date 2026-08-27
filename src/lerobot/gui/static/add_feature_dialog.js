// Generic "Add feature" dialog — schema-add for non-default custom features.
// `reward` and `success` go through the banner (FeatureEditing.addDefaultsFor)
// so they're rejected here with an in-dialog error.
//
// State / DOM lookups talk to feature_editing.js's globals
// (window.currentDataset, window.datasets, window.FeatureEditing).
(function () {
    "use strict";

    const LOG = "[add-feature-dialog]";
    const _log = (...a) => console.info(LOG, ...a);
    const _err = (...a) => console.error(LOG, ...a);
    _log("module loaded");

    const DTYPE_DEFAULTS = {
        float32: "0",
        int64: "0",
        int8: "0",
        bool: "false",
        string: "",
    };

    // Which fields each kind owns. The dialog asks for a *kind* of column, not
    // a storage dtype -- "Flags" has no dtype to choose, and a number has no
    // flag list -- so showing all of them at once invites a wrong answer.
    //
    // A dropdown rather than a row of buttons: the kinds are a growing list
    // (stored masks are already a fourth in this codebase) and a row does not
    // survive that, while a select does and is what the dialog's other
    // exclusive choices already use.
    const KIND_FIELDS = {
        number: ["add-feature-dtype-row", "add-feature-shape-row", "add-feature-fill-row"],
        text: ["add-feature-fill-row"],
        flags: ["add-feature-flags-row"],
    };
    let currentKind = "number";

    function dialog() { return document.getElementById("add-feature-dialog"); }
    function form() { return document.getElementById("add-feature-form"); }
    function errBox() { return document.getElementById("add-feature-error"); }

    function showError(msg) {
        const e = errBox();
        if (!e) return;
        e.textContent = msg;
        e.hidden = false;
    }

    function clearError() {
        const e = errBox();
        if (e) e.hidden = true;
    }

    function setKind(kind) {
        currentKind = kind in KIND_FIELDS ? kind : "number";
        const shown = new Set(KIND_FIELDS[currentKind]);
        for (const ids of Object.values(KIND_FIELDS)) {
            for (const id of ids) {
                const row = document.getElementById(id);
                if (row) row.hidden = !shown.has(id);
            }
        }
        const picker = document.getElementById("add-feature-kind");
        if (picker && picker.value !== currentKind) picker.value = currentKind;
        clearError();
    }

    function resetForm() {
        const f = form();
        if (!f) return;
        f.reset();
        f.shape.value = "[1]";
        f.fill_value.value = DTYPE_DEFAULTS.float32;
        setKind("number");
        clearError();
    }

    // Auto-default the fill_value when dtype or per_episode changes. Keeps
    // the user from typing 0 and getting a string "0" coerced as int — the
    // dialog's text input has no dtype itself, so this is the only signal.
    function autoUpdateFill() {
        const f = form();
        if (!f) return;
        const dtype = f.dtype.value;
        const isPerEpisodeBool = f.per_episode.checked && dtype === "bool";
        // Per-episode bool defaults to True (the "successful demo by default"
        // pattern for per-episode flags).
        f.fill_value.value = isPerEpisodeBool ? "true" : (DTYPE_DEFAULTS[dtype] ?? "0");
    }

    function parseFillValue(raw, dtype) {
        if (dtype === "bool") {
            const t = raw.trim().toLowerCase();
            if (t === "true" || t === "1") return true;
            if (t === "false" || t === "0" || t === "") return false;
            throw new Error(`bool fill_value must be true / false, got '${raw}'`);
        }
        if (dtype === "string") return raw;
        if (dtype === "float32") {
            const v = parseFloat(raw);
            if (Number.isNaN(v)) throw new Error(`float32 fill_value must be numeric, got '${raw}'`);
            return v;
        }
        const v = parseInt(raw, 10);
        if (Number.isNaN(v)) throw new Error(`${dtype} fill_value must be an integer, got '${raw}'`);
        return v;
    }

    function parseShape(raw) {
        const trimmed = raw.trim();
        if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) {
            throw new Error("Shape must be a JSON list, e.g. [1] or [3, 64, 64]");
        }
        let arr;
        try {
            arr = JSON.parse(trimmed);
        } catch (e) {
            throw new Error(`Shape: invalid JSON — ${e.message}`);
        }
        if (!Array.isArray(arr) || arr.length === 0 || !arr.every(n => Number.isInteger(n) && n > 0)) {
            throw new Error("Shape must be a non-empty list of positive ints");
        }
        return arr;
    }

    async function submit(e) {
        e.preventDefault();
        clearError();
        const f = form();
        const datasetId = window.currentDataset;
        if (!datasetId) {
            showError("No dataset open");
            return;
        }
        const name = f.name.value.trim();
        if (!name) {
            showError("Name is required");
            return;
        }
        if (name === "reward" || name === "success") {
            showError(`'${name}' is a default feature — use the banner above instead.`);
            return;
        }
        let body;
        if (currentKind === "flags") {
            const flags = f.flags.value.split("\n").map((x) => x.trim()).filter(Boolean);
            if (!flags.length) {
                showError("A flags column needs at least one flag");
                return;
            }
            const repeated = flags.filter((x, i) => flags.indexOf(x) !== i);
            if (repeated.length) {
                // Caught here as well as server-side so the operator sees it
                // against the list they just typed, before the confirm dialog.
                showError(`Repeated flag(s): ${[...new Set(repeated)].join(", ")}`);
                return;
            }
            // The bitset contract fixes storage; the operator chose a kind of
            // column, not a dtype, so these are not read from the form.
            body = { name, dtype: "int64", shape: [1], flags,
                     per_episode: f.per_episode.checked, fill_value: 0 };
        } else {
            const dtype = currentKind === "text" ? "string" : f.dtype.value;
            let shape, fillValue;
            try {
                shape = currentKind === "text" ? [1] : parseShape(f.shape.value);
                fillValue = parseFillValue(f.fill_value.value, dtype);
            } catch (err) {
                showError(err.message);
                return;
            }
            body = { name, dtype, shape, per_episode: f.per_episode.checked, fill_value: fillValue };
        }
        // Confirm: this rewrites every parquet shard, can't be undone via
        // Discard. Cheap to keep here even though the dialog has a warning
        // banner — explicit second click before something irreversible.
        const totalEpisodes = window.datasets?.[datasetId]?.total_episodes ?? "?";
        const totalFrames = window.datasets?.[datasetId]?.total_frames ?? "?";
        const what = body.flags
            ? `flags column "${name}" with ${body.flags.length} flag(s): ${body.flags.join(", ")}`
            : `column "${name}" (${body.dtype}[${body.shape.join(",")}]) ` +
              `with initial fill ${JSON.stringify(body.fill_value)}`;
        const ok = window.confirm(
            `Add ${what}\n\nto ${totalFrames} frames across ${totalEpisodes} episodes?\n\n` +
            "This rewrites the dataset's parquet shards in place. " +
            "Cannot be undone via Discard."
        );
        if (!ok) return;
        const submitBtn = document.getElementById("add-feature-submit");
        submitBtn.disabled = true;
        try {
            const r = await fetch(
                `/api/datasets/${encodeURIComponent(datasetId)}/features`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body),
                }
            );
            if (!r.ok) {
                const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
                showError(`Add failed: ${detail}`);
                return;
            }
            const payload = await r.json();
            // Push the new schema into window.datasets so feature_editing
            // re-renders against the post-mutation info.
            if (window.FeatureEditing && window.FeatureEditing.refreshAfterSchemaAdd) {
                window.FeatureEditing.refreshAfterSchemaAdd(datasetId, payload.info);
            } else if (payload.info) {
                window.datasets[datasetId] = payload.info;
            }
            window.setStatus && window.setStatus(
                `Added feature: ${(payload.added || []).join(", ")}`
            );
            dialog().close();
        } catch (err) {
            _err("submit failed", err);
            showError(`Add failed: ${err.message}`);
        } finally {
            submitBtn.disabled = false;
        }
    }

    function open() {
        const d = dialog();
        if (!d) {
            _err("dialog DOM not present");
            return;
        }
        resetForm();
        d.showModal();
    }

    document.addEventListener("DOMContentLoaded", () => {
        const f = form();
        if (!f) {
            _err("form not present at DOMContentLoaded");
            return;
        }
        const kindPicker = document.getElementById("add-feature-kind");
        if (kindPicker) kindPicker.addEventListener("change", () => setKind(kindPicker.value));
        f.dtype.addEventListener("change", autoUpdateFill);
        f.per_episode.addEventListener("change", autoUpdateFill);
        f.addEventListener("submit", submit);
        const cancelBtn = document.getElementById("add-feature-cancel");
        if (cancelBtn) {
            cancelBtn.addEventListener("click", () => dialog().close());
        }
    });

    window.AddFeatureDialog = { open };
}());
