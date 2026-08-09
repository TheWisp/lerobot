// Free-text notes on datasets, training runs, and checkpoints.
//
// One interaction everywhere: the note shows under the row it belongs to,
// clicking it opens an editor in place, Ctrl/Cmd+Enter or blur saves, Esc
// cancels, and clearing the text deletes the note. Rows without a note show a
// pencil on hover instead, so an empty tree stays quiet.
//
// Callers render `notesLine(path)` after their row markup and call
// `notesEnsure(paths, rerender)` once the paths are known; the fetch is
// batched, and `rerender` only fires when something actually arrived.

//: Paths per /bulk request. Keeps the query string well under any server or
//: proxy line limit when a source folder holds hundreds of datasets.
const NOTES_BATCH = 40;

const _noteCache = new Map();   // absolute path -> note text ('' = no note)
const _noteInFlight = new Set(); // paths in a batch that has not returned yet
let _noteEditing = null;        // path currently open in an editor

function noteOf(path) {
    return _noteCache.get(path) || '';
}

function _noteEsc(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
}

/** Fetch notes for any of `paths` not already cached, then call `rerender`. */
async function notesEnsure(paths, rerender) {
    const missing = paths.filter(p => p && !_noteCache.has(p) && !_noteInFlight.has(p));
    if (!missing.length) return;
    // In-flight, not cached: a batch that fails must leave the paths unknown so
    // the next render retries them. Seeding '' here would make one failed
    // request hide every note until a page reload.
    missing.forEach(p => _noteInFlight.add(p));
    try {
        // Chunked so a source folder with hundreds of datasets cannot build a
        // query string past what the server will accept.
        for (let i = 0; i < missing.length; i += NOTES_BATCH) {
            const chunk = missing.slice(i, i + NOTES_BATCH);
            const qs = chunk.map(p => `paths=${encodeURIComponent(p)}`).join('&');
            const res = await fetch(`/api/notes/bulk?${qs}`);
            if (!res.ok) continue;
            const got = await res.json();
            for (const [p, text] of Object.entries(got)) _noteCache.set(p, text);
        }
        if (rerender && !_noteEditing) rerender();
    } catch {
        /* a tree render must not break because notes failed to load */
    } finally {
        missing.forEach(p => _noteInFlight.delete(p));
    }
}

/** Forget cached notes so the next render refetches — e.g. after an SSH edit. */
function notesInvalidate() {
    _noteCache.clear();
    _noteInFlight.clear();
}

/** The note under a row, or '' when there is nothing to show.
 *
 * Truncated to its first line by default — tree rows have one line of space.
 * Pass 'note-block' where the full text fits (a detail panel). */
function notesLine(path, extraClass = '') {
    const note = noteOf(path);
    if (!note) return '';
    const body = extraClass.includes('note-block') ? note : (note.split('\n').find(l => l.trim()) || '');
    return `<div class="note-line ${extraClass}" data-note-path="${_noteEsc(path)}"
        title="${_noteEsc(note)}">${_noteEsc(body)}</div>`;
}

/** The hover affordance that adds a note to a row that has none.
 *
 * Always emits the slot, even when the row already has a note: it sits in a
 * flex row before the meta label, so rendering it conditionally would shift
 * every annotated row's label out of line with its neighbours. */
function notesAddButton(path) {
    if (noteOf(path)) return '<span class="note-add note-add-spacer"></span>';
    return `<span class="note-add" data-note-add="${_noteEsc(path)}" title="Add note">&#9998;</span>`;
}

/** A table cell's contents: the note if there is one, else the add control.
 *
 * Both render inside the cell, so the editor always anchors to the same
 * element — otherwise it lands in the <td> for annotated rows and in the <tr>
 * for un-annotated ones, and visibly jumps between them. */
function notesCell(path) {
    return notesLine(path, 'note-inline') || notesAddButton(path);
}

/** A labelled add-note control for a detail panel (quote-safe attribute). */
function notesAddInline(path) {
    return `<span class="note-add-inline" data-note-add="${_noteEsc(path)}">&#9998; Add note</span>`;
}

/** Open the editor for `path`, anchored under `anchorEl`. */
function notesEdit(path, anchorEl) {
    if (_noteEditing) notesCloseEditor();
    _noteEditing = path;

    const box = document.createElement('div');
    box.className = 'note-editor';
    box.innerHTML = `<textarea spellcheck="false" rows="1"></textarea>
        <div class="note-editor-actions">
            <button type="button" class="note-btn note-btn-save">Save</button>
            <button type="button" class="note-btn">Cancel</button>
            <span class="note-editor-hint">Enter saves &middot; Shift+Enter new line</span>
        </div>`;
    anchorEl.insertAdjacentElement('afterend', box);
    // Editing replaces the note in place rather than stacking under it, or the
    // truncated line and the full text are both on screen saying the same thing.
    if (anchorEl.classList.contains('note-line')) anchorEl.style.display = 'none';

    const ta = box.querySelector('textarea');
    const [saveBtn, cancelBtn] = box.querySelectorAll('.note-btn');
    let cancelled = false;

    const grow = () => {
        ta.style.height = 'auto';
        ta.style.height = Math.min(180, Math.max(26, ta.scrollHeight)) + 'px';
    };
    ta.value = noteOf(path);
    grow();
    ta.focus();
    ta.setSelectionRange(ta.value.length, ta.value.length);

    const commit = () => {
        const value = ta.value;
        notesCloseEditor();
        if (!cancelled) notesSave(path, value);
    };

    ta.addEventListener('input', grow);
    ta.addEventListener('keydown', (ev) => {
        ev.stopPropagation();   // the tree owns arrow keys / Delete; the editor wins here
        if (ev.key === 'Escape') {
            cancelled = true;
            commit();
        } else if (ev.key === 'Enter' && !ev.shiftKey) {
            // Notes are a line or two, not an essay — Enter is the common case
            // and Shift+Enter is the escape hatch for the rare multi-line one.
            ev.preventDefault();
            commit();
        }
    });
    // preventDefault on mousedown keeps focus in the textarea, so the button's
    // own click decides the outcome instead of a blur racing it.
    for (const btn of [saveBtn, cancelBtn]) {
        btn.addEventListener('mousedown', (ev) => ev.preventDefault());
    }
    saveBtn.addEventListener('click', commit);
    cancelBtn.addEventListener('click', () => { cancelled = true; commit(); });
    // Clicking anywhere else saves rather than discarding what was typed.
    ta.addEventListener('blur', (ev) => {
        if (box.contains(ev.relatedTarget)) return;
        commit();
    });
}

function notesCloseEditor() {
    document.querySelectorAll('.note-editor').forEach(el => {
        const hidden = el.previousElementSibling;
        if (hidden && hidden.classList.contains('note-line')) hidden.style.display = '';
        el.remove();
    });
    _noteEditing = null;
}

/** Persist a note. Empty text deletes it. Re-renders whatever owns the row. */
async function notesSave(path, text) {
    const previous = noteOf(path);
    if (text.trim() === previous) return;
    try {
        const res = await fetch(`/api/notes?path=${encodeURIComponent(path)}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ note: text }),
        });
        if (!res.ok) throw new Error(await res.text());
        _noteCache.set(path, (await res.json()).note);
    } catch (e) {
        _noteCache.set(path, previous);
        if (typeof showToast === 'function') showToast('Error', `Could not save note: ${e.message}`, 'error');
    }
    notesRerender();
}

/** Re-render every surface that shows notes. Set by the tabs that use them. */
const _noteRenderers = [];
function notesOnRerender(fn) { _noteRenderers.push(fn); }
function notesRerender() {
    // Never re-render out from under someone who is typing: the rerender
    // replaces innerHTML, which would drop the editor and their text.
    if (_noteEditing) return;
    _noteRenderers.forEach(fn => { try { fn(); } catch { /* keep going */ } });
}

// One delegated listener for every note surface: trees, panels, tables.
document.addEventListener('click', (ev) => {
    const add = ev.target.closest('[data-note-add]');
    if (add) {
        ev.stopPropagation();   // don't also open the dataset / select the run
        notesEdit(add.dataset.noteAdd, add.closest('.model-ckpt-note, .source-dataset, .tree-header, .model-note-anchor') || add);
        return;
    }
    const line = ev.target.closest('[data-note-path]');
    if (line) {
        ev.stopPropagation();
        notesEdit(line.dataset.notePath, line);
    }
}, true);
