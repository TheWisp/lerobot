// Confirm / alert / prompt dialogs that belong to this application.
//
// The GUI asked its most consequential questions -- delete this dataset, stop
// this training run, reassign a port a robot is holding -- through
// `window.confirm`. That draws the operating system's own dialog: a different
// typeface, a different palette, a title bar naming the origin ("127.0.0.1:8000
// says"), and a position the page does not control. It also reads as
// browser chrome rather than as part of the app, which is exactly the wrong
// signal for a question about the user's data.
//
// This is built on the native <dialog> element and `showModal()`, which is what
// buys back everything `window.confirm` gave for free: the top layer (so a
// dialog can never be trapped beneath a stacking context), a real backdrop, a
// focus trap, Escape-to-dismiss, and the rest of the page marked inert to
// assistive technology. A hand-rolled <div> overlay has none of those, and that
// is the usual way a "nicer" dialog ends up less usable than the one it
// replaced.
//
// The one thing that cannot be preserved is the blocking call. `window.confirm`
// returns a boolean; these return a promise, so every caller awaits.
//
// That difference is not only about control flow. `window.confirm` froze the
// event loop, so while it was up the page received nothing. A modal <dialog> is
// inert to focus and hit-testing, but the events raised inside it still travel
// the whole tree -- and this app keeps its shortcuts on `document`. Left alone,
// Space played the episode behind a delete confirm (and its preventDefault
// stopped the focused button from answering), Delete staged an episode
// deletion, Escape cleared the timeline selection the question was about, and
// answering a confirm raised from the Transfers tray closed the tray.
//
// See `standDownWhileOpen` below for how that is put back.
(function () {
    "use strict";

    // ── Standing the page down while a dialog is up ───────────────────────
    //
    // Stopping these events at the dialog's own boundary is the obvious fix and
    // it is not enough: a `document` listener registered in the CAPTURE phase
    // runs before the event reaches the dialog at all, and this app has five of
    // those. So the guard goes where phase cannot route around it -- around the
    // listener itself. `document` and `window` are the only ancestors the page
    // attaches to, so wrapping registration on both covers every page handler,
    // including ones not yet written.
    //
    // WHICH events stand down is a rule, not a list of type names. A list of
    // names fails open: the day someone listens for a type nobody thought of,
    // the leak is back and silent. The DOM already classifies user input by
    // interface, so that is what gets asked.
    const isUserInput = (event) =>
        event instanceof KeyboardEvent ||
        event instanceof MouseEvent ||          // PointerEvent extends this
        event instanceof WheelEvent ||
        (typeof TouchEvent !== "undefined" && event instanceof TouchEvent);

    // The single carve-out, and it is shaped so that forgetting something can
    // only over-protect rather than leak: a pointer gesture already in flight
    // when the dialog appeared still has to be able to END. A drag whose mouseup
    // never arrives never stops dragging. Movement, release, and the hover
    // events that pair with them are what finish one; everything else -- press,
    // click, wheel, context menu -- STARTS something and is gated.
    //
    // Matched on the suffix rather than on known names, so `pointerup`,
    // `touchend` and whatever the platform adds next are classified without
    // anyone editing this.
    const FINISHES_A_GESTURE = /(?:move|up|end|out|leave|over|enter)$/;

    // The keyboard has no in-flight gesture to protect, so all of it stands
    // down: a dialog owns the keyboard for as long as it is up.
    const shouldStandDown = (event) =>
        isUserInput(event) &&
        (event instanceof KeyboardEvent || !FINISHES_A_GESTURE.test(event.type));

    // Counted, not a boolean: a question can be raised while another is up, and
    // the page must stay down until the last one is gone.
    let openDialogs = 0;

    function standDownWhileOpen(target) {
        const add = target.addEventListener.bind(target);
        const remove = target.removeEventListener.bind(target);
        // listener -> (type|phase -> wrapper). Callers remove with the function
        // they passed, which is not the function the browser holds, so the
        // mapping has to be kept or every removal silently does nothing -- and
        // the seam drags remove their own document handlers. Weak, so a handler
        // that goes out of scope stays collectable.
        const wrapped = new WeakMap();
        const keyOf = (type, opts) =>
            `${type}|${typeof opts === "object" && opts ? !!opts.capture : !!opts}`;

        target.addEventListener = function (type, listener, opts) {
            if (typeof listener !== "function") return add(type, listener, opts);
            const key = keyOf(type, opts);
            let byKey = wrapped.get(listener);
            if (!byKey) wrapped.set(listener, (byKey = new Map()));

            // `once` is honoured here rather than by the browser: a suppressed
            // call must not spend the single use, or a dialog open at the wrong
            // moment would consume the listener without ever running it.
            const once = typeof opts === "object" && opts ? !!opts.once : false;
            const wrapper = function (event) {
                if (openDialogs > 0 && shouldStandDown(event)) return undefined;
                if (once) target.removeEventListener(type, listener, opts);
                return listener.call(this, event);
            };
            byKey.set(key, wrapper);
            return add(type, wrapper, once ? { ...opts, once: false } : opts);
        };

        target.removeEventListener = function (type, listener, opts) {
            const wrapper = wrapped.get(listener)?.get(keyOf(type, opts));
            return remove(type, wrapper || listener, opts);
        };
    }

    standDownWhileOpen(document);
    standDownWhileOpen(window);

    // Rendering the message with textContent, never innerHTML: most of these
    // strings interpolate a repo id, a file path, or a server error message.
    function setText(el, text) {
        el.textContent = text == null ? "" : String(text);
    }

    // Ids have to be unique per dialog: `aria-labelledby` resolves by id, and
    // two dialogs can be open at once (a question raised while one is up stacks
    // in the top layer). A shared id would point both at whichever came first.
    let seq = 0;

    // `role` follows the ARIA authoring practices: a dialog that interrupts to
    // report or to require a decision is an alertdialog, which is announced
    // immediately; a dialog that collects input is a plain dialog.
    function build({ kind, title, message, confirmLabel, cancelLabel, danger, defaultValue, placeholder }) {
        const uid = `app-dialog-${++seq}`;
        const dlg = document.createElement("dialog");
        dlg.className = "app-dialog" + (danger ? " danger" : "");
        dlg.setAttribute("role", kind === "prompt" ? "dialog" : "alertdialog");

        // All padding lives on the form, so a click whose target is the
        // <dialog> itself is unambiguously a click on the backdrop.
        const form = document.createElement("form");
        form.method = "dialog";
        form.className = "app-dialog-form";

        if (title) {
            const h = document.createElement("h3");
            h.className = "app-dialog-title";
            h.id = `${uid}-title`;
            setText(h, title);
            form.appendChild(h);
            dlg.setAttribute("aria-labelledby", h.id);
        }

        const body = document.createElement("p");
        body.className = "app-dialog-message";
        body.id = `${uid}-message`;
        setText(body, message);
        form.appendChild(body);
        dlg.setAttribute("aria-describedby", body.id);
        if (!title) dlg.setAttribute("aria-labelledby", body.id);

        let input = null;
        if (kind === "prompt") {
            input = document.createElement("input");
            input.type = "text";
            input.className = "app-dialog-input";
            input.value = defaultValue == null ? "" : String(defaultValue);
            if (placeholder) input.placeholder = placeholder;
            form.appendChild(input);
        }

        const actions = document.createElement("menu");
        actions.className = "app-dialog-actions";

        let cancel = null;
        if (kind !== "alert") {
            cancel = document.createElement("button");
            cancel.type = "submit";
            cancel.value = "cancel";
            cancel.className = "app-dialog-btn";
            setText(cancel, cancelLabel || "Cancel");
            actions.appendChild(cancel);
        }

        const ok = document.createElement("button");
        ok.type = "submit";
        ok.value = "confirm";
        ok.className = "app-dialog-btn primary" + (danger ? " danger" : "");
        setText(ok, confirmLabel || "OK");
        actions.appendChild(ok);

        form.appendChild(actions);
        dlg.appendChild(form);
        return { dlg, input, ok, cancel };
    }

    function open(spec) {
        const { dlg, input, ok, cancel } = build(spec);
        document.body.appendChild(dlg);

        return new Promise((resolve) => {
            let settled = false;
            const finish = (accepted) => {
                if (settled) return;
                settled = true;
                openDialogs -= 1;
                const value = input ? input.value : null;
                dlg.remove();
                resolve({ accepted, value });
            };

            // `method="dialog"` closes on submit and reports which button did
            // it, so Escape (returnValue "") and the backdrop fall through to
            // the same cancel path without a separate code path each.
            dlg.addEventListener("close", () => finish(dlg.returnValue === "confirm"));

            // Dismiss on the backdrop, but only when the press BEGAN there. A
            // `click` is delivered to the nearest common ancestor of its
            // mousedown and mouseup, so dragging a text selection out of the
            // field and releasing past the panel edge reports the <dialog>
            // itself and would otherwise read as a backdrop click -- throwing
            // away what the user had just typed.
            let pressedOnBackdrop = false;
            dlg.addEventListener("mousedown", (e) => { pressedOnBackdrop = e.target === dlg; });
            dlg.addEventListener("click", (e) => {
                if (e.target === dlg && pressedOnBackdrop) dlg.close("cancel");
            });

            if (input) {
                // Implicit form submission would pick the first submit button,
                // which is Cancel -- so Enter in the field is bound explicitly
                // to the affirmative action instead.
                input.addEventListener("keydown", (e) => {
                    // `isComposing` (and the legacy 229) exclude the Enter that
                    // commits an IME candidate. Without it, typing a name with a
                    // Japanese/Chinese/Korean IME and pressing Enter to accept
                    // the candidate submits the pre-composition buffer instead,
                    // and preventDefault swallows the commit.
                    if (e.key === "Enter" && !e.isComposing && e.keyCode !== 229) {
                        e.preventDefault();
                        dlg.close("confirm");
                    }
                });
            }

            openDialogs += 1;
            dlg.showModal();
            // Initial focus goes to the field being filled in, and otherwise to
            // the least destructive action -- so holding Enter on a destructive
            // confirm cannot carry the deletion through.
            if (input) input.select();
            else if (spec.danger && cancel) cancel.focus();
            else ok.focus();
        });
    }

    const Dialogs = {
        /** Ask a yes/no question. Resolves true only if the user accepts.
         *
         * Escape, the backdrop, and the cancel button all resolve false, so a
         * caller may treat anything other than true as "do not proceed".
         */
        async confirm(message, opts = {}) {
            const r = await open({ kind: "confirm", message, ...opts });
            return r.accepted;
        },

        /** State something and wait for acknowledgement. Resolves when dismissed. */
        async alert(message, opts = {}) {
            await open({ kind: "alert", message, ...opts });
        },

        /** Ask for a line of text. Resolves the string, or null if cancelled.
         *
         * Matches `window.prompt`: an empty string is a real answer and is
         * returned as such; only cancelling gives null.
         */
        async prompt(message, defaultValue = "", opts = {}) {
            const r = await open({ kind: "prompt", message, defaultValue, ...opts });
            return r.accepted ? r.value : null;
        },

        /** Whether a dialog is up, and the page therefore stood down. */
        isOpen() {
            return openDialogs > 0;
        },
    };

    window.Dialogs = Dialogs;
})();
