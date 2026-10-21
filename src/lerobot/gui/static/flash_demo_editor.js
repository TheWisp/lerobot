/* FLASH demo preparation editor. The Run tab owns replay selection and subprocess status. */

function _flashEditorField(parent, labelText, value, onChange, options = null, type = 'text') {
    const label = document.createElement('label');
    label.textContent = labelText;
    let control;
    if (options) {
        control = document.createElement('select');
        const choices = options.includes(String(value)) ? options : [String(value), ...options];
        for (const optionValue of choices) {
            const option = document.createElement('option');
            option.value = optionValue;
            option.textContent = optionValue;
            control.append(option);
        }
    } else {
        control = document.createElement('input');
        control.type = type;
        if (type === 'number') control.step = 'any';
    }
    control.setAttribute('aria-label', labelText);
    control.value = String(value ?? '');
    control.addEventListener('change', () => {
        onChange(type === 'number' ? (control.value === '' ? null : Number(control.value)) : control.value);
        _flashEditorUpdateStatus();
        _flashRenderTimeline();
    });
    label.append(control);
    parent.append(label);
    return control;
}

function _flashEditorSection(parent, title, description = '') {
    const section = document.createElement('section');
    const heading = document.createElement('strong');
    heading.textContent = title;
    section.append(heading);
    if (description) {
        const note = document.createElement('p');
        note.textContent = description;
        section.append(note);
    }
    parent.append(section);
    return section;
}

function _flashEditorButton(parent, label, action) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.onclick = () => {
        action();
        _flashEditorUpdateStatus();
        _flashRenderTimeline();
    };
    parent.append(button);
    return button;
}

function _flashEditorCurrentFrame() {
    return flashReplay.frames[Number(document.getElementById('flash-tracker-scrub').value)];
}

function _flashEditorJump(frameIndex) {
    const nearest = flashReplay.frames.reduce((best, frame, index) =>
        Math.abs(frame.frame_index - frameIndex) < Math.abs(flashReplay.frames[best].frame_index - frameIndex)
            ? index : best, 0);
    seekFlashReplay(nearest);
}

function _flashEditorSetReferenceFrame(frameIndex) {
    const annotation = flashDemoEditor.annotation;
    if (Number(annotation.reference_frame) !== Number(frameIndex)) {
        annotation.reference_frame = Number(frameIndex);
        delete annotation.reference_polygons_by_role;
        flashDemoEditor.message = 'Reference frame changed; draw new polygons if the automatic mask needs correction.';
    }
    _flashEditorRender();
    _flashRenderTimeline();
}

function _flashEditorUpdateStatus() {
    if (!flashDemoEditor) return;
    const changed = JSON.stringify([flashDemoEditor.annotation, flashDemoEditor.keypointsDocument]) !== flashDemoEditor.original;
    const status = document.getElementById('flash-demo-editor-change-status');
    if (status) status.textContent = flashDemoEditor.raw
        ? changed ? 'Unsaved setup · prepare to review a new version' : 'Raw demo · no preparation saved'
        : changed ? 'Unsaved edits · regenerate to review a new version' : 'Viewing saved preparation';
}

function _flashEditorUpdateFrame() {
    if (!flashDemoEditor) return;
    const current = _flashEditorCurrentFrame();
    const annotation = flashDemoEditor.annotation;
    const isReference = current.frame_index === Number(annotation.reference_frame);
    const role = flashDemoEditor.role;
    const mask = document.querySelector('#rerun-viewer .flash-demo-mask');
    const svg = document.querySelector('#rerun-viewer .flash-demo-polygon');
    if (!svg || !mask) return;
    const saved = annotation.reference_polygons_by_role?.[role];
    const points = flashDemoEditor.drawing ? flashDemoEditor.vertices : saved;
    const width = current.snapshot.camera.width;
    const height = current.snapshot.camera.height;
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.replaceChildren();
    if (isReference && points?.length) {
        const shape = document.createElementNS('http://www.w3.org/2000/svg', points.length >= 3 ? 'polygon' : 'polyline');
        shape.setAttribute('points', points.map(point => point.join(',')).join(' '));
        shape.setAttribute('fill', points.length >= 3 ? '#16c5ff44' : 'none');
        shape.setAttribute('stroke', '#16c5ff');
        shape.setAttribute('stroke-width', '2');
        svg.append(shape);
        for (const [x, y] of points) {
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', x); circle.setAttribute('cy', y);
            circle.setAttribute('r', '3'); circle.setAttribute('fill', '#fff');
            svg.append(circle);
        }
    }
    svg.classList.toggle('is-drawing', !!flashDemoEditor.drawing && isReference);
    mask.hidden = !isReference || !!saved?.length || !flashDemoEditor.referenceMasks[role]?.available ||
        Number(flashDemoEditor.referenceMasks[role]?.frame_index) !== Number(annotation.reference_frame);
    const frameText = document.getElementById('flash-demo-editor-frame');
    if (frameText) frameText.textContent = `Frame ${current.frame_index} · ${current.snapshot.demo.time_s.toFixed(2)} s`;
    const polygonText = document.getElementById('flash-demo-editor-polygon-status');
    if (polygonText) polygonText.textContent = flashDemoEditor.drawing
        ? `${flashDemoEditor.vertices.length} vertices; finish after at least 3`
        : saved?.length ? `Manual ${role} mask: ${saved.length} vertices` :
            flashDemoEditor.raw ? 'No prepared mask yet; the text prompt will create one when prepared.' :
                `Prepared ${role} mask shown on its reference frame`;
}

function _flashEditorMaskRole(role) {
    flashDemoEditor.role = role;
    flashDemoEditor.drawing = false;
    flashDemoEditor.vertices = [];
    const mask = document.querySelector('#rerun-viewer .flash-demo-mask');
    if (flashReplay.bundle_path) {
        mask.src = `/api/run/flash/demo/reference-mask?${new URLSearchParams({
            dataset_root: flashReplay.datasetRoot, bundle_path: flashReplay.bundle_path, role,
        })}`;
    } else {
        mask.removeAttribute('src');
    }
    _flashEditorUpdateFrame();
}

function _flashEditorPolygonClick(event) {
    if (!flashDemoEditor?.drawing) return;
    const current = _flashEditorCurrentFrame();
    if (current.frame_index !== Number(flashDemoEditor.annotation.reference_frame)) return;
    const image = document.querySelector('#rerun-viewer .flash-replay-grid img');
    const rect = image.getBoundingClientRect();
    const width = current.snapshot.camera.width;
    const height = current.snapshot.camera.height;
    const scale = Math.min(rect.width / width, rect.height / height);
    const left = rect.left + (rect.width - width * scale) / 2;
    const top = rect.top + (rect.height - height * scale) / 2;
    const x = Math.round((event.clientX - left) / scale);
    const y = Math.round((event.clientY - top) / scale);
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    flashDemoEditor.vertices.push([x, y]);
    _flashEditorUpdateFrame();
}

function _flashEditorRender() {
    const editor = document.getElementById('flash-demo-editor');
    const trackerPanel = document.getElementById('flash-tracker-panel');
    editor.replaceChildren();
    const state = flashDemoEditor;
    const a = state.annotation;
    const roles = Object.keys(a.prompts_by_role);
    const heading = document.createElement('h3');
    heading.textContent = 'Demo preparation';
    editor.append(heading);
    const summary = document.createElement('p');
    summary.textContent = state.raw
        ? 'This recording has no FLASH objects yet. Name the objects, mark the action, then prepare a reviewable candidate.'
        : 'Object names come from this demo’s saved preparation. Choose an object, event or key frame to inspect; regeneration creates a new version.';
    editor.append(summary);
    const changed = document.createElement('p');
    changed.id = 'flash-demo-editor-change-status';
    editor.append(changed);
    const report = state.preparationReport;
    if (report) {
        if (report.review_status === 'candidate') {
            const candidate = document.createElement('p');
            candidate.textContent = 'Review candidate · inspect its masks and poses before using it for a live trial.';
            editor.append(candidate);
        }
        const evidence = report.pose_evidence_frames;
        const note = document.createElement('p');
        note.textContent = evidence
            ? Object.entries(evidence).map(([role, counts]) =>
                `${role}: ${counts.observed || 0} observed, ${counts.predicted || 0} predicted, ${counts.lost || 0} lost`
            ).join(' · ')
            : 'This version has no per-source evidence count; inspect its pose labels on the timeline.';
        editor.append(note);
    }
    const frameText = document.createElement('p');
    frameText.id = 'flash-demo-editor-frame';
    editor.append(frameText);

    const tabs = document.createElement('nav');
    tabs.className = 'flash-editor-tabs';
    tabs.setAttribute('aria-label', 'Demo editor sections');
    const panes = {};
    for (const [key, label] of [
        ['setup', 'Setup'], ['objects', 'Objects'], ['events', 'Events'], ['keyframes', 'Key frames'],
        ['pose', 'Pose evidence'], ['trackers', 'Trackers'], ['advanced', 'Advanced'],
    ]) {
        const button = _flashEditorButton(tabs, label, () => {
            state.section = key; _flashEditorRender();
        });
        button.classList.toggle('active', state.section === key);
        button.setAttribute('aria-current', state.section === key ? 'page' : 'false');
        const pane = document.createElement('div');
        pane.className = 'flash-editor-pane';
        pane.hidden = state.section !== key;
        panes[key] = pane;
    }
    editor.append(tabs);
    for (const pane of Object.values(panes)) editor.append(pane);

    const setup = _flashEditorSection(panes.setup, 'Recording and calibration', state.raw
        ? 'The raw frames contain no FLASH object names or camera-to-arm calibration. Supply the calibration used when this demo was recorded.'
        : 'These capture settings were saved with the selected preparation.');
    _flashEditorField(setup, 'Dataset identity', a.repo_id, value => { a.repo_id = value; });
    _flashEditorField(setup, 'Arm side', a.robot_arm, value => { a.robot_arm = value; }, ['', 'left', 'right']);
    _flashEditorField(setup, 'Robot/camera calibration file', a.calibration_path, value => { a.calibration_path = value; });

    const pose = _flashEditorSection(panes.pose, 'Pose evidence', state.raw
        ? 'No object pose exists yet. Preparation will extract a first pose trace from the RGB-D recording.'
        : 'Hover an item to highlight its projection in the image. This is the pose saved in the selected prepared version.');
    if (trackerPanel) pose.append(trackerPanel);

    const trackers = _flashEditorSection(panes.trackers, 'Compare pose trackers', state.raw
        ? 'Prepare object names, references and calibration before comparing tracker poses.'
        : 'Choose a tracker to process all annotated objects. You can limit processing to one object for diagnosis. Preview the pink candidate alongside the saved pose. Apply replaces the available poses in a new demo version; missing frames keep the saved pose.');
    if (!state.raw) {
        const trackerChoices = (state.trackers?.trackers || []).map(item => item.id);
        const trackerSelect = _flashEditorField(trackers, 'Tracker', state.tracker, value => {
            state.tracker = value; _flashEditorRender();
        }, trackerChoices);
        for (const option of trackerSelect.options) {
            option.textContent = state.trackers.trackers.find(item => item.id === option.value)?.name || option.value;
        }
        trackerSelect.disabled = state.running;
        const selectedTracker = state.trackers?.trackers?.find(item => item.id === state.tracker);
        const trackerName = selectedTracker?.name;
        if (trackerName) {
            const identity = document.createElement('p');
            identity.textContent = `${trackerName} · ${selectedTracker.runnable ? 'local runner ready' : 'runner not installed; import output available'}`;
            trackers.append(identity);
        }
        const roleSelect = _flashEditorField(trackers, 'Processing scope', state.trackerRole,
            value => { state.trackerRole = value; }, ['', ...roles]);
        roleSelect.options[0].textContent = 'All annotated objects';
        roleSelect.disabled = state.running;
        if (state.tracker === 'trackeverything' && selectedTracker?.supports_point_growth !== false) {
            const growth = _flashEditorSection(trackers, 'Visible-surface point growth',
                'Seed additional TrackEverything queries in later SAM3 masks where aligned depth shows uncovered surface. The saved pose fit still uses the initial reference points; inspect the new visible tracks before treating them as pose evidence.');
            const toggle = document.createElement('input');
            toggle.type = 'checkbox';
            toggle.checked = state.growPoints;
            toggle.disabled = state.running;
            toggle.setAttribute('aria-label', 'Grow points from later views');
            toggle.onchange = () => { state.growPoints = toggle.checked; _flashEditorRender(); };
            const label = document.createElement('label');
            label.textContent = 'Grow points from later views';
            label.append(toggle);
            growth.append(label);
            if (state.growPoints) {
                const total = _flashEditorField(growth, 'Maximum points per object', state.maxPoints,
                    value => { state.maxPoints = value; }, null, 'number');
                total.min = String(selectedTracker.initial_points_per_object || 3);
                total.step = '1';
                const added = _flashEditorField(growth, 'Maximum new points per view', state.pointsPerView,
                    value => { state.pointsPerView = value; }, null, 'number');
                added.min = '3';
                added.step = '1';
                const tolerance = _flashEditorField(growth, 'Visible depth agreement (mm)', state.surfaceToleranceMm,
                    value => { state.surfaceToleranceMm = value; }, null, 'number');
                tolerance.min = '0.1';
                tolerance.step = '0.1';
                for (const input of [total, added, tolerance]) input.disabled = state.running;
            }
        }
        const processButton = _flashEditorButton(trackers, 'Process demo and preview',
            () => void _flashEditorProcessTracker());
        processButton.disabled = state.running || !selectedTracker?.runnable;
        const importDetails = document.createElement('details');
        const importSummary = document.createElement('summary');
        importSummary.textContent = 'Import an existing result';
        importDetails.append(importSummary);
        importDetails.open = !!state.trackerSource;
        trackers.append(importDetails);
        const importRole = _flashEditorField(importDetails, 'Object in imported trace', state.trackerImportRole,
            value => { state.trackerImportRole = value; }, roles);
        importRole.disabled = state.running;
        const sourceInput = _flashEditorField(importDetails, 'Saved tracker output (.npz)', state.trackerSource,
            value => { state.trackerSource = value; _flashEditorRender(); });
        sourceInput.disabled = state.running;
        const format = document.createElement('p');
        format.textContent = 'NPZ format: frames, poses and lost arrays. Each pose maps the reference camera frame to the current camera frame. The reference frame must be included.';
        importDetails.append(format);
        const importButton = _flashEditorButton(importDetails, 'Import output and preview',
            () => void _flashEditorImportTracker());
        importButton.disabled = state.running || !state.trackerSource?.trim();
        const candidates = state.trackers?.candidates || [];
        if (candidates.length) {
            const heading = document.createElement('p');
            heading.textContent = 'Saved candidates';
            trackers.append(heading);
            for (const candidate of candidates) {
                const item = document.createElement('div');
                item.className = 'flash-editor-keypoint';
                const label = document.createElement('strong');
                label.textContent = `${candidate.tracker}${candidate.inference_mode ? ` (${candidate.inference_mode} context, initial-query pose fit)` : ''} · ${_flashTrackerCoverage(candidate)}`;
                item.append(label);
                const hasConflicts = Object.values(candidate.coverage_by_role || {}).some(info => info.temporal_conflict_frames > 0);
                const missingEvidence = candidate.tracker === 'trackeverything' && !candidate.inference_mode;
                if (hasConflicts || missingEvidence) {
                    const warning = document.createElement('p');
                    warning.className = 'flash-editor-status';
                    warning.textContent = missingEvidence
                        ? 'Earlier sparse-mode result: point-motion diagnostics are unavailable. Regenerate before considering this pose for a demo.'
                        : 'Some pose changes contradict tracked point motion. Inspect affected replay frames before applying.';
                    item.append(warning);
                }
                if (candidate.tracker === 'trackeverything' && !missingEvidence) {
                    const note = document.createElement('p');
                    note.className = 'flash-editor-status';
                    note.textContent = candidate.point_growth
                        ? `Visible-surface review: ${candidate.point_growth.events.length} added views, capped at ${candidate.point_growth.max_points_per_object} points per object and ${candidate.point_growth.new_points_per_view} per view. The pink pose is still fitted from the original points.`
                        : 'This pose is fitted from fixed reference-point tracks. New scene points are not associated with objects; coverage and fit support do not establish pose accuracy.';
                    item.append(note);
                }
                const highlight = () => {
                    if (flashReplay.tracker_candidate?.candidate_id === candidate.candidate_id)
                        _flashSelectOverlay({kind: 'layer', role: candidate.role || null, estimator: 'candidate_tracker'});
                };
                item.onpointerenter = highlight;
                item.onpointermove = highlight;
                item.onfocusin = highlight;
                item.onpointerleave = () => _flashSelectOverlay(null);
                item.onfocusout = event => {
                    if (!item.contains(event.relatedTarget)) _flashSelectOverlay(null);
                };
                const preview = _flashEditorButton(item,
                    flashReplay.tracker_candidate?.candidate_id === candidate.candidate_id ? 'Showing' : 'Preview',
                    () => void _flashEditorPreviewTracker(candidate.candidate_id));
                preview.disabled = state.running;
                const apply = _flashEditorButton(item, 'Apply as new demo version',
                    () => void _flashEditorApplyTracker(candidate.candidate_id));
                apply.disabled = state.running;
                trackers.append(item);
            }
            if (flashReplay.tracker_candidate) _flashEditorButton(trackers, 'Hide candidate overlay',
                () => void _flashEditorPreviewTracker(null));
        }
        if (state.trackerMessage) {
            const note = document.createElement('p');
            note.className = 'flash-editor-status';
            note.textContent = state.trackerMessage;
            trackers.append(note);
        }
    }

    const events = _flashEditorSection(panes.events, 'Reference and events', 'Move the playhead to the desired frame, then choose “Use current”.');
    _flashEditorField(events, 'Appearance reference frame', a.reference_frame,
        value => _flashEditorSetReferenceFrame(value), null, 'number');
    _flashEditorButton(events, 'Use current for appearance', () => {
        _flashEditorSetReferenceFrame(_flashEditorCurrentFrame().frame_index);
    });
    _flashEditorButton(events, 'Jump to appearance', () => _flashEditorJump(Number(a.reference_frame)));
    _flashEditorField(events, 'Last clear grasp view (frame)', a.grasp_reference_frame, value => {
        a.grasp_reference_frame = value;
    }, null, 'number');
    _flashEditorButton(events, 'Use current for grasp view', () => {
        a.grasp_reference_frame = _flashEditorCurrentFrame().frame_index; _flashEditorRender();
    });
    _flashEditorButton(events, 'Jump to grasp view', () => _flashEditorJump(Number(a.grasp_reference_frame)));
    for (const [key, label] of [
        ['grasp_time_s', 'Grasp time (s)'], ['release_time_s', 'Release time (s)'],
        ['start_anchored_until_s', 'Start-anchored until (s)'],
    ]) {
        _flashEditorField(events, label, a[key], value => { a[key] = value; }, null, 'number');
        _flashEditorButton(events, `Use current for ${label.toLowerCase()}`, () => {
            a[key] = _flashEditorCurrentFrame().snapshot.demo.time_s; _flashEditorRender();
        });
    }

    const objects = _flashEditorSection(panes.objects, 'Objects and masks', state.raw
        ? 'This SO-107 importer handles one held object and one target. Choose their names; prompts describe what SAM should segment in the reference frame.'
        : 'These names were entered during preparation. Select an object to inspect its reference mask or draw a correction.');
    if (state.raw && roles.length < 2) {
        _flashEditorField(objects, 'New object name', state.newRole || '', value => { state.newRole = value; });
        _flashEditorField(objects, 'What it looks like', state.newPrompt || '', value => { state.newPrompt = value; });
        _flashEditorButton(objects, 'Add object', () => {
            const name = (state.newRole || '').trim();
            const prompt = (state.newPrompt || '').trim();
            if (!/^[A-Za-z][A-Za-z0-9 _-]{0,39}$/.test(name) || !prompt || a.prompts_by_role[name]) {
                showToast('Name this object', 'Use a distinct name and a visual segmentation prompt.', 'warning');
                return;
            }
            a.prompts_by_role[name] = prompt;
            state.role = name;
            if (!a.held_role) a.held_role = name;
            else if (!a.target_role) a.target_role = name;
            state.newRole = ''; state.newPrompt = '';
            _flashEditorRender();
        });
    }
    if (roles.length) {
        _flashEditorField(objects, 'Object', state.role, role => { _flashEditorMaskRole(role); _flashEditorRender(); }, roles);
        _flashEditorField(objects, 'Segmentation prompt', a.prompts_by_role[state.role], value => {
            a.prompts_by_role[state.role] = value;
        });
        _flashEditorField(objects, 'Held object', a.held_role, value => { a.held_role = value; }, roles);
        _flashEditorField(objects, 'Target object', a.target_role, value => { a.target_role = value; }, roles);
        if (state.raw) _flashEditorButton(objects, 'Remove selected object', () => {
            delete a.prompts_by_role[state.role];
            delete a.axial_symmetry_roles?.[state.role];
            delete a.reference_polygons_by_role?.[state.role];
            state.role = Object.keys(a.prompts_by_role)[0] || '';
            if (!a.prompts_by_role[a.held_role]) a.held_role = state.role;
            if (!a.prompts_by_role[a.target_role]) a.target_role = '';
            _flashEditorRender();
        });
    }
    const symmetry = a.axial_symmetry_roles || {};
    const selectedSymmetry = symmetry[state.role] || {};
    if (state.role) {
        const symmetryNote = document.createElement('p');
        symmetryNote.textContent = 'The selected axis is estimated from this object’s reference RGB-D points; the object name does not choose the axis.';
        objects.append(symmetryNote);
    }
    if (state.role) _flashEditorField(objects, 'Rotational symmetry', selectedSymmetry.axis_kind || 'none', value => {
        const rules = a.axial_symmetry_roles ||= {};
        if (value === 'none') delete rules[state.role];
        else rules[state.role] = { ...rules[state.role], axis_kind: value };
        _flashEditorRender();
    }, ['none', 'long_axis', 'surface_normal']);
    if (state.role && selectedSymmetry.axis_kind) {
        for (const [property, label] of [
            ['ends_interchangeable', 'Ends interchangeable'],
            ['lock_to_baseline', 'Lock axis to baseline'],
        ]) {
            const check = document.createElement('label');
            const box = document.createElement('input');
            box.type = 'checkbox'; box.checked = !!selectedSymmetry[property];
            box.onchange = () => { symmetry[state.role][property] = box.checked; _flashEditorUpdateStatus(); };
            check.append(box, document.createTextNode(` ${label}`));
            objects.append(check);
        }
        _flashEditorField(objects, 'Maximum observed tilt (degrees; blank = unlimited)',
            selectedSymmetry.max_observed_tilt_deg ?? '', value => {
                if (value === '') delete symmetry[state.role].max_observed_tilt_deg;
                else symmetry[state.role].max_observed_tilt_deg = Number(value);
            });
    }
    const polygonStatus = document.createElement('p');
    polygonStatus.id = 'flash-demo-editor-polygon-status';
    objects.append(polygonStatus);
    if (state.role) {
        _flashEditorButton(objects, 'Draw replacement mask', () => {
            _flashEditorJump(Number(a.reference_frame));
            state.drawing = true; state.vertices = [];
            _flashEditorUpdateFrame();
        });
        _flashEditorButton(objects, 'Finish polygon', () => {
            if (state.vertices.length < 3) { showToast('Mask needs three points', 'Click at least three vertices.', 'warning'); return; }
            (a.reference_polygons_by_role ||= {})[state.role] = [...state.vertices];
            state.drawing = false; state.vertices = [];
            _flashEditorUpdateFrame();
        });
        _flashEditorButton(objects, 'Use automatic mask', () => {
            delete a.reference_polygons_by_role?.[state.role];
            state.drawing = false; state.vertices = [];
            _flashEditorUpdateFrame();
        });
    }

    const keypoints = _flashEditorSection(panes.keyframes, 'Key frames', 'Click a marker on the timeline or choose a key frame here to inspect its motion constraint.');
    _flashEditorButton(keypoints, 'Add key frame at current time', () => {
        const time = _flashEditorCurrentFrame().snapshot.demo.time_s;
        const points = state.keypointsDocument.keypoints;
        if (points.some(point => Math.abs(point.t - time) < 1e-6)) {
            showToast('Key frame already exists', 'Choose another frame or edit the existing event.', 'warning');
            return;
        }
        const prior = [...points].reverse().find(point => point.t < time) || points[0] ||
            {kind: 'key', frame: 'world', move: 'line', grip: 'first', constraint: 'pose'};
        points.push({ ...prior, t: time, kind: 'key' });
        points.sort((left, right) => left.t - right.t);
        state.keypointIndex = points.findIndex(point => point.t === time);
        _flashEditorRender();
        _flashRenderTimeline();
    });
    const keypointList = document.createElement('div');
    keypointList.className = 'flash-editor-keypoint-list';
    keypoints.append(keypointList);
    state.keypointsDocument.keypoints.forEach((point, index) => {
        const select = _flashEditorButton(keypointList, `${index + 1} · ${point.t.toFixed(2)} s`, () => {
            state.keypointIndex = index;
            _flashEditorRender();
            _flashEditorJumpTime(point.t);
        });
        select.classList.toggle('active', index === state.keypointIndex);
        if (index !== state.keypointIndex) return;
        const details = document.createElement('details');
        details.className = 'flash-editor-keypoint';
        details.open = true;
        const title = document.createElement('summary');
        title.textContent = `${index + 1}. ${point.t.toFixed(2)} s · ${point.move} · ${point.frame || 'world'}`;
        details.append(title);
        _flashEditorField(details, 'Time (s)', point.t, value => { point.t = value; title.textContent = `${index + 1}. ${value.toFixed(2)} s · ${point.move} · ${point.frame || 'world'}`; }, null, 'number');
        _flashEditorButton(details, 'Use current time', () => {
            point.t = _flashEditorCurrentFrame().snapshot.demo.time_s; _flashEditorRender();
        });
        _flashEditorButton(details, 'Jump', () => {
            _flashEditorJumpTime(point.t);
        });
        _flashEditorField(details, 'Reference', point.frame || 'world', value => { point.frame = value; }, ['world', ...roles]);
        _flashEditorField(details, 'Motion', point.move, value => { point.move = value; }, ['line', 'follow', 'lift']);
        _flashEditorField(details, 'Constraint', point.constraint || 'pose', value => { point.constraint = value; }, ['pose', 'best_effort']);
        _flashEditorField(details, 'Grip', point.grip || 'first', value => { point.grip = value; }, ['first', 'along']);
        if (point.move === 'lift') _flashEditorField(details, 'Lift (mm)', point.up_mm || 0, value => { point.up_mm = value; }, null, 'number');
        _flashEditorButton(details, 'Remove key frame', () => {
            if (state.keypointsDocument.keypoints.length <= 1) {
                showToast('Keep one key frame', 'The prepared skill needs at least one event.', 'warning');
                return;
            }
            state.keypointsDocument.keypoints.splice(index, 1);
            state.keypointIndex = Math.max(0, Math.min(index, state.keypointsDocument.keypoints.length - 1));
            _flashEditorRender();
            _flashRenderTimeline();
        });
        keypoints.append(details);
    });

    const advanced = document.createElement('details');
    advanced.className = 'flash-editor-keypoint';
    advanced.open = state.raw;
    const advancedTitle = document.createElement('summary');
    advancedTitle.textContent = 'Preparation and matching parameters';
    advanced.append(advancedTitle);
    if (state.raw) {
        const note = document.createElement('p');
        note.textContent = 'These are not inferred from the recording. Review them before preparation, or explicitly apply the starter values from prior SO-107 trials.';
        advanced.append(note);
        _flashEditorButton(advanced, 'Apply SO-107 starter parameters', () => {
            Object.assign(a, flashReplay.starter_parameters);
            _flashEditorRender();
        });
    }
    for (const [key, label] of [
        ['max_demo_pose_gap_frames', 'Maximum demo pose gap (frames)'],
        ['connection_speed_m_s', 'Connection speed (m/s)'],
        ['min_similarity', 'Descriptor similarity'],
        ['registration_threshold_m', 'Registration threshold (m)'],
        ['registration_min_inliers', 'Minimum registration inliers'],
        ['registration_trials', 'Registration trials'],
        ['sam_resolution', 'SAM resolution'],
    ]) _flashEditorField(advanced, label, a[key], value => { a[key] = value; }, null, 'number');
    _flashEditorField(advanced, 'Descriptor model', a.descriptor_model_id, value => {
        a.descriptor_model_id = value;
    });
    panes.advanced.append(advanced);

    const output = _flashEditorSection(editor, state.raw ? 'Prepare' : 'Regenerate',
        'Creates a review candidate. The original demo is unchanged.');
    output.classList.add('flash-editor-footer');
    output.hidden = state.section === 'trackers';
    const generate = _flashEditorButton(output, state.raw ? 'Prepare spatial demo' : 'Regenerate prepared demo',
        () => void _flashEditorRegenerate());
    generate.disabled = state.running;
    const message = document.createElement('p');
    message.id = 'flash-demo-editor-status';
    message.className = 'flash-editor-status';
    message.textContent = state.message || '';
    output.append(message);
    _flashEditorMaskRole(state.role);
    _flashEditorUpdateStatus();
}

function _flashEditorJumpTime(time) {
    const nearest = flashReplay.frames.reduce((best, frame, frameIndex) =>
        Math.abs(frame.snapshot.demo.time_s - time) < Math.abs(flashReplay.frames[best].snapshot.demo.time_s - time)
            ? frameIndex : best, 0);
    seekFlashReplay(nearest);
}

async function toggleFlashDemoEditor() {
    if (flashDemoEditor) { closeFlashDemoEditor(); return; }
    if (!['demo', 'raw_demo'].includes(flashReplay?.kind)) return;
    if (flashReplayTimer) toggleFlashReplay();
    const replay = flashReplay;
    const toggle = document.getElementById('flash-demo-edit-toggle');
    toggle.disabled = true;
    try {
        let result;
        const raw = replay.kind === 'raw_demo';
        if (raw) {
            const annotation = {
                repo_id: `local/${replay.datasetRoot.split('/').pop()}`,
                episode: replay.episode, robot_arm: '', calibration_path: '',
                reference_frame: _flashEditorCurrentFrame().frame_index,
                grasp_reference_frame: null, prompts_by_role: {}, held_role: '', target_role: '',
                grasp_time_s: null, release_time_s: null, start_anchored_until_s: null,
                axial_symmetry_roles: {},
            };
            for (const key of Object.keys(replay.starter_parameters)) annotation[key] = null;
            result = {annotation, keypoints_document: {keypoints: []}, reference_masks: {}, preparation_report: null};
        } else {
            const response = await fetch(`/api/run/flash/demo/preparation?${new URLSearchParams({
                dataset_root: replay.datasetRoot, bundle_path: replay.bundle_path, episode: String(replay.episode),
            })}`);
            result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'Cannot read preparation inputs');
        }
        if (flashReplay !== replay) return;
        let trackers = {trackers: [], candidates: []};
        let trackerMessage = '';
        if (!raw) {
            try {
                const query = new URLSearchParams({dataset_root: replay.datasetRoot, bundle_path: replay.bundle_path});
                const response = await fetch(`/api/run/flash/demo/trackers?${query}`);
                trackers = await response.json();
                if (!response.ok) throw new Error(trackers.detail || 'Cannot load demo trackers');
            } catch (error) {
                trackers = {trackers: [], candidates: []};
                trackerMessage = `Tracker candidates unavailable: ${error.message}`;
            }
        }
        if (flashReplay !== replay) return;
        flashDemoEditor = {
            annotation: structuredClone(result.annotation),
            keypointsDocument: structuredClone(result.keypoints_document),
            referenceMasks: result.reference_masks,
            preparationReport: result.preparation_report,
            role: result.annotation.held_role,
            section: raw ? 'setup' : 'objects', keypointIndex: 0,
            raw, newRole: '', newPrompt: '',
            original: JSON.stringify([result.annotation, result.keypoints_document]),
            drawing: false, vertices: [], running: false, message: '',
            trackers, tracker: trackers.trackers.find(item => item.runnable)?.id || 'point2pose',
            growPoints: true,
            maxPoints: (trackers.trackers.find(item => item.id === 'trackeverything')?.initial_points_per_object || 128) * 3,
            pointsPerView: trackers.trackers.find(item => item.id === 'trackeverything')?.initial_points_per_object || 128,
            surfaceToleranceMm: Number(result.annotation.registration_threshold_m || 0.005) * 1000,
            trackerRole: '',
            trackerImportRole: result.annotation.held_role,
            trackerSource: '', trackerMessage,
        };
        const viewer = document.getElementById('rerun-viewer');
        viewer.classList.add('flash-demo-editing');
        const grid = viewer.querySelector('.flash-replay-grid');
        const mask = document.createElement('img');
        mask.className = 'flash-demo-mask';
        mask.alt = '';
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        polygon.classList.add('flash-demo-polygon');
        polygon.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        polygon.addEventListener('pointerdown', _flashEditorPolygonClick);
        grid.append(mask, polygon);
        const panel = document.createElement('aside');
        panel.id = 'flash-demo-editor';
        panel.className = 'flash-demo-editor';
        panel.setAttribute('aria-label', 'Edit FLASH demo in playback');
        viewer.append(panel);
        toggle.textContent = 'Close editor';
        _flashEditorRender();
        _flashRenderTimeline();
    } catch (error) { showToast('Cannot edit demo', error.message, 'error'); }
    finally { toggle.disabled = false; }
}

async function _flashEditorRefreshTrackers(state) {
    const query = new URLSearchParams({dataset_root: flashReplay.datasetRoot, bundle_path: flashReplay.bundle_path});
    const response = await fetch(`/api/run/flash/demo/trackers?${query}`);
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Cannot list tracker candidates');
    if (flashDemoEditor === state) state.trackers = result;
}

function _flashTrackerCoverage(candidate) {
    const coverage = candidate.coverage_by_role || {[candidate.role]: candidate};
    return Object.entries(coverage).map(([role, info]) =>
        `${role}: ${info.valid_frames}/${info.total_frames} pose frames` +
        (info.median_inliers == null ? '' : ` · median fit support ${info.median_inliers}/${info.median_visible_queries} visible tracks`) +
        (info.temporal_conflict_frames ? ` · ${info.temporal_conflict_frames} motion conflicts` : '')).join(' · ');
}

async function _flashEditorProcessTracker() {
    const state = flashDemoEditor;
    if (!state || state.running) return;
    const replay = flashReplay;
    const growthEnabled = state.tracker === 'trackeverything' && state.growPoints &&
        state.trackers.trackers.find(item => item.id === state.tracker)?.supports_point_growth !== false;
    state.running = true;
    state.trackerMessage = `Processing ${state.trackerRole || 'all annotated objects'} with ${state.tracker} on the recorded RGB-D frames…`;
    _flashEditorRender();
    try {
        const response = await fetch('/api/run/flash/demo/trackers/process', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({dataset_root: flashReplay.datasetRoot, bundle_path: flashReplay.bundle_path,
                episode: flashReplay.episode, tracker: state.tracker, role: state.trackerRole || null,
                max_points_per_object: growthEnabled ? state.maxPoints : null,
                new_points_per_view: growthEnabled ? state.pointsPerView : null,
                surface_tolerance_m: growthEnabled
                    ? state.surfaceToleranceMm / 1000 : null}),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Tracker could not process the demo');
        if (flashDemoEditor !== state || flashReplay !== replay) {
            showToast('Tracker candidate saved', 'Open the source demo’s Trackers section to review it.', 'success');
            return;
        }
        await _flashEditorRefreshTrackers(state);
        await previewFlashDemoTrackerCandidate(result.candidate_id);
        state.trackerMessage = `${_flashTrackerCoverage(result)}. Review the pink candidate against the saved pose before applying.`;
    } catch (error) {
        state.trackerMessage = error.message;
        showToast('Cannot process demo', error.message, 'error');
    } finally {
        state.running = false;
        if (flashDemoEditor === state) _flashEditorRender();
    }
}

async function _flashEditorImportTracker() {
    const state = flashDemoEditor;
    if (!state || state.running || !state.trackerSource?.trim()) return;
    const replay = flashReplay;
    state.running = true;
    state.trackerMessage = 'Validating and aligning saved tracker poses…';
    _flashEditorRender();
    try {
        const response = await fetch('/api/run/flash/demo/trackers/import', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({dataset_root: flashReplay.datasetRoot, bundle_path: flashReplay.bundle_path,
                episode: flashReplay.episode, tracker: state.tracker, role: state.trackerImportRole,
                source_path: state.trackerSource}),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Tracker output could not be imported');
        if (flashDemoEditor !== state || flashReplay !== replay) return;
        await _flashEditorRefreshTrackers(state);
        await previewFlashDemoTrackerCandidate(result.candidate_id);
        state.trackerMessage = `${_flashTrackerCoverage(result)}. Review the pink candidate against the saved outline before applying.`;
    } catch (error) {
        state.trackerMessage = error.message;
        showToast('Cannot import tracker output', error.message, 'error');
    } finally {
        state.running = false;
        if (flashDemoEditor === state) _flashEditorRender();
    }
}

async function _flashEditorPreviewTracker(candidateId) {
    const state = flashDemoEditor;
    if (!state || state.running) return;
    state.running = true;
    try {
        await previewFlashDemoTrackerCandidate(candidateId);
        state.trackerMessage = candidateId ? 'Pink: candidate pose. Green/orange: pose used by saved skill.' : '';
    } catch (error) {
        state.trackerMessage = error.message;
        showToast('Cannot preview tracker', error.message, 'error');
    } finally {
        state.running = false;
        if (flashDemoEditor === state) _flashEditorRender();
    }
}

async function _flashEditorApplyTracker(candidateId) {
    const state = flashDemoEditor;
    if (!state || state.running) return;
    const replay = flashReplay;
    state.running = true;
    state.trackerMessage = 'Creating a new prepared demo version…';
    _flashEditorRender();
    try {
        const response = await fetch('/api/run/flash/demo/trackers/apply', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({dataset_root: flashReplay.datasetRoot, bundle_path: flashReplay.bundle_path,
                episode: flashReplay.episode, candidate_id: candidateId}),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Tracker poses could not be applied');
        if (flashDemoEditor !== state || flashReplay !== replay) {
            showToast('Tracker poses applied', 'The new version is saved in the source demo.', 'success');
            return;
        }
        const {datasetRoot, episode} = replay;
        await _loadFlashRecordings();
        if (flashReplay !== replay) return;
        await loadFlashDemoReplay(datasetRoot, episode, result.bundle_path);
        showToast('Tracker poses applied',
            `New demo version: ${result.report.changed_reference_samples} reference and ${result.report.changed_held_samples} held samples changed.`, 'success');
    } catch (error) {
        state.trackerMessage = error.message;
        showToast('Cannot apply tracker', error.message, 'error');
    } finally {
        state.running = false;
        if (flashDemoEditor === state) _flashEditorRender();
    }
}

async function _flashEditorRegenerate() {
    if (!flashDemoEditor || flashDemoEditor.running) return;
    const state = flashDemoEditor;
    const a = state.annotation;
    const points = state.keypointsDocument.keypoints;
    const roles = Object.keys(a.prompts_by_role);
    if (state.raw) {
        const requiredNumbers = [
            'grasp_time_s', 'release_time_s', 'start_anchored_until_s',
            'max_demo_pose_gap_frames', 'connection_speed_m_s', 'min_similarity',
            'registration_threshold_m', 'registration_min_inliers', 'registration_trials', 'sam_resolution',
        ];
        if (roles.length !== 2 || a.held_role === a.target_role ||
            !roles.includes(a.held_role) || !roles.includes(a.target_role)) {
            state.section = 'objects'; state.message = 'Name two distinct objects and select held and target.';
            _flashEditorRender(); return;
        }
        if (!a.repo_id?.trim() || !a.robot_arm || !a.calibration_path?.trim()) {
            state.section = 'setup'; state.message = 'Dataset identity, arm side and calibration file are required.';
            _flashEditorRender(); return;
        }
        if (!a.descriptor_model_id?.trim() || requiredNumbers.some(key => !Number.isFinite(a[key])) ||
            !(a.grasp_time_s > 0 && a.grasp_time_s < a.release_time_s) ||
            !(a.start_anchored_until_s >= a.grasp_time_s && a.start_anchored_until_s <= a.release_time_s) ||
            !Number.isInteger(a.reference_frame) || !Number.isInteger(a.grasp_reference_frame)) {
            state.section = 'events'; state.message = 'Choose reference frames, grasp, release and lift completion; review matching parameters.';
            _flashEditorRender(); return;
        }
        if (!points.length) {
            state.section = 'keyframes'; state.message = 'Add at least one key frame from the playback timeline.';
            _flashEditorRender(); return;
        }
    }
    if (points.some((point, index) => !Number.isFinite(point.t) || (index && point.t <= points[index - 1].t))) {
        state.message = 'Key frame times must increase.'; _flashEditorRender(); return;
    }
    state.running = true;
    state.message = 'Preparing a new version…';
    _flashEditorRender();
    try {
        const response = await fetch('/api/run/flash/demo/prepare', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_root: flashReplay.datasetRoot, bundle_path: flashReplay.bundle_path,
                episode: flashReplay.episode, annotation: state.annotation,
                keypoints_document: state.keypointsDocument,
            }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Preparation could not start');
        connectOutputSSE();
        updateRunUI(true, 'flash_prepare');
        while (true) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            const statusResponse = await fetch('/api/run/status');
            const status = await statusResponse.json();
            if (status.running) continue;
            if (status.last_error?.command === 'flash_prepare') {
                throw new Error(status.last_error.reason || 'Preparation failed; inspect Run output');
            }
            break;
        }
        const {datasetRoot, episode} = flashReplay;
        await _loadFlashRecordings();
        await loadFlashDemoReplay(datasetRoot, episode, result.bundle_path);
        showToast('Prepared demo updated', 'The new version is open in playback.', 'success');
    } catch (error) {
        state.message = error.message;
        if (flashDemoEditor === state) _flashEditorRender();
        showToast('Cannot regenerate demo', error.message, 'error');
    } finally {
        state.running = false;
        if (flashDemoEditor === state) _flashEditorRender();
    }
}
