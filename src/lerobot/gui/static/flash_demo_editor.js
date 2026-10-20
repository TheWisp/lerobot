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
    control.value = String(value ?? '');
    control.addEventListener('change', () => onChange(type === 'number' ? Number(control.value) : control.value));
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
    button.onclick = action;
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
        Number(flashDemoEditor.referenceMasks[role].frame_index) !== Number(annotation.reference_frame);
    const frameText = document.getElementById('flash-demo-editor-frame');
    if (frameText) frameText.textContent = `Frame ${current.frame_index} · ${current.snapshot.demo.time_s.toFixed(2)} s`;
    const polygonText = document.getElementById('flash-demo-editor-polygon-status');
    if (polygonText) polygonText.textContent = flashDemoEditor.drawing
        ? `${flashDemoEditor.vertices.length} vertices; finish after at least 3`
        : saved?.length ? `Manual ${role} mask: ${saved.length} vertices` :
            `Prepared ${role} mask shown on its reference frame`;
}

function _flashEditorMaskRole(role) {
    flashDemoEditor.role = role;
    flashDemoEditor.drawing = false;
    flashDemoEditor.vertices = [];
    const mask = document.querySelector('#rerun-viewer .flash-demo-mask');
    mask.src = `/api/run/flash/demo/reference-mask?${new URLSearchParams({
        dataset_root: flashReplay.datasetRoot, bundle_path: flashReplay.bundle_path, role,
    })}`;
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
    editor.replaceChildren();
    const state = flashDemoEditor;
    const a = state.annotation;
    const roles = Object.keys(a.prompts_by_role);
    const heading = document.createElement('h3');
    heading.textContent = 'Edit this demo';
    editor.append(heading);
    const summary = document.createElement('p');
    summary.textContent = 'Review the actual camera frames and extracted poses. Changes create a new prepared version; this recording stays intact.';
    editor.append(summary);
    const report = state.preparationReport;
    if (report) {
        if (report.review_status === 'candidate') {
            const candidate = document.createElement('p');
            candidate.textContent = 'Review candidate · the prior prepared version remains the default.';
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
    const trackerPanel = document.getElementById('flash-tracker-panel');
    if (trackerPanel) editor.append(trackerPanel);
    const frameText = document.createElement('p');
    frameText.id = 'flash-demo-editor-frame';
    editor.append(frameText);

    const events = _flashEditorSection(editor, 'Reference and events', 'Use the playback scrubber to select an exact frame.');
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

    const objects = _flashEditorSection(editor, 'Objects and masks', 'Select a role, inspect its saved mask, or draw a replacement on the appearance reference frame.');
    _flashEditorField(objects, 'Mask role', state.role, role => _flashEditorMaskRole(role), roles);
    for (const role of roles) {
        _flashEditorField(objects, `${role} prompt`, a.prompts_by_role[role], value => {
            a.prompts_by_role[role] = value;
        });
    }
    _flashEditorField(objects, 'Held role', a.held_role, value => { a.held_role = value; }, roles);
    _flashEditorField(objects, 'Target role', a.target_role, value => { a.target_role = value; }, roles);
    const symmetry = a.axial_symmetry_roles || (a.axial_symmetry_roles = {});
    const selectedSymmetry = symmetry[state.role] || {};
    _flashEditorField(objects, `${state.role} axial symmetry`, selectedSymmetry.axis_kind || 'none', value => {
        if (value === 'none') delete symmetry[state.role];
        else symmetry[state.role] = { ...symmetry[state.role], axis_kind: value };
        _flashEditorRender();
    }, ['none', 'long_axis', 'surface_normal']);
    if (selectedSymmetry.axis_kind) {
        for (const [property, label] of [
            ['ends_interchangeable', 'Ends interchangeable'],
            ['lock_to_baseline', 'Lock axis to baseline'],
        ]) {
            const check = document.createElement('label');
            const box = document.createElement('input');
            box.type = 'checkbox'; box.checked = !!selectedSymmetry[property];
            box.onchange = () => { symmetry[state.role][property] = box.checked; };
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

    const keypoints = _flashEditorSection(editor, 'Key frames', 'These event times and constraints control which motions are preserved or replanned.');
    _flashEditorButton(keypoints, 'Add key frame at current time', () => {
        const time = _flashEditorCurrentFrame().snapshot.demo.time_s;
        const points = state.keypointsDocument.keypoints;
        if (points.some(point => Math.abs(point.t - time) < 1e-6)) {
            showToast('Key frame already exists', 'Choose another frame or edit the existing event.', 'warning');
            return;
        }
        const prior = [...points].reverse().find(point => point.t < time) || points[0];
        points.push({ ...prior, t: time, kind: 'key' });
        points.sort((left, right) => left.t - right.t);
        _flashEditorRender();
    });
    state.keypointsDocument.keypoints.forEach((point, index) => {
        const details = document.createElement('details');
        details.className = 'flash-editor-keypoint';
        const title = document.createElement('summary');
        title.textContent = `${index + 1}. ${point.t.toFixed(2)} s · ${point.move} · ${point.frame || 'world'}`;
        details.append(title);
        _flashEditorField(details, 'Time (s)', point.t, value => { point.t = value; title.textContent = `${index + 1}. ${value.toFixed(2)} s · ${point.move} · ${point.frame || 'world'}`; }, null, 'number');
        _flashEditorButton(details, 'Use current time', () => {
            point.t = _flashEditorCurrentFrame().snapshot.demo.time_s; _flashEditorRender();
        });
        _flashEditorButton(details, 'Jump', () => {
            const nearest = flashReplay.frames.reduce((best, frame, frameIndex) =>
                Math.abs(frame.snapshot.demo.time_s - point.t) < Math.abs(flashReplay.frames[best].snapshot.demo.time_s - point.t)
                    ? frameIndex : best, 0);
            seekFlashReplay(nearest);
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
            _flashEditorRender();
        });
        keypoints.append(details);
    });

    const advanced = document.createElement('details');
    advanced.className = 'flash-editor-keypoint';
    const advancedTitle = document.createElement('summary');
    advancedTitle.textContent = 'Preparation and matching parameters';
    advanced.append(advancedTitle);
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
    editor.append(advanced);

    const output = _flashEditorSection(editor, 'Regenerate', 'Runs preparation off the robot and opens the resulting version in this same playback.');
    const generate = _flashEditorButton(output, 'Regenerate prepared demo', () => void _flashEditorRegenerate());
    generate.disabled = state.running;
    const message = document.createElement('p');
    message.id = 'flash-demo-editor-status';
    message.className = 'flash-editor-status';
    message.textContent = state.message || '';
    output.append(message);
    _flashEditorMaskRole(state.role);
}

async function toggleFlashDemoEditor() {
    if (flashDemoEditor) { closeFlashDemoEditor(); return; }
    if (flashReplay?.kind !== 'demo') return;
    if (flashReplayTimer) toggleFlashReplay();
    const replay = flashReplay;
    const toggle = document.getElementById('flash-demo-edit-toggle');
    toggle.disabled = true;
    try {
        const response = await fetch(`/api/run/flash/demo/preparation?${new URLSearchParams({
            dataset_root: replay.datasetRoot, bundle_path: replay.bundle_path, episode: String(replay.episode),
        })}`);
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Cannot read preparation inputs');
        if (flashReplay !== replay) return;
        flashDemoEditor = {
            annotation: structuredClone(result.annotation),
            keypointsDocument: structuredClone(result.keypoints_document),
            referenceMasks: result.reference_masks,
            preparationReport: result.preparation_report,
            role: result.annotation.held_role,
            drawing: false, vertices: [], running: false, message: '',
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
    } catch (error) { showToast('Cannot edit demo', error.message, 'error'); }
    finally { toggle.disabled = false; }
}

async function _flashEditorRegenerate() {
    if (!flashDemoEditor || flashDemoEditor.running) return;
    const state = flashDemoEditor;
    const points = state.keypointsDocument.keypoints;
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
