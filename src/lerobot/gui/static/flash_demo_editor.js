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
    if (title) {
        const heading = document.createElement('strong');
        heading.textContent = title;
        section.append(heading);
    }
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

function _flashEditorReferenceFrame(role) {
    const annotation = flashDemoEditor.annotation;
    return Number(annotation.reference_frames_by_role?.[role] ?? annotation.reference_frame);
}

function _flashEditorSetReferenceFrame(frameIndex) {
    const annotation = flashDemoEditor.annotation;
    const role = flashDemoEditor.role;
    if (!role || !Number.isInteger(Number(frameIndex))) return;
    if (_flashEditorReferenceFrame(role) !== Number(frameIndex)) {
        (annotation.reference_frames_by_role ||= {})[role] = Number(frameIndex);
        if (annotation.reference_polygons_by_role) delete annotation.reference_polygons_by_role[role];
        flashDemoEditor.message = `${role} reference changed; review its new mask.`;
        flashDemoEditor.referenceReview = null;
    }
    _flashEditorRender();
    _flashRenderTimeline();
}

async function _flashEditorReviewCurrentView() {
    const state = flashDemoEditor;
    const replay = flashReplay;
    if (!state || state.raw || !state.role || state.referenceReviewRunning || !replay?.bundle_path) return;
    const role = state.role;
    const frameIndex = _flashEditorCurrentFrame().frame_index;
    state.referenceReviewRunning = true;
    state.referenceReview = null;
    _flashEditorRender();
    try {
        const response = await fetch('/api/run/flash/demo/reference-view/review', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({dataset_root: replay.datasetRoot, episode: replay.episode,
                bundle_path: replay.bundle_path, role, frame_index: frameIndex}),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Could not check this view');
        if (flashDemoEditor === state && flashReplay === replay)
            state.referenceReview = {role, frameIndex, result};
    } catch (error) {
        if (flashDemoEditor === state && flashReplay === replay)
            state.referenceReview = {role, frameIndex, error: error.message};
    } finally {
        if (flashDemoEditor === state) {
            state.referenceReviewRunning = false;
            _flashEditorRender();
            document.getElementById('flash-reference-review-result')?.scrollIntoView({block: 'center'});
        }
    }
}

function _flashEditorUpdateStatus() {
    if (!flashDemoEditor) return;
    const changed = JSON.stringify([flashDemoEditor.annotation, flashDemoEditor.keypointsDocument]) !== flashDemoEditor.original;
    const status = document.getElementById('flash-demo-editor-change-status');
    if (status) status.textContent = flashDemoEditor.raw
        ? changed ? 'Unsaved setup · prepare to review a new version' : 'Raw demo · no preparation saved'
        : changed ? 'Unsaved edits · preview to review' : 'Viewing saved preparation';
}

function _flashEditorFollowPlayhead() {
    const state = flashDemoEditor;
    if (!state || state.drawing || state.pickingReference || state.pickingTurn) return false;
    const points = state.keypointsDocument.keypoints;
    if (!points.length) return false;
    const time = _flashEditorCurrentFrame().snapshot.demo.time_s;
    const nearest = points.reduce((best, point, index) =>
        Math.abs(point.t - time) < Math.abs(points[best].t - time) ? index : best, 0);
    if (nearest === state.keypointIndex) return false;
    state.keypointIndex = nearest;
    return true;
}

function _flashEditorUpdateFrame() {
    if (!flashDemoEditor) return;
    const current = _flashEditorCurrentFrame();
    const annotation = flashDemoEditor.annotation;
    const role = flashDemoEditor.role;
    const referenceFrame = _flashEditorReferenceFrame(role);
    const isReference = current.frame_index === referenceFrame;
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
        Number(flashDemoEditor.referenceMasks[role]?.frame_index) !== referenceFrame;
    const frameText = document.getElementById('flash-demo-editor-frame');
    if (frameText) frameText.textContent = `Frame ${current.frame_index} · ${current.snapshot.demo.time_s.toFixed(2)} s`;
    const playheadTitle = document.getElementById('flash-editor-playhead-title');
    if (playheadTitle) playheadTitle.textContent = `At ${current.snapshot.demo.time_s.toFixed(2)} s`;
    const polygonText = document.getElementById('flash-demo-editor-polygon-status');
    if (polygonText) polygonText.textContent = flashDemoEditor.drawing
        ? `${flashDemoEditor.vertices.length} points · click at least 3 to finish`
        : saved?.length ? `Corrected ${role} mask` :
            flashDemoEditor.raw ? 'Mask will come from the description' :
                `Automatic mask · frame ${referenceFrame}`;
}

function _flashEditorMaskRole(role) {
    const changed = flashDemoEditor.role !== role;
    flashDemoEditor.role = role;
    if (changed) {
        flashDemoEditor.drawing = false;
        flashDemoEditor.vertices = [];
        flashDemoEditor.referenceReview = null;
    }
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
    if (current.frame_index !== _flashEditorReferenceFrame(flashDemoEditor.role)) return;
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
    const previousScroll = editor.scrollTop;
    editor.replaceChildren();
    const state = flashDemoEditor;
    const a = state.annotation;
    const roles = Object.keys(a.prompts_by_role);
    const heading = document.createElement('h3');
    heading.textContent = state.raw ? 'Prepare demo' : 'Edit demo';
    editor.append(heading);
    const guidance = document.createElement('p');
    guidance.textContent = 'Scrub the replay. Select an object, mark the important moments, then edit the nearest key frame.';
    editor.append(guidance);
    const changed = document.createElement('p');
    changed.id = 'flash-demo-editor-change-status';
    editor.append(changed);
    const frameText = document.createElement('p');
    frameText.id = 'flash-demo-editor-frame';
    editor.append(frameText);

    const panes = {};
    const requestedSection = state.section;
    state.section = null;
    for (const key of ['setup', 'objects', 'events', 'references', 'keyframes',
        'pose', 'trackers', 'objectOptions', 'exactTimes', 'advanced']) {
        panes[key] = document.createElement('div');
        panes[key].className = 'flash-editor-pane';
        panes[key].dataset.section = key;
    }
    const requestedView = ({
        keyframes: 'motion', events: 'motion', objects: 'objects', references: 'objects',
        objectOptions: 'objects', pose: 'tracking', trackers: 'tracking',
        setup: 'source', exactTimes: 'source', advanced: 'source',
    })[requestedSection];
    if (requestedView) state.editorView = requestedView;
    const views = state.raw
        ? [['source', 'Source'], ['objects', 'Objects'], ['motion', 'Motion']]
        : [['motion', 'Motion'], ['objects', 'Objects'], ['tracking', 'Tracking'], ['source', 'Source']];
    if (!views.some(([key]) => key === state.editorView)) state.editorView = state.raw ? 'source' : 'motion';
    const navigation = document.createElement('nav');
    navigation.className = 'flash-editor-navigation';
    navigation.setAttribute('aria-label', 'Demo editing tools');
    for (const [key, label] of views) {
        const button = _flashEditorButton(navigation, label, () => {
            state.editorView = key;
            _flashEditorRender();
            editor.scrollTop = 0;
        });
        button.classList.toggle('is-selected', state.editorView === key);
        button.setAttribute('aria-current', state.editorView === key ? 'page' : 'false');
    }
    editor.append(navigation);
    const view = document.createElement('div');
    view.className = 'flash-editor-view';
    editor.append(view);
    if (state.editorView === 'motion') view.append(panes.keyframes, panes.events);
    if (state.editorView === 'objects') view.append(panes.objects, panes.objectOptions);
    if (state.editorView === 'tracking') view.append(panes.pose, panes.trackers);
    if (state.editorView === 'source') view.append(panes.setup, panes.exactTimes, panes.advanced);
    if (!state.raw && state.editorView !== 'tracking') {
        const retainedPosePanel = document.createElement('div');
        retainedPosePanel.hidden = true;
        retainedPosePanel.append(panes.pose);
        editor.append(retainedPosePanel);
    }

    const setup = _flashEditorSection(panes.setup, 'Recording and calibration', state.raw
        ? 'The raw frames contain no FLASH object names or camera-to-arm calibration. Supply the calibration used when this demo was recorded.'
        : 'These capture settings were saved with the selected preparation.');
    _flashEditorField(setup, 'Dataset identity', a.repo_id, value => { a.repo_id = value; });
    _flashEditorField(setup, 'Arm side', a.robot_arm, value => { a.robot_arm = value; }, ['', 'left', 'right']);
    _flashEditorField(setup, 'Robot/camera calibration file', a.calibration_path, value => { a.calibration_path = value; });
    const sceneTracking = document.createElement('p');
    sceneTracking.className = 'form-hint';
    sceneTracking.textContent = a.causal_tracking
        ? 'Scene tracking is saved in this demo. Group predictions can bridge missing direct views.'
        : 'Direct vision only: no group bridge is saved in this demo.';
    setup.append(sceneTracking);
    _flashEditorField(setup, 'Trial settings file for scene tracking', state.trackingConfigPath,
        value => { state.trackingConfigPath = value; });

    const pose = _flashEditorSection(panes.pose, 'Pose evidence', state.raw
        ? 'No object pose exists yet. Preparation will extract a first pose trace from the RGB-D recording.'
        : 'Hover an item to highlight its projection in the image. This is the pose saved in the selected prepared version.');
    if (trackerPanel && !state.raw) pose.append(trackerPanel);

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
                const growingFit = candidate.point_growth?.pose_fit === 'causal_growing_rgbd';
                const fitName = growingFit ? 'growing-point RGB-D pose fit' : 'initial-query pose fit';
                label.textContent = `${candidate.tracker}${candidate.inference_mode ? ` (${candidate.inference_mode} context, ${fitName})` : ''} · ${_flashTrackerCoverage(candidate)}`;
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
                    if (growingFit) {
                        const corrected = Object.values(candidate.point_growth.surface_refined_frames_by_role || {}).reduce((a, b) => a + b, 0);
                        note.textContent = `${candidate.point_growth.events.length} added views · ${corrected} frames accepted an RGB-D surface correction. The pink pose uses depth-supported growing points. Review the mask, depth and point tracks before applying it.`;
                    } else if (candidate.point_growth) {
                        note.textContent = `Earlier visible-surface review: ${candidate.point_growth.events.length} added views. The pink pose still uses only the original points; regenerate to use them for the fit.`;
                    } else {
                        note.textContent = 'This pose is fitted from fixed reference-point tracks. New scene points are not associated with objects; coverage and fit support do not establish pose accuracy.';
                    }
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

    const events = _flashEditorSection(panes.events, 'Mark moments');
    events.querySelector('strong').id = 'flash-editor-playhead-title';
    for (const [key, label] of [
        ['grasp_time_s', 'Grasp'], ['release_time_s', 'Release'],
    ]) {
        const row = document.createElement('div');
        row.className = 'flash-editor-moment';
        const value = document.createElement('span');
        value.textContent = a[key] == null ? `${label} not marked` : `${label} · ${Number(a[key]).toFixed(2)} s`;
        row.append(value);
        _flashEditorButton(row, `Mark ${label.toLowerCase()} here`, () => {
            a[key] = _flashEditorCurrentFrame().snapshot.demo.time_s; _flashEditorRender();
        });
        events.append(row);
    }
    const exactTimes = _flashEditorSection(panes.exactTimes, 'Exact frame and event values');
    _flashEditorField(exactTimes, 'Grasp time (s)', a.grasp_time_s, value => {
        a.grasp_time_s = value; _flashEditorRender();
    }, null, 'number');
    _flashEditorField(exactTimes, 'Release time (s)', a.release_time_s, value => {
        a.release_time_s = value; _flashEditorRender();
    }, null, 'number');
    const references = _flashEditorSection(panes.references, 'Clear views',
        'Choose a clear view of the selected object for matching.');
    for (const [label, value, buttonLabel, mark] of [
        [`Clear view of ${state.role || 'selected object'}`, `frame ${_flashEditorReferenceFrame(state.role)}`, 'Set appearance here', () =>
            _flashEditorSetReferenceFrame(_flashEditorCurrentFrame().frame_index)],
        ['Last clear view before hold', a.grasp_reference_frame == null ? 'not marked' :
            `frame ${a.grasp_reference_frame}`, 'Set grasp view here', () => {
                a.grasp_reference_frame = _flashEditorCurrentFrame().frame_index;
                _flashEditorRender();
            }],
        ['Clear of tray after lift', a.start_anchored_until_s == null ? 'not marked' :
            `${Number(a.start_anchored_until_s).toFixed(2)} s`, 'Mark lift clear here', () => {
                a.start_anchored_until_s = _flashEditorCurrentFrame().snapshot.demo.time_s;
                _flashEditorRender();
            }],
    ]) {
        const row = document.createElement('div');
        row.className = 'flash-editor-moment';
        row.append(Object.assign(document.createElement('span'), {textContent: `${label} · ${value}`}));
        const button = _flashEditorButton(row, buttonLabel, mark);
        if (buttonLabel === 'Set appearance here') button.disabled = !state.role;
        references.append(row);
    }
    if (!state.raw) {
        _flashEditorButton(references, 'Go to appearance view', () =>
            _flashEditorJump(_flashEditorReferenceFrame(state.role)));
        const review = _flashEditorButton(references,
            state.referenceReviewRunning ? 'Checking view…' : 'Check this view',
            () => void _flashEditorReviewCurrentView());
        review.disabled = !state.role || state.referenceReviewRunning ||
            !state.preparationReport?.pose_trace_cache;
        const explanation = document.createElement('p');
        explanation.textContent = 'Uses the saved demo. This check does not add the view or include unsaved edits.';
        references.append(explanation);
        if (state.referenceReview) {
            const result = state.referenceReview;
            const box = document.createElement('div');
            box.className = 'flash-editor-keypoint';
            box.id = 'flash-reference-review-result';
            const title = document.createElement('strong');
            title.textContent = `${result.role} · frame ${result.frameIndex}`;
            box.append(title);
            const message = document.createElement('p');
            box.append(message);
            const candidate = result.result?.candidate;
            if (result.error) message.textContent = result.error;
            else if (candidate?.status !== 'registration_hypothesis')
                message.textContent = candidate?.reason || 'This view could not be checked.';
            else {
                const score = candidate.cross_frame;
                const mm = value => value == null ? 'unavailable' : `${(value * 1000).toFixed(2)} mm`;
                message.textContent = `Checked on ${score.frames} other observed frames. Saved view → with this view:`;
                const surface = document.createElement('p');
                surface.textContent = `Measured points to surface: ${mm(score.first_view_measured_p90_median_m)} → ${mm(score.combined_views_measured_p90_median_m)}`;
                box.append(surface);
                const depth = document.createElement('p');
                depth.textContent = `Surface to measured depth: ${mm(score.first_view_reverse_p90_median_m)} → ${mm(score.combined_views_reverse_p90_median_m)}`;
                box.append(depth);
                const caveat = document.createElement('p');
                caveat.textContent = 'The same estimated poses placed and checked this view. This does not prove its position is correct.';
                box.append(caveat);
            }
            if (candidate?.status === 'registration_hypothesis' &&
                result.result.largest_reverse_conflicts?.length) {
                const label = document.createElement('p');
                label.textContent = 'Check frames with the largest depth disagreement:';
                box.append(label);
                for (const row of result.result.largest_reverse_conflicts.slice(0, 3)) {
                    _flashEditorButton(box, `Frame ${row.frame_index}`, () => _flashEditorJump(row.frame_index));
                }
            }
            references.append(box);
        }
    }
    _flashEditorField(exactTimes, 'Appearance reference frame for selected object',
        _flashEditorReferenceFrame(state.role), value => _flashEditorSetReferenceFrame(value), null, 'number');
    _flashEditorField(exactTimes, 'Last clear grasp view (frame)', a.grasp_reference_frame, value => {
        a.grasp_reference_frame = value;
    }, null, 'number');
    _flashEditorField(exactTimes, 'Start-anchored until (s)', a.start_anchored_until_s,
        value => { a.start_anchored_until_s = value; }, null, 'number');

    const objects = _flashEditorSection(panes.objects, 'Objects', state.raw
        ? 'Name objects to track. Choose one held object and its target for this imported action.' : '');
    const addObject = state.raw ? objects : document.createElement('details');
    if (!state.raw) {
        addObject.append(Object.assign(document.createElement('summary'), {textContent: 'Manage tracked objects'}));
    }
    _flashEditorField(addObject, 'New object name', state.newRole || '', value => { state.newRole = value; });
    _flashEditorField(addObject, 'What it looks like', state.newPrompt || '', value => { state.newPrompt = value; });
    _flashEditorButton(addObject, 'Add object', () => {
        const name = (state.newRole || '').trim();
        const prompt = (state.newPrompt || '').trim();
        if (!/^[A-Za-z][A-Za-z0-9 _-]{0,39}$/.test(name) || !prompt || prompt.includes('.') ||
            a.prompts_by_role[name] || Object.values(a.prompts_by_role).includes(prompt)) {
            showToast('Name this object', 'Use a distinct name and a unique visual prompt without periods.', 'warning');
            return;
        }
        a.prompts_by_role[name] = prompt;
        state.role = name;
        if (!a.held_role) a.held_role = name;
        else if (!a.target_role) a.target_role = name;
        state.newRole = ''; state.newPrompt = '';
        _flashEditorRender();
    });
    if (roles.length) {
        _flashEditorField(objects, 'Object', state.role, role => { _flashEditorMaskRole(role); _flashEditorRender(); }, roles);
        _flashEditorField(objects, 'Segmentation prompt', a.prompts_by_role[state.role], value => {
            a.prompts_by_role[state.role] = value;
        });
        const actionRoles = objects;
        _flashEditorField(actionRoles, 'Held object', a.held_role, value => {
            a.held_role = value; _flashEditorRender();
        }, roles);
        _flashEditorField(actionRoles, 'Target object', a.target_role, value => {
            a.target_role = value; _flashEditorRender();
        }, roles);
        _flashEditorButton(state.raw ? objects : addObject, 'Remove selected object', () => {
            delete a.prompts_by_role[state.role];
            delete a.axial_symmetry_roles?.[state.role];
            delete a.reference_polygons_by_role?.[state.role];
            delete a.reference_frames_by_role?.[state.role];
            state.role = Object.keys(a.prompts_by_role)[0] || '';
            if (!a.prompts_by_role[a.held_role]) a.held_role = state.role;
            if (!a.prompts_by_role[a.target_role]) a.target_role = '';
            _flashEditorRender();
        });
    }
    if (!state.raw) objects.append(addObject);
    const symmetry = a.axial_symmetry_roles || {};
    const selectedSymmetry = symmetry[state.role] || {};
    const objectOptions = _flashEditorSection(panes.objectOptions, 'Object rotation options');
    if (state.role) {
        const symmetryNote = document.createElement('p');
        symmetryNote.textContent = 'The selected axis is estimated from this object’s reference RGB-D points; the object name does not choose the axis.';
        objectOptions.append(symmetryNote);
    }
    if (state.role) _flashEditorField(objectOptions, 'Rotational symmetry', selectedSymmetry.axis_kind || 'none', value => {
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
            objectOptions.append(check);
        }
        _flashEditorField(objectOptions, 'Maximum observed tilt (degrees; blank = unlimited)',
            selectedSymmetry.max_observed_tilt_deg ?? '', value => {
                if (value === '') delete symmetry[state.role].max_observed_tilt_deg;
                else symmetry[state.role].max_observed_tilt_deg = Number(value);
            });
    }
    panes.objects.append(panes.references);
    const maskTools = _flashEditorSection(panes.objects, 'Mask',
        'Use the automatic mask, or correct it on the clear-view frame.');
    const polygonStatus = document.createElement('p');
    polygonStatus.id = 'flash-demo-editor-polygon-status';
    polygonStatus.hidden = !state.role;
    maskTools.append(polygonStatus);
    if (state.role) {
        if (state.drawing) {
            _flashEditorButton(maskTools, 'Save mask', () => {
                if (state.vertices.length < 3) {
                    showToast('Mask needs three points', 'Click at least three vertices.', 'warning'); return;
                }
                (a.reference_polygons_by_role ||= {})[state.role] = [...state.vertices];
                state.drawing = false; state.vertices = [];
                _flashEditorRender();
            });
            _flashEditorButton(maskTools, 'Cancel drawing', () => {
                state.drawing = false; state.vertices = [];
                _flashEditorRender();
            });
        } else {
            _flashEditorButton(maskTools, 'Correct mask', async () => {
                if (flashSceneMode === '3d') await toggleFlashSceneView();
                _flashEditorJump(_flashEditorReferenceFrame(state.role));
                state.drawing = true; state.vertices = [];
                _flashEditorRender();
            });
            if (a.reference_polygons_by_role?.[state.role]?.length) {
                _flashEditorButton(maskTools, 'Use automatic mask', () => {
                    delete a.reference_polygons_by_role[state.role];
                    _flashEditorRender();
                });
            }
        }
    }

    const keypoints = _flashEditorSection(panes.keyframes, 'Key frames');
    const points = state.keypointsDocument.keypoints;
    _flashEditorButton(keypoints, 'Add key frame at current time', () => {
        const time = _flashEditorCurrentFrame().snapshot.demo.time_s;
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
    if (points.length) {
        state.keypointIndex = Math.max(0, Math.min(state.keypointIndex, points.length - 1));
        const index = state.keypointIndex;
        const point = points[index];
        const turnRule = _flashEditorTurnForPoint(state, point);
        const navigation = document.createElement('div');
        navigation.className = 'flash-editor-keypoint-navigation';
        const previous = _flashEditorButton(navigation, '‹', () => {
            state.keypointIndex--;
            _flashEditorJumpTime(points[state.keypointIndex].t);
            _flashEditorRender();
        });
        previous.setAttribute('aria-label', 'Previous key frame');
        previous.disabled = index === 0;
        const position = document.createElement('strong');
        position.textContent = `Nearest ${index + 1}/${points.length} · ${point.t.toFixed(2)} s`;
        navigation.append(position);
        const next = _flashEditorButton(navigation, '›', () => {
            state.keypointIndex++;
            _flashEditorJumpTime(points[state.keypointIndex].t);
            _flashEditorRender();
        });
        next.setAttribute('aria-label', 'Next key frame');
        next.disabled = index === points.length - 1;
        keypoints.append(navigation);
        const details = document.createElement('div');
        details.className = 'flash-editor-keypoint';
        const startAnchored = Number.isFinite(Number(a.grasp_time_s)) &&
            Number.isFinite(Number(a.start_anchored_until_s)) &&
            point.t > Number(a.grasp_time_s) && point.t <= Number(a.start_anchored_until_s);
        const reference = _flashEditorField(details, 'Move relative to', point.frame || 'world',
            value => { point.frame = value; _flashEditorRender(); }, ['world', ...roles]);
        for (const option of reference.options) {
            if (option.value === 'world') option.textContent = 'World (fixed)';
        }
        reference.disabled = startAnchored || !!turnRule;
        if (!state.raw) {
            const pick = _flashEditorButton(details,
                state.pickingReference ? 'Cancel scene pick' : 'Pick in 3D scene',
                async () => {
                    if (state.pickingReference) {
                        state.pickingReference = false;
                        _flashEditorRender();
                        return;
                    }
                    if (flashSceneMode !== '3d') await toggleFlashSceneView();
                    state.pickingReference = flashSceneMode === '3d';
                    if (state.pickingReference) _flashEditorJumpTime(point.t);
                    _flashEditorRender();
                });
            pick.disabled = state.running || startAnchored || !!turnRule;
            if (startAnchored || state.pickingReference) {
                const note = document.createElement('p');
                note.textContent = startAnchored
                    ? 'This interval follows the saved start pose.'
                    : 'Click a measured object in the 3D scene to use it as this key frame’s reference.';
                details.append(note);
                if (startAnchored) _flashEditorButton(details, 'Change anchored interval', () => {
                    state.section = 'references';
                    _flashEditorRender();
                });
            }
        }
        const motion = _flashEditorField(details, 'Path', point.move,
            value => { point.move = value; _flashEditorRender(); }, ['line', 'follow', 'lift']);
        for (const option of motion.options) option.textContent = ({
            line: 'Straight connection', follow: 'Follow recording', lift: 'Lift then move',
        })[option.value] || option.value;
        if (!state.raw && !startAnchored && point.frame !== 'world') {
            const turn = document.createElement('div');
            turn.className = 'flash-editor-turn';
            details.append(turn);
            if (!turnRule || state.pickingTurn) {
                const lockedOrientation = !!a.axial_symmetry_roles?.[point.frame]?.lock_to_baseline;
                const invariant = flashReplay.pose_constraints?.[point.frame];
                const choose = _flashEditorButton(turn,
                    state.pickingTurn ? 'Cancel point pick' :
                        lockedOrientation ? 'Allow rotation around center' : 'Allow rotation around a point',
                    async () => {
                        if (state.pickingTurn) {
                            state.pickingTurn = false;
                        } else if (lockedOrientation) {
                            if (!invariant?.center_object_m || !invariant?.axis_object) return;
                            _flashEditorUseInvariantTurn(state, point, index);
                            if (flashSceneMode !== '3d') await toggleFlashSceneView();
                        } else {
                            if (flashSceneMode !== '3d') await toggleFlashSceneView();
                            state.pickingTurn = flashSceneMode === '3d';
                        }
                        _flashEditorRender();
                    });
                choose.disabled = state.running || (lockedOrientation &&
                    (!invariant?.center_object_m || !invariant?.axis_object));
                if (lockedOrientation && !state.pickingTurn) {
                    const note = document.createElement('p');
                    note.textContent = 'Uses the saved object center and direction. Check the marker in the 3D view.';
                    turn.append(note);
                }
                if (state.pickingTurn) {
                    const instruction = document.createElement('p');
                    instruction.textContent = `Click a measured point on ${point.frame}.`;
                    turn.append(instruction);
                }
            } else {
                const title = document.createElement('strong');
                title.textContent = turnRule.source_evidence === 'symmetry_invariant'
                    ? 'Turn around object center' : 'Turn around picked point';
                turn.append(title);
                const evidence = document.createElement('p');
                const fullTurn = Math.abs(turnRule.min_angle_rad + Math.PI) < 1e-6 &&
                    Math.abs(turnRule.max_angle_rad - Math.PI) < 1e-6;
                evidence.textContent = `${turnRule.reference_role} · ${fullTurn ? 'Any angle' : 'Limited angles'} · ` +
                    (({observed: 'Measured frame', symmetry_invariant: 'Object center',
                        predicted: 'Predicted frame'})[turnRule.source_evidence] || 'Unverified frame');
                turn.append(evidence);
                if (turnRule.source_evidence !== 'observed' &&
                    turnRule.source_evidence !== 'symmetry_invariant') {
                    const warning = document.createElement('p');
                    warning.className = 'flash-editor-status';
                    warning.textContent = a.axial_symmetry_roles?.[turnRule.reference_role]?.lock_to_baseline
                        ? 'This demo locks the object’s orientation to its starting direction. Its point frame is predicted, so Test cannot use this turn yet.'
                        : 'This object pose was predicted. Pick the point on a frame with a measured pose before testing.';
                    turn.append(warning);
                }
                if (turnRule.source_evidence === 'symmetry_invariant') {
                    const note = document.createElement('p');
                    note.textContent = 'The center and direction stay in place as this object spins.';
                    turn.append(note);
                } else {
                    if (a.axial_symmetry_roles?.[point.frame]?.lock_to_baseline &&
                        flashReplay.pose_constraints?.[point.frame]?.center_object_m)
                        _flashEditorButton(turn, 'Use stable center', () => {
                            _flashEditorUseInvariantTurn(state, point, index);
                            _flashEditorRender();
                        });
                    _flashEditorButton(turn, 'Change point', async () => {
                        if (flashSceneMode !== '3d') await toggleFlashSceneView();
                        state.pickingTurn = flashSceneMode === '3d';
                        _flashEditorRender();
                    });
                }
                _flashEditorButton(turn, 'Remove', () => {
                    const rules = state.keypointsDocument.turn_rules;
                    rules.splice(rules.indexOf(turnRule), 1);
                    _flashEditorRender();
                });
                const refine = document.createElement('details');
                refine.className = 'flash-editor-turn-refine';
                const refineTitle = document.createElement('summary');
                refineTitle.textContent = 'Refine turn';
                refine.append(refineTitle);
                _flashEditorField(refine, 'Minimum angle (degrees)',
                    (turnRule.min_angle_rad * 180 / Math.PI).toFixed(0), value => {
                        if (value === null || value > 0 || value * Math.PI / 180 >= turnRule.max_angle_rad) {
                            showToast('Invalid turn range', 'The minimum must include zero and stay below the maximum.', 'warning');
                            _flashEditorRender();
                            return;
                        }
                        turnRule.min_angle_rad = value * Math.PI / 180;
                    }, null, 'number');
                _flashEditorField(refine, 'Maximum angle (degrees)',
                    (turnRule.max_angle_rad * 180 / Math.PI).toFixed(0), value => {
                        if (value === null || value < 0 || value * Math.PI / 180 <= turnRule.min_angle_rad) {
                            showToast('Invalid turn range', 'The maximum must include zero and stay above the minimum.', 'warning');
                            _flashEditorRender();
                            return;
                        }
                        turnRule.max_angle_rad = value * Math.PI / 180;
                    }, null, 'number');
                const span = _flashEditorField(refine, 'Through key frame', String(turnRule.end_s), value => {
                    turnRule.end_s = Number(value);
                    _flashEditorRender();
                }, points.filter(item => item.t > turnRule.start_s).map(item => String(item.t)));
                for (const option of span.options) {
                    const keyIndex = points.findIndex(item => String(item.t) === option.value);
                    option.textContent = keyIndex < 0 ? option.value : `${keyIndex + 1} · ${Number(option.value).toFixed(2)} s`;
                }
                turn.append(refine);
            }
        }
        const exactKeyframe = document.createElement('details');
        exactKeyframe.className = 'flash-editor-keyframe-more';
        exactKeyframe.append(Object.assign(document.createElement('summary'), {textContent: 'More key frame options'}));
        _flashEditorField(exactKeyframe, 'Time (s)', point.t.toFixed(2), value => {
            if (!Number.isFinite(value)) return;
            point.t = value;
            position.textContent = `Nearest ${index + 1}/${points.length} · ${value.toFixed(2)} s`;
        }, null, 'number');
        _flashEditorButton(details, 'Move key frame to playhead', () => {
            point.t = _flashEditorCurrentFrame().snapshot.demo.time_s; _flashEditorRender();
        });
        const constraint = _flashEditorField(details, 'Reach', point.constraint || 'pose', value => {
            point.constraint = value;
        }, ['pose', 'best_effort']);
        constraint.options[0].textContent = 'Must reach';
        constraint.options[1].textContent = 'Best effort';
        if (point.move === 'lift') _flashEditorField(details, 'Lift (mm)', point.up_mm || 0, value => { point.up_mm = value; }, null, 'number');
        _flashEditorButton(exactKeyframe, 'Remove key frame', () => {
            if (turnRule) {
                showToast('Remove turn first', 'This key frame is part of a saved turn span.', 'warning');
                return;
            }
            if (state.keypointsDocument.keypoints.length <= 1) {
                showToast('Keep one key frame', 'The prepared skill needs at least one event.', 'warning');
                return;
            }
            state.keypointsDocument.keypoints.splice(index, 1);
            state.keypointIndex = Math.max(0, Math.min(index, state.keypointsDocument.keypoints.length - 1));
            _flashEditorRender();
            _flashRenderTimeline();
        });
        details.append(exactKeyframe);
        keypoints.append(details);
    }
    const advanced = document.createElement('div');
    advanced.className = 'flash-editor-advanced';
    if (state.raw) {
        const note = document.createElement('p');
        note.textContent = 'These are not inferred from the recording. Review them before preparation, or explicitly apply the starter values from prior SO-107 trials.';
        advanced.append(note);
        _flashEditorButton(advanced, 'Apply SO-107 starter parameters', () => {
            Object.assign(a, flashReplay.starter_parameters);
            _flashEditorRender();
        });
    }
    _flashEditorField(advanced, 'Vision pipeline (demo and run)', a.vision_pipeline?.backend || 'appearance', value => {
        a.vision_pipeline = {...(a.vision_pipeline || {}), backend: value};
    }, ['appearance', 'surface']);
    const correction = a.vision_pipeline?.detection_correction || {};
    _flashEditorField(advanced, 'Correct name-seeded masks with fresh detections',
        correction.enabled ? 'On' : 'Off', value => {
            a.vision_pipeline = {...(a.vision_pipeline || {}),
                detection_correction: {...(a.vision_pipeline?.detection_correction || {}), enabled: value === 'On'}};
        }, ['Off', 'On']);
    for (const [key, label, fallback] of [
        ['offline_interval_frames', 'Demo detection check (processed frames)', 16],
        ['runtime_interval_frames', 'Test detection check (processed frames)', 60],
    ]) _flashEditorField(advanced, label, correction[key] ?? fallback, value => {
        a.vision_pipeline = {...(a.vision_pipeline || {}),
            detection_correction: {...(a.vision_pipeline?.detection_correction || {}), [key]: value}};
    }, null, 'number');
    for (const [key, label] of [
        ['max_demo_pose_gap_frames', 'Maximum demo pose gap (frames)'],
        ['connection_speed_m_s', 'Connection speed (m/s)'],
        ['min_similarity', 'Descriptor similarity'],
        ['registration_threshold_m', 'Registration threshold (m)'],
        ['registration_min_inliers', 'Minimum registration inliers'],
        ['registration_trials', 'Registration trials'],
        ['sam_resolution', 'SAM resolution'],
    ]) _flashEditorField(advanced, label, a.vision_pipeline?.[key] ?? a[key], value => {
        a[key] = value;
        if (a.vision_pipeline && ['min_similarity', 'registration_threshold_m', 'registration_min_inliers', 'registration_trials', 'sam_resolution'].includes(key)) a.vision_pipeline[key] = value;
    }, null, 'number');
    _flashEditorField(advanced, 'Descriptor model', a.descriptor_model_id, value => {
        a.descriptor_model_id = value;
    });
    panes.advanced.append(advanced);

    const output = document.createElement('section');
    output.classList.add('flash-editor-footer');
    editor.append(output);
    const generate = _flashEditorButton(output, state.raw ? 'Prepare demo' : 'Preview updated demo',
        () => void _flashEditorRegenerate());
    generate.disabled = state.running;
    const message = document.createElement('p');
    message.id = 'flash-demo-editor-status';
    message.className = 'flash-editor-status';
    message.textContent = state.message || '';
    output.append(message);
    _flashEditorMaskRole(state.role);
    _flashEditorUpdateStatus();
    flashSceneView?.setTurnPreview(
        state.keypointsDocument.keypoints[state.keypointIndex]
            ? _flashEditorTurnForPoint(state, state.keypointsDocument.keypoints[state.keypointIndex]) || null
            : null,
    );
    if (!requestedSection) editor.scrollTop = previousScroll;
    if (requestedSection) requestAnimationFrame(() => {
        const target = editor.querySelector(`.flash-editor-pane[data-section="${requestedSection}"]`);
        if (target) editor.scrollTo({
            top: editor.scrollTop + target.getBoundingClientRect().top - editor.getBoundingClientRect().top,
            behavior: 'smooth',
        });
    });
}

function _flashEditorJumpTime(time) {
    const nearest = flashReplay.frames.reduce((best, frame, frameIndex) =>
        Math.abs(frame.snapshot.demo.time_s - time) < Math.abs(flashReplay.frames[best].snapshot.demo.time_s - time)
            ? frameIndex : best, 0);
    seekFlashReplay(nearest);
}

function _flashEditorTurnForPoint(state, point) {
    return (state.keypointsDocument.turn_rules || []).find(rule =>
        point.t > Number(rule.start_s) && point.t <= Number(rule.end_s));
}

function _flashEditorUseInvariantTurn(state, point, index) {
    const source = flashReplay.pose_constraints?.[point.frame];
    if (!source?.orientation_locked_to_baseline || !source.center_object_m || !source.axis_object)
        throw new Error('This object has no prepared stable center and direction');
    const rules = state.keypointsDocument.turn_rules ||= [];
    const prior = _flashEditorTurnForPoint(state, point);
    const rule = {
        rule_id: prior?.rule_id || `turn-${crypto.randomUUID()}`,
        reference_role: point.frame,
        start_s: prior?.start_s ?? (index ? state.keypointsDocument.keypoints[index - 1].t : 0),
        end_s: prior?.end_s ?? point.t,
        pivot_reference_m: source.center_object_m,
        axis_reference: source.axis_object,
        min_angle_rad: prior?.min_angle_rad ?? -Math.PI,
        max_angle_rad: prior?.max_angle_rad ?? Math.PI,
        source_frame_index: source.source_frame_index,
        source_evidence: 'symmetry_invariant',
    };
    if (prior) rules.splice(rules.indexOf(prior), 1, rule);
    else rules.push(rule);
    state.message = 'Turn added. Preview to check it.';
}

function flashDemoEditorPickTurn(picked) {
    const state = flashDemoEditor;
    if (!state?.pickingTurn) return;
    const points = state.keypointsDocument.keypoints;
    const index = state.keypointIndex;
    const point = points[index];
    if (!point || picked.role !== point.frame) {
        showToast('Pick the reference object', 'The pivot must lie on this key frame’s reference surface.', 'warning');
        return;
    }
    if (!picked.reference_xyz_m || !picked.reference_normal) {
        showToast('No local surface point', 'This object needs a usable pose at the selected frame.', 'warning');
        return;
    }
    const rules = state.keypointsDocument.turn_rules ||= [];
    const prior = _flashEditorTurnForPoint(state, point);
    const rule = {
        rule_id: prior?.rule_id || `turn-${crypto.randomUUID()}`,
        reference_role: point.frame,
        start_s: prior?.start_s ?? (index ? points[index - 1].t : 0),
        end_s: prior?.end_s ?? point.t,
        pivot_reference_m: picked.reference_xyz_m,
        axis_reference: picked.reference_normal,
        min_angle_rad: prior?.min_angle_rad ?? -Math.PI,
        max_angle_rad: prior?.max_angle_rad ?? Math.PI,
        source_frame_index: _flashEditorCurrentFrame().frame_index,
        source_evidence: picked.reference_pose_evidence || 'unverified',
    };
    if (prior) rules.splice(rules.indexOf(prior), 1, rule);
    else rules.push(rule);
    state.pickingTurn = false;
    state.message = picked.reference_pose_evidence === 'observed'
        ? `Turn pivot saved on ${point.frame}. Regenerate to check the compiled motion.`
        : `Pivot saved as a draft: ${point.frame} had a predicted pose at this frame.`;
    _flashEditorRender();
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
            result = {annotation, keypoints_document: {path_selection_version: 2, keypoints: []},
                reference_masks: {}, preparation_report: null};
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
            referenceReview: null, referenceReviewRunning: false,
            role: result.annotation.held_role,
            section: null,
            keypointIndex: result.keypoints_document.keypoints.reduce((best, point, index, points) =>
                Math.abs(point.t - _flashEditorCurrentFrame().snapshot.demo.time_s) <
                    Math.abs(points[best].t - _flashEditorCurrentFrame().snapshot.demo.time_s) ? index : best, 0),
            pickingReference: false,
            pickingTurn: false,
            raw, newRole: '', newPrompt: '',
            original: JSON.stringify([result.annotation, result.keypoints_document]),
            drawing: false, vertices: [], running: false,
            message: result.path_selection_migrated
                ? 'This older demo replayed a path at grasp despite its saved “straight” label. The editor now shows that path as “Follow recording”; preview keeps its motion.' : '',
            trackers, tracker: trackers.trackers.find(item => item.runnable)?.id || 'point2pose',
            growPoints: true,
            maxPoints: (trackers.trackers.find(item => item.id === 'trackeverything')?.initial_points_per_object || 128) * 3,
            pointsPerView: trackers.trackers.find(item => item.id === 'trackeverything')?.initial_points_per_object || 128,
            surfaceToleranceMm: Number(result.annotation.registration_threshold_m || 0.005) * 1000,
            trackerRole: '',
            trackerImportRole: result.annotation.held_role,
            trackerSource: '', trackerMessage,
            trackingConfigPath: raw ? (_val('run-flash-config') || '') : '',
        };
        const viewer = document.getElementById('rerun-viewer');
        viewer.classList.add('flash-demo-editing');
        document.getElementById('tab-run')?.classList.add('flash-demo-editing');
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

function flashDemoEditorPickReference(role) {
    const state = flashDemoEditor;
    if (!state?.pickingReference || !state.annotation.prompts_by_role[role]) return;
    const point = state.keypointsDocument.keypoints[state.keypointIndex];
    if (!point) return;
    point.frame = role;
    state.role = role;
    state.pickingReference = false;
    state.message = `Key frame ${state.keypointIndex + 1} now references ${role}. Regenerate to apply.`;
    _flashEditorRender();
    _flashRenderTimeline();
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
        if (roles.length < 2 || a.held_role === a.target_role ||
            !roles.includes(a.held_role) || !roles.includes(a.target_role)) {
            state.section = 'objects'; state.message = 'Name at least two objects and select distinct held and target objects.';
            _flashEditorRender(); return;
        }
        if (!a.repo_id?.trim() || !a.robot_arm || !a.calibration_path?.trim()) {
            state.section = 'setup'; state.message = 'Dataset identity, arm side and calibration file are required.';
            _flashEditorRender(); return;
        }
        if (!a.descriptor_model_id?.trim() || requiredNumbers.some(key => !Number.isFinite(a[key])) ||
            !(a.grasp_time_s > 0 && a.grasp_time_s < a.release_time_s) ||
            !(a.start_anchored_until_s >= a.grasp_time_s && a.start_anchored_until_s <= a.release_time_s) ||
            !Number.isInteger(a.reference_frame) || !Number.isInteger(a.grasp_reference_frame) ||
            roles.some(role => !Number.isInteger(_flashEditorReferenceFrame(role)))) {
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
                tracking_config_path: state.trackingConfigPath.trim() || null,
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
