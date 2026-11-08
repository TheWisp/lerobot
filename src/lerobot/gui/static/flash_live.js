/* Live acquisition/display only. The worker uses SurfaceFrameSession, as experiments do. */
const flashLive = {visible: false, timer: null, generation: 0, packet: null, overlay: null, bitmap: null,
    viewer: null, view: '2d', frame: null, previewDepth: null, depthColorRange: null,
    settings: null, processed: null, busy: false,
    historyActive: false, picking: false, playing: false, episodePlayer: null};
const liveElement = id => document.getElementById('flash-live-' + id);
async function liveRequest(path, body) {
    const response = await fetch('/api/run/flash/tracking/' + path, body === undefined ? {} :
        {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    if (!response.ok) { const error = await response.json(); throw new Error(typeof error.detail === 'string' ? error.detail : JSON.stringify(error.detail)); }
    return response.json();
}
async function showFlashLive() {
    if (flashLive.visible) return;
    const generation = ++flashLive.generation;
    try {
        setFlashSource(true);
        document.getElementById('flash-experiment-video').pause();
        flashLive.visible = true;
        liveElement('status').textContent = 'Loading shared pipeline settings…';
        flashLive.busy = true;
        liveElement('start').disabled = true;
        for(const id of ['matching','matcher','camera','resolution','history-seconds']) liveElement(id).disabled = true;
        const data = await liveRequest('shared-defaults');
        const current = await liveRequest('state');
        if (!flashLive.visible || generation !== flashLive.generation) return;
        flashLive.defaults = structuredClone(data.settings);
        if(current.running && current.config?.engine === 'experiment') {
            data.settings = current.config.pipeline_settings;
        }
        flashLive.settings = data.settings;
        if(current.config?.width && current.config?.height)
            liveElement('resolution').value=`${current.config.width}x${current.config.height}`;
        for(const key of ['spatial','temporal','hole-filling'])
            liveElement('depth-'+key).checked=!!current.config?.depth_processing?.[key.replace('-','_')];
        setFlashDepthSource(current.config?.depth_source||'sensor',true);
        updateFlashDepthFilterSummary();
        liveElement('matching').value = data.settings.shared_matching?.backend || 'lk';
        liveElement('matcher').replaceChildren(...Object.entries(data.matchers).map(([value,text]) => new Option(text,value)));
        liveElement('matcher').value = data.settings.config.shape.registration.backend;
        renderFlashPipelineSettings(data.fields, data.settings);
        liveElement('status').textContent = 'Press Start, then enter an object name or choose Pick object.';
        flashLive.busy = false;
        void refreshFlashLiveCameras();
        await refreshFlashHistory();
        if(!current.running && flashLive.historyFrames?.length)await selectFlashHistory('recent');
        await pollFlashLive();
    } catch (error) { liveElement('status').textContent = error.message; }
    finally { flashLive.busy = false; }
}
async function hideFlashLive(restore = true) {
    flashLive.generation++;
    clearTimeout(flashLive.timer);
    try {
        await flashLive.starting?.catch(() => {});
        await liveRequest('stop', {});
    } catch(error) { liveElement('status').textContent = error.message; pollFlashLive(); return false; }
    setFlashSource(false);
    flashLive.visible = false; flashLive.generation++; clearTimeout(flashLive.timer);
    flashLive.viewer?.dispose(); flashLive.viewer = null;
    stopFlashHistoryPlayback();
    closeFlashEpisode();
    flashLive.bitmap?.close(); flashLive.bitmap = null; flashLive.packet = null; flashLive.overlay = null;
    const selected = flashExperiments.items?.find(item => item.id === flashExperiments.selected);
    if(restore && selected) await selectFlashExperiment(selected);
    return true;
}
async function refreshFlashLiveCameras() {
    const generation = flashLive.generation, picker = liveElement('camera'), saved = picker.value;
    try {
        const data = await liveRequest('cameras', {});
        if (!flashLive.visible || generation !== flashLive.generation) return;
        picker.replaceChildren(new Option('Auto camera', ''),
            ...data.cameras.map(c => new Option(c.name || c.id, c.id)));
        picker.value = saved;
        if (!picker.value && data.cameras.length) picker.value = data.cameras[0].id;
    } catch(error) {
        if (flashLive.visible && generation === flashLive.generation)
            liveElement('status').textContent = `Camera discovery: ${error.message}`;
    }
}
async function startFlashLive() {
    if(flashLive.busy) return;
    flashLive.busy = true;
    const generation = flashLive.generation;
    try {
        const settings = readFlashPipelineSettings();
        settings.config.shape.registration.backend = liveElement('matcher').value;
        if(settings.shared_matching) settings.shared_matching.backend = liveElement('matching').value;
        else if(liveElement('matching').value === 'lk') throw new Error('Enable shared_matching in settings to use LK');
        flashLive.settings = settings;
        if(!liveElement('camera').value) await refreshFlashLiveCameras();
        if (!flashLive.visible || generation !== flashLive.generation) return;
        const picker = liveElement('camera');
        if(!picker.value) throw new Error('No RGB-D camera found');
        const depthProcessing=Object.fromEntries(['spatial','temporal','hole-filling'].map(key=>
            [key.replace('-','_'),liveElement('depth-'+key).checked]));
        liveElement('model-depth').checked=liveElement('depth-source').dataset.mode!=='sensor';
        const [width,height]=liveElement('resolution').value.split('x').map(Number);
        flashLive.starting = liveRequest('start', {engine: 'experiment', pipeline_settings: settings,
            serial: picker.value, width, height, fps:30,
            depth_processing:depthProcessing,
            depth_source:liveElement('depth-source').dataset.mode,
            history_seconds:Number(liveElement('history-seconds').value)});
        await flashLive.starting;
        if (!flashLive.visible || generation !== flashLive.generation) return;
        flashLive.processed = null; flashLive.processedMode=null; flashLive.packet = null; flashLive.overlay=null; flashLive.previewDepth=null;
        flashLive.depthColorRange=null; flashLive.frame = null;
        liveElement('objects').textContent='';
        flashLive.historyActive=false; liveElement('recording').value='recent'; flashLive.historyUpdated=0;
        updateFlashLiveMode();
        liveElement('status').textContent = 'Opening camera…';
    } catch(error) { liveElement('status').textContent = error.message; }
    finally { flashLive.busy = false; flashLive.starting = null; }
}
async function stopFlashLive() {
    try { await liveRequest('stop', {}); await refreshFlashHistory(); }
    catch(error) { liveElement('status').textContent = error.message; }
}
async function controlFlashLive(action, point) {
    try {
        if(point && (flashLive.historyActive || flashLive.frame?.frame_id == null))
            throw new Error('Pick an object in the live camera image');
        const command = {id: crypto.randomUUID(), action, prompt:liveElement('name').value,
            ...(point ? {point, frame_id:flashLive.frame.frame_id} : {})};
        await liveRequest('control', command);
        liveElement('status').textContent = action === 'clear' ? 'Clearing object…' : 'Selecting object…';
    } catch(error) { liveElement('status').textContent = error.message; }
}
function updateFlashDepthFilterSummary() {
    const enabled=['spatial','temporal','hole-filling'].filter(key=>liveElement('depth-'+key).checked);
    liveElement('depth-summary').textContent=`Camera depth filters: ${enabled.length?enabled.join(', ').replace('hole-filling','hole filling'):'off'}`;
}
function setFlashDepthSource(source,force=false) {
    if(flashLive.running && !force)return;
    const control=liveElement('depth-source');
    if(!['sensor','lingbot','foundation_stereo','lite_any_stereo_v2',
        'lite_any_stereo_v2_s','lite_any_stereo_v2_m','lite_any_stereo_v2_l'].includes(source))return;
    control.dataset.mode=source;
    for(const button of control.querySelectorAll('button'))
        button.setAttribute('aria-pressed',String(button.dataset.mode===source));
}
function showFlashCameraStreams(streams) {
    const color=streams.color,depth=streams.depth;
    if(!color||!depth)return;
    liveElement('stream-info').textContent=`Captured: color ${color.width} × ${color.height} @ ${color.fps} fps; `+
        `depth ${depth.width} × ${depth.height} @ ${depth.fps} fps, aligned to color.`;
}
async function controlFlashDepthFilters() {
    updateFlashDepthFilterSummary();
    if(!flashLive.running)return;
    const id=crypto.randomUUID();
    flashLive.depthFilterPendingId=id;
    const depth_processing=Object.fromEntries(['spatial','temporal','hole-filling'].map(key=>
        [key.replace('-','_'),liveElement('depth-'+key).checked]));
    try {await liveRequest('control',{id,action:'depth_filters',depth_processing});}
    catch(error){flashLive.depthFilterPendingId=null;liveElement('status').textContent=error.message;}
}
function setFlashLiveView(mode) {
    if(!['2d','3d','depth'].includes(mode))return;
    flashLive.view = mode;
    if(flashLive.episodePlayer && mode!=='2d')void loadFlashEpisodeDepth(flashLive.episodePlayer.frame());
    if(mode!=='2d' && flashLive.picking) toggleFlashLivePicker();
    const control=liveElement('view-mode');
    control.dataset.mode=mode;
    for(const button of control.querySelectorAll('button'))
        button.setAttribute('aria-pressed', String(button.dataset.mode===mode));
    liveElement('camera-view').hidden=mode!=='3d';
    liveElement('mask').closest('label').hidden=mode!=='2d';
    liveElement('surface').closest('label').hidden=mode==='depth';
    liveElement('filled-control').hidden=mode==='2d';
    liveElement('groups').closest('.flash-group-control').hidden=mode==='depth';
    liveElement('pick').disabled=mode!=='2d' || !flashLive.running || flashLive.historyActive;
    if((mode==='depth'||mode==='3d') && !flashLive.packet && flashLive.frame?.frame_id!=null && !flashLive.previewDepth) {
        const frame=flashLive.frame;
        void loadFlashPreviewDepth(frame).then(depth=>{
            if(depth && flashLive.frame?.frame_id===frame.frame_id && !flashLive.historyActive){
                flashLive.previewDepth=depth;drawFlashLive();
            }
        });
    }
    drawFlashLive();
}
async function loadFlashPreviewDepth(frame) {
    const response=await fetch('/api/run/flash/tracking/preview-depth');
    if(!response.ok)return null;
    const buffer=await response.arrayBuffer(), id=new DataView(buffer).getUint32(0,true);
    if(id!==frame.frame_id || flashLive.historyActive)return null;
    const expected=frame.width*frame.height;
    if(buffer.byteLength!==4+expected*2 && buffer.byteLength!==4+expected*4)return null;
    const raw=new Uint16Array(buffer,4,expected);
    const depth=new Float32Array(expected), scale=frame.depth_m_per_unit;
    for(let i=0;i<expected;i++)depth[i]=raw[i]*scale;
    const rawBefore=buffer.byteLength===4+expected*4?new Uint16Array(buffer,4+expected*2,expected):null;
    const depthBeforeFilters=rawBefore?new Float32Array(expected):null;
    if(rawBefore)for(let i=0;i<expected;i++)depthBeforeFilters[i]=rawBefore[i]*scale;
    return {depth,camera:{width:frame.width,height:frame.height,intrinsics:frame.intrinsics},
        depthBeforeFilters,
        depth_processing:frame.depth_processing,
        depth_filter_revision:frame.depth_filter_revision};
}
const flashDepthPalette=DepthChip.PALETTE.map(hex=>
    [1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)));
function flashDepthSource(packet) {
    const filters=Object.entries(packet.depth_processing||{}).filter(([,on])=>on).map(([name])=>name.replace('_',' '));
    const active=packet.depth_processing==null?'unknown (older recording)':filters.join(', ')||'off';
    const source=({lingbot:'LingBot-Depth-v0.5',foundation_stereo:'Fast-FoundationStereo',
        lite_any_stereo_v2_s:'Lite Any Stereo V2-S',lite_any_stereo_v2_m:'Lite Any Stereo V2-M',
        lite_any_stereo_v2_l:'Lite Any Stereo V2-L',lite_any_stereo_v2:'Lite Any Stereo V2-H'})[packet.depth_source]||'RealSense depth';
    return `aligned to color · Depth: ${source} · Camera filters: ${active}`+
        `${packet.depth_filter_revision!=null?` · revision ${packet.depth_filter_revision}`:''}`;
}
function drawFlashDepth(ctx, packet, canvas) {
    const depth=packet.depth, width=packet.camera.width, height=packet.camera.height;
    const before=packet.depthBeforeFilters, highlight=liveElement('filled').checked && !!before;
    if(depth.length!==width*height)throw new Error('Depth and image sizes differ');
    const sample=[];
    for(let y=0;y<height;y+=8)for(let x=0;x<width;x+=8){
        const z=depth[y*width+x];if(Number.isFinite(z)&&z>0)sample.push(z);
    }
    sample.sort((a,b)=>a-b);
    // One color scale per viewing session. Recomputing percentiles every frame
    // makes static depths flash different colors as the camera's valid pixels vary.
    if(!flashLive.depthColorRange && sample.length)
        flashLive.depthColorRange={near:sample[Math.floor((sample.length-1)*0.01)],
            far:sample[Math.floor((sample.length-1)*0.99)]};
    const {near,far}=flashLive.depthColorRange||{near:0,far:1};
    const span=Math.max(0.001,far-near), image=ctx.createImageData(width,height), pixels=image.data;
    let holes=0;
    for(let i=0;i<depth.length;i++){
        const z=depth[i], j=i*4;
        if(!Number.isFinite(z)||z<=0){holes++;pixels[j+3]=255;continue;}
        if(highlight && before[i]<=0){pixels[j]=255;pixels[j+1]=89;pixels[j+2]=191;pixels[j+3]=255;continue;}
        const position=Math.max(0,Math.min(10,(z-near)/span*10));
        const lo=Math.min(9,Math.floor(position)), t=position-lo;
        for(let channel=0;channel<3;channel++)
            pixels[j+channel]=flashDepthPalette[lo][channel]*(1-t)+flashDepthPalette[lo+1][channel]*t;
        pixels[j+3]=255;
    }
    canvas.width=width;canvas.height=height;ctx.putImageData(image,0,0);
    flashLive.depthSummary=`${near.toFixed(3)} m near → ${far.toFixed(3)} m far · `+
        `${(100*holes/depth.length).toFixed(1)}% holes · ${flashDepthSource(packet)}`+
        (highlight?' · pink = filter-created depth':'');
    liveElement('depth-info').textContent=flashLive.depthSummary;
}
function inspectFlashLiveDepth(event) {
    const packet=flashLive.packet||flashLive.previewDepth;
    if(flashLive.view!=='depth'||!packet)return;
    const canvas=event.currentTarget, rect=canvas.getBoundingClientRect();
    const scale=Math.min(rect.width/canvas.width,rect.height/canvas.height);
    const x=Math.floor((event.clientX-rect.left-(rect.width-canvas.width*scale)/2)/scale);
    const y=Math.floor((event.clientY-rect.top-(rect.height-canvas.height*scale)/2)/scale);
    if(x<0||y<0||x>=canvas.width||y>=canvas.height)return;
    const z=packet.depth[y*canvas.width+x];
    liveElement('depth-info').textContent=flashLive.depthSummary+
        ` · (${x}, ${y}) ${Number.isFinite(z)&&z>0?z.toFixed(4)+' m':'no measurement'}`;
}
function toggleFlashLivePicker() {
    flashLive.picking=!flashLive.picking;
    liveElement('pick').setAttribute('aria-pressed', String(flashLive.picking));
    liveElement('image').classList.toggle('flash-live-picking',flashLive.picking);
}
function projectLive(points, pose, k) {
    return points.map(p => {
        const q = pose.slice(0,3).map(r => r[0]*p[0]+r[1]*p[1]+r[2]*p[2]+r[3]);
        return q[2] > 0 ? [k[0][0]*q[0]/q[2]+k[0][2], k[1][1]*q[1]/q[2]+k[1][2]] : null;
    });
}
async function drawFlashLive() {
    const state=flashLive, packet=state.packet, overlay=state.overlay,
        image=state.episodePlayer ? state.episodeImage : state.bitmap;
    if(!image || !state.visible) return;
    const shownGroups = overlay ? flashSceneGroups(overlay.scene_group_tracks, overlay.objects,
        flashGroupMode('flash-live-groups')) : [];
    liveElement('group-count').textContent = overlay
        ? `${overlay.scene_group_tracks?.length || 0} groups · ${shownGroups.length} shown` : '';
    liveElement('groups').title = 'Connected measured points; group predictions remain in the recording.';
    // Saved episodes decode RGB and depth separately. An older live packet
    // must never override the depth for the episode frame on screen.
    const capturedDepthPacket=state.episodePlayer ? state.previewDepth : packet||state.previewDepth;
    // Live tracking packets carry the selected model depth as their primary
    // depth and the aligned sensor map as the preceding measurement. History
    // packets already carry these as separate sensor/model fields.
    const rawDepthPacket=capturedDepthPacket?.depth_source!=='sensor' &&
        capturedDepthPacket?.depthBeforeFilters && !capturedDepthPacket?.modelDepth
        ? {...capturedDepthPacket, depth:capturedDepthPacket.depthBeforeFilters,
           modelDepth:capturedDepthPacket.depth, depthBeforeFilters:null,
           depth_source:'sensor', tracking_depth_source:capturedDepthPacket.depth_source}
        : capturedDepthPacket;
    liveElement('model-depth-control').hidden=!rawDepthPacket?.modelDepth || state.view==='2d';
    const depthPacket=rawDepthPacket?.modelDepth && liveElement('model-depth').checked
        ? {...rawDepthPacket, depth:rawDepthPacket.modelDepth, depthBeforeFilters:rawDepthPacket.depth,
           depth_source:rawDepthPacket.tracking_depth_source} : rawDepthPacket;
    liveElement('filled').disabled=!depthPacket?.depthBeforeFilters;
    liveElement('filled-control').title=depthPacket?.depthBeforeFilters
        ? 'Pink points had no aligned sensor depth before filtering or model prediction.'
        : 'Pre-filter depth is unavailable for this frame (or camera filters are off).';
    const three=state.view==='3d' && !!depthPacket;
    liveElement('image').hidden=state.view==='3d'; liveElement('scene').hidden=!three;
    if(state.view==='3d' && !three)return;
    liveElement('depth-info').hidden=state.view==='2d'||!depthPacket;
    liveElement('depth-info').classList.toggle('flash-live-depth-info--scene',state.view==='3d');
    if(three)liveElement('depth-info').textContent=`Camera points: ${flashDepthSource(depthPacket)} · zero depth omitted`+
        (liveElement('filled').checked && depthPacket.depthBeforeFilters?' · pink = inferred depth where the sensor had a hole':'');
    if(state.view==='depth'){
        const canvas=liveElement('image'),ctx=canvas.getContext('2d');
        if(depthPacket)drawFlashDepth(ctx,depthPacket,canvas);
        else {ctx.fillStyle='#101923';ctx.fillRect(0,0,canvas.width,canvas.height);}
        return;
    }
    if(three) {
        const module=await import('/static/rgbd_scene.js?v=9');
        if(!state.visible || (state.packet||state.previewDepth)!==capturedDepthPacket) return;
        state.viewer ||= module.createRgbdScene(liveElement('scene'));
        const descriptions=[];
        for(const obj of Object.values(overlay?.objects||{})) {
            descriptions.push({layer:'estimate', points:obj.surface, pose:obj.pose, axisPoints:obj.axis_points, color:0xffa343});
        }
        const displayGroups=overlay?flashSceneGroups(overlay.scene_group_tracks, overlay.objects, flashGroupMode('flash-live-groups')):[];
        for(const group of displayGroups)
            descriptions.push({layer:'groups', points:flashGroupWirePoints(group).map(index =>
                (group.camera_xyz || group.display_camera_xyz || [])[index]),
                axes:false, pointSize:3, color:flashGroupHex(group.id || 0),
                pose:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]});
        state.viewer.update(depthPacket.depth,image,depthPacket.camera,descriptions,
            depthPacket.depthBeforeFilters);
        state.viewer.setOptions({estimate:liveElement('surface').checked,
            groups:flashGroupMode('flash-live-groups') !== 'none',axes:true,
            highlightFilled:liveElement('filled').checked});
        return;
    }
    const canvas=liveElement('image'), ctx=canvas.getContext('2d');
    canvas.width=image.width; canvas.height=image.height;
    ctx.drawImage(image,0,0);
    if(!overlay) return;
    for(const obj of Object.values(overlay.objects)) {
        if(liveElement('mask').checked) {
            ctx.strokeStyle='#65dcba'; ctx.lineWidth=2;
            for(const contour of obj.contours || []) {ctx.beginPath();contour.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke();}
        }
        if(liveElement('surface').checked && obj.pose) {
            ctx.fillStyle='#ffa343';
            for(const p of projectLive(obj.surface,obj.pose,overlay.camera.intrinsics)) if(p) ctx.fillRect(p[0]-1,p[1]-1,2,2);
            const axes=projectLive(obj.axis_points,obj.pose,overlay.camera.intrinsics);
            ['#ff6969','#6fe08a','#75a8ff'].forEach((color,i)=>{
                if(!axes[0] || !axes[i+1])return;
                ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(...axes[0]);ctx.lineTo(...axes[i+1]);ctx.stroke();
            });
        }
    }
    for(const group of shownGroups) {
        const pixels=group.pixels || group.display_pixels || [];
        ctx.strokeStyle=flashGroupColor(group.id || 0);ctx.fillStyle=ctx.strokeStyle;
        ctx.lineWidth=1.5;ctx.beginPath();
        for(const [a,b] of flashGroupLinks(group)) if(pixels[a]&&pixels[b]){ctx.moveTo(...pixels[a]);ctx.lineTo(...pixels[b]);}
        ctx.stroke();
        for(const index of flashGroupWirePoints(group)) {
            const [x,y]=pixels[index];ctx.beginPath();ctx.arc(x,y,2,0,2*Math.PI);ctx.fill();
        }
    }
}
async function pollFlashLive() {
    clearTimeout(flashLive.timer);
    if(!flashLive.visible) return;
    const generation=flashLive.generation;
    try {
        const data=await liveRequest('state');
        if(!flashLive.visible || generation!==flashLive.generation)return;
        const estimate=data.estimate || {}, running=data.running && data.config?.engine==='experiment';
        if(estimate.camera_streams)showFlashCameraStreams(estimate.camera_streams);
        flashLive.previewDepthProcessing=estimate.depth_processing||data.config?.depth_processing;
        flashLive.running=running; updateFlashPipelineControls(running);
        for(const button of liveElement('depth-source').querySelectorAll('button'))button.disabled=running;
        if(estimate.depth_processing && (!flashLive.depthFilterPendingId ||
            estimate.command_id===flashLive.depthFilterPendingId)){
            for(const key of ['spatial','temporal','hole-filling'])
                liveElement('depth-'+key).checked=!!estimate.depth_processing[key.replace('-','_')];
            flashLive.depthFilterPendingId=null;
            updateFlashDepthFilterSummary();
        }
        updateFlashLiveMode();
        for(const id of ['matching','matcher','camera','resolution','history-seconds']) liveElement(id).disabled=running;
        liveElement('start').disabled=data.running || flashLive.busy;
        liveElement('stop').disabled=!running;
        liveElement('pick').disabled=!running || flashLive.historyActive || flashLive.view!=='2d';
        liveElement('select').disabled=!running || flashLive.historyActive;
        liveElement('clear').disabled=!running;
        const timing=estimate.timings;
        const modelTime=estimate.depth_source && estimate.depth_source!=='sensor'
            ?`model depth ${(timing?.depth_ms||0).toFixed(0)} ms · `:'';
        if(running) liveElement('status').textContent=estimate.message || (
            estimate.status==='preview_model' ? `${modelTime}Pick an object to track` : timing?.processed_fps ?
            `${timing.processed_fps.toFixed(1)} fps · ${modelTime}mask ${timing.mask_ms.toFixed(0)} ms · tracking ${(timing.processing_ms-timing.mask_ms).toFixed(0)} ms · frame age ${timing.age_ms.toFixed(0)} ms` : estimate.status);
        else liveElement('status').textContent=data.estimate?.status==='error' ? data.estimate.message : 'Camera stopped. Press Start for a new tracking session.';
        if(!flashLive.historyActive && running && ['tracking','preview_model'].includes(estimate.status) &&
            (estimate.frame_id!==flashLive.processed || estimate.status!==flashLive.processedMode)) {
            const request=flashLive.historyRequest||0;
            const response=await fetch('/api/run/flash/tracking/scene');
            if(!response.ok) throw new Error('Processed scene unavailable');
            await displayFlashLivePacket(await response.arrayBuffer(), generation, request);
        }
        if(!flashLive.historyActive && running &&
            !(data.config?.depth_source && data.config.depth_source!=='sensor' && flashLive.packet)) {
            const frame=estimate.status==='frozen' || estimate.status==='loading_models' ? estimate.frame : data.preview;
            if(frame && frame.frame_id>(flashLive.frame?.frame_id ?? -1)) {
                const request=flashLive.historyRequest||0;
                const image=new Image();image.src=frame.image;await image.decode();
                const bitmap=await createImageBitmap(image);
                if(!flashLive.visible || generation!==flashLive.generation || request!==(flashLive.historyRequest||0)){bitmap.close();return;}
                const needsDepth=flashLive.view==='depth'||flashLive.view==='3d';
                const depth=needsDepth?await loadFlashPreviewDepth(frame):null;
                // Keep the last paired RGB-D frame when the camera advances
                // between the JSON snapshot and depth fetch. Never flash RGB
                // into Depth mode or combine depth with a different RGB frame.
                if(!flashLive.visible || generation!==flashLive.generation || request!==(flashLive.historyRequest||0)
                    || flashLive.historyActive || (needsDepth && !depth)){bitmap.close();return;}
                flashLive.bitmap?.close();flashLive.bitmap=bitmap;flashLive.packet=null;flashLive.previewDepth=depth;
                flashLive.frame={...frame,frozen:estimate.status==='frozen'};
                await drawFlashLive();
            } else if(frame)flashLive.frame.frozen=estimate.status==='frozen';
        }
        if(!flashLive.historyUpdated || Date.now()-flashLive.historyUpdated>2000) await refreshFlashHistory();
    } catch(error) { liveElement('status').textContent=error.message; }
    finally { if(flashLive.visible)flashLive.timer=setTimeout(pollFlashLive,150); }
}
function clickFlashLive(event) {
    if(!flashLive.picking || flashLive.historyActive || flashLive.frame?.frame_id == null)return;
    const canvas=liveElement('image'),rect=canvas.getBoundingClientRect();
    // object-fit: contain can letterbox; discard clicks in the bars.
    const scale=Math.min(rect.width/canvas.width,rect.height/canvas.height);
    const x=(event.clientX-rect.left-(rect.width-canvas.width*scale)/2)/scale;
    const y=(event.clientY-rect.top-(rect.height-canvas.height*scale)/2)/scale;
    if(x>=0 && y>=0 && x<canvas.width && y<canvas.height) {
        toggleFlashLivePicker();
        controlFlashLive('select',[x,y]);
    }
}

function rememberFlashOverlay(packet, recording) {
    flashLive.overlay={frame:packet.frame, camera:packet.camera, objects:packet.objects||{},
        scene_group_tracks:packet.scene_group_tracks||[]};
    flashLive.overlayRecording=recording;
    liveElement('objects').textContent=Object.values(flashLive.overlay.objects).map(o=>
        `${o.name}: ${o.status}${o.reason ? ` — ${o.reason}` : ''}`).join(' · ');
}
async function displayFlashLivePacket(buffer, generation, request, replay=false) {
    const n=new DataView(buffer).getUint32(0,true);
    const packet=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,n))), start=4+n;
    if(!flashLive.visible || generation!==flashLive.generation || request!==(flashLive.historyRequest||0))return;
    if(packet.result!=null || Object.keys(packet.objects||{}).length || packet.scene_group_tracks?.length)
        rememberFlashOverlay(packet,replay?liveElement('recording').value:'live');
    if(!replay){
        flashLive.processed=packet.frame;
        flashLive.processedMode=packet.preview_model?'preview_model':'tracking';
        if(packet.preview_model)flashLive.overlay=null;
        // The tracker may finish an older frame after a newer camera preview is
        // already visible. Update its overlay without jumping the image back.
        if(packet.frame<(flashLive.frame?.frame_id ?? -1) &&
            !(packet.depth_source && packet.depth_source!=='sensor')){await drawFlashLive();return;}
    }
    const measured=buffer.slice(start,start+packet.depth_bytes);
    if(packet.depth_format==='sensor_u16'){
        const raw=new Uint16Array(measured), scale=packet.depth_m_per_unit;
        packet.depth=new Float32Array(raw.length);
        for(let i=0;i<raw.length;i++)packet.depth[i]=raw[i]*scale;
    }else packet.depth=new Float32Array(measured);
    const beforeBytes=packet.depth_before_filter_bytes||0;
    if(beforeBytes){
        const rawBefore=new Uint16Array(buffer.slice(start+packet.depth_bytes,start+packet.depth_bytes+beforeBytes));
        packet.depthBeforeFilters=new Float32Array(rawBefore.length);
        const scale=packet.depth_before_filter_m_per_unit||packet.depth_m_per_unit||0.001;
        for(let i=0;i<rawBefore.length;i++)packet.depthBeforeFilters[i]=rawBefore[i]*scale;
    }
    const modelStart=start+packet.depth_bytes+beforeBytes+2*(packet.infrared_bytes||0);
    if(packet.model_depth_bytes){
        packet.modelDepth=new Float32Array(buffer.slice(modelStart,modelStart+packet.model_depth_bytes));
    }
    const rgb=buffer.slice(modelStart+(packet.model_depth_bytes||0));
    let bitmap;
    if(packet.rgb_format==='raw_rgb8'){
        const raw=new Uint8Array(rgb), rgba=new Uint8ClampedArray(packet.camera.width*packet.camera.height*4);
        for(let i=0,j=0;i<raw.length;i+=3,j+=4){
            rgba[j]=raw[i];rgba[j+1]=raw[i+1];rgba[j+2]=raw[i+2];rgba[j+3]=255;
        }
        bitmap=await createImageBitmap(new ImageData(rgba,packet.camera.width,packet.camera.height));
    }else bitmap=await createImageBitmap(new Blob([rgb],{type:packet.rgb_format==='jpeg'?'image/jpeg':'image/png'}));
    if(!flashLive.visible || generation!==flashLive.generation || request!==(flashLive.historyRequest||0)){bitmap.close();return;}
    flashLive.bitmap?.close();flashLive.bitmap=bitmap;flashLive.packet=packet;flashLive.previewDepth=null;
    flashLive.frame={frame_id:packet.frame, frozen:false};
    await drawFlashLive();
}
async function refreshFlashHistory() {
    if(flashLive.renaming)return;
    const recording=liveElement('recording').value||'recent';
    const data=await liveRequest('history?recording='+encodeURIComponent(recording));
    flashLive.historyUpdated=Date.now();
    if(recording!==liveElement('recording').value)return;
    flashLive.recordings=data.recordings;
    flashLive.overlayIndices=data.overlay_indices||[];
    flashLive.episodeCamera=data.episode_camera;
    if(flashLive.historyActive && data.camera_streams)showFlashCameraStreams(data.camera_streams);
    const list=liveElement('recordings');
    list.replaceChildren();
    for(const item of [{id:'recent', label:'Recent history', frames:data.recent_frames ?? (data.recording==='recent'?data.frames.length:0)},
        ...data.recordings.map(r=>({id:r.id,name:r.name||'',label:r.name||(r.objects||[]).join(', ')||'Saved clip',frames:r.frames}))]) {
        const row=document.createElement('div');row.className='flash-live-recording-row';row.setAttribute('role','listitem');
        const select=document.createElement('button');select.type='button';select.className='flash-live-recording-select';
        select.textContent=item.frames==null?item.label:`${item.label} · ${item.frames} frames`;
        select.title=select.textContent;
        select.setAttribute('aria-current',String(flashLive.historyActive && item.id===recording));
        select.onclick=()=>selectFlashHistory(item.id);
        const action=document.createElement('button');action.type='button';
        const remove=document.createElement('button');remove.type='button';remove.className='flash-live-delete';
        if(item.id==='recent'){
            action.id='flash-live-save-history';action.textContent='Save';action.title='Save recent history';
            action.onclick=()=>saveFlashHistory();action.disabled=!item.frames;
            remove.textContent='Clear';remove.setAttribute('aria-label','Clear recent history');
        }else{
            action.textContent='✎';action.title=`Rename ${item.label}`;
            action.setAttribute('aria-label',`Rename recording ${item.label}`);
            action.onclick=()=>editFlashHistoryName(item,row);
            remove.textContent='×';remove.title=`Delete ${item.label}`;
            remove.setAttribute('aria-label',`Delete recording ${item.label}`);
        }
        remove.onclick=()=>deleteFlashHistory(item.id);
        row.append(select,action,remove);list.append(row);
    }
    flashLive.historyFrames=data.frames.filter(f=>!f.context_only);
    const frames=flashLive.historyFrames, slider=liveElement('history-position');
    slider.max=Math.max(0,frames.length-1);slider.disabled=!frames.length;
    if(!flashLive.historyActive)slider.value=slider.max;
    else {const index=frames.findIndex(f=>f.frame===flashLive.frame?.frame_id);if(index>=0)slider.value=index;}
    const span=frames.length>1?frames.at(-1).time_s-frames[0].time_s:0;
    const missing=frames.filter(f=>f.missing_dependencies?.length).length;
    const gaps=(data.dropped_frames?` · ${data.dropped_frames} frames not logged`:'') +
        (missing?` · older source frames unavailable for ${missing} frames`:'');
    liveElement('history-info').textContent=`${frames.length} frames · ${span.toFixed(1)} seconds retained · ${((data.bytes||0)/1048576).toFixed(0)} MiB${gaps}${data.error?' · '+data.error:''}`;
    if(flashLive.historyActive && !flashLive.packet && !flashLive.episodePlayer && frames.length && recording==='recent')
        await loadFlashHistory(frames.length-1);
}
function editFlashHistoryName(item,row) {
    flashLive.renaming=item.id;
    const input=document.createElement('input');input.value=item.name||item.label;
    input.maxLength=80;input.setAttribute('aria-label','Recording name');
    const save=document.createElement('button');save.type='button';save.textContent='Save';
    const cancel=document.createElement('button');cancel.type='button';cancel.textContent='Cancel';
    const finish=()=>{flashLive.renaming=null;void refreshFlashHistory();};
    save.onclick=async()=>{
        const name=input.value.trim();if(!name){input.focus();return;}
        save.disabled=true;
        try {
            const response=await fetch('/api/run/flash/tracking/history/name',
                {method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({recording:item.id,name})});
            if(!response.ok){const error=await response.json();throw new Error(error.detail||'Could not rename recording');}
            finish();
        }catch(error){liveElement('history-info').textContent=error.message;save.disabled=false;}
    };
    cancel.onclick=finish;
    input.onkeydown=event=>{if(event.key==='Enter')void save.click();if(event.key==='Escape')finish();};
    row.replaceChildren(input,save,cancel);input.focus();input.select();
}
function closeFlashEpisode() {
    flashLive.episodePlayer?.close();
    flashLive.episodePlayer=null;
    flashLive.episodeHasPreFilterDepth=false;
    flashLive.episodeHasModelDepth=false;
    flashLive.episodeImage=null;
    flashLive.episodeDepthRequest=(flashLive.episodeDepthRequest||0)+1;
}
async function openFlashEpisode(recording) {
    const selected=flashLive.recordings?.find(item=>item.id===recording);
    if(!selected)return;
    const request=flashLive.historyRequest;
    liveElement('history-info').textContent='Preparing episode playback…';
    let root=selected.episode_root;
    if(!root){
        const exported=await liveRequest('history/episode',{recording});
        root=exported.episode_root;
    }
    const response=await fetch('/api/datasets',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({local_path:root})});
    if(!response.ok)throw new Error(`Could not open saved episode: ${await response.text()}`);
    const dataset=await response.json();
    flashLive.episodeHasPreFilterDepth=!!dataset.features_schema?.['observation.images.camera_depth_before_filters'];
    flashLive.episodeHasModelDepth=!!dataset.features_schema?.['observation.images.camera_depth_model'];
    if(request!==flashLive.historyRequest || !flashLive.historyActive || liveElement('recording').value!==recording)return;
    const camera=flashLive.episodeCamera || flashLive.overlay?.camera;
    const stored=dataset.features_schema?.['observation.images.camera']?.shape;
    const width=camera?.width || stored?.[1], height=camera?.height || stored?.[0];
    if(!width || !height)throw new Error('Saved episode has no camera dimensions');
    flashLive.episodeImage=document.createElement('canvas');
    flashLive.episodeImage.width=width;flashLive.episodeImage.height=height;
    const tile=document.createElement('canvas');
    const player=window.ChunkPlayer.create({datasetId:dataset.id,fps:dataset.fps,
        length:flashLive.historyFrames.length,cameras:['observation.images.camera'],
        tiles:()=>tile,
        onPaint:frame=>{if(flashLive.episodePlayer!==player)return;
            flashLive.episodeImage.getContext('2d').drawImage(tile,0,0,width,height);
            const row=flashLive.historyFrames[frame];
            flashLive.frame={frame_id:row.frame,frozen:false};
            // Never render this RGB frame against the preceding depth frame.
            flashLive.previewDepth=null;
            liveElement('history-position').value=frame;
            liveElement('play-time').textContent=`${(row.time_s-flashLive.historyFrames[0].time_s).toFixed(1)} s`;
            if(flashLive.view!=='2d')void loadFlashEpisodeDepth(frame);
            void loadFlashEpisodeOverlay(frame,recording);
            void drawFlashLive();
        },
        onHold:waiting=>{if(waiting)liveElement('history-info').textContent='Buffering episode…';},
        onLog:line=>console.debug('[FLASH episode]',line),
    });
    flashLive.episodePlayer=player;
    flashLive.episodeDataset=dataset.id;
    player.open(0);
    player.seek(flashLive.historyFrames.length-1);
    liveElement('history-info').textContent=`${flashLive.historyFrames.length} frames · LeRobot episode`;
}
async function loadFlashEpisodeOverlay(index,recording) {
    const indices=flashLive.overlayIndices||[];
    const source=indices.findLast(i=>i<=index);
    if(source===flashLive.episodeOverlaySource)return;
    flashLive.episodeOverlaySource=source;
    if(source==null){flashLive.overlay=null;liveElement('objects').textContent='';return;}
    const response=await fetch(`/api/run/flash/tracking/history/overlay?recording=${encodeURIComponent(recording)}&index=${index}`);
    if(!response.ok)return;
    const packet=await response.json();
    if(flashLive.episodeOverlaySource!==source || liveElement('recording').value!==recording)return;
    rememberFlashOverlay(packet,recording);
    void drawFlashLive();
}
async function loadFlashEpisodeDepth(index) {
    const player=flashLive.episodePlayer;if(!player)return;
    const request=++flashLive.episodeDepthRequest;
    const url=`/api/datasets/${encodeURIComponent(flashLive.episodeDataset)}/episodes/0/frame/${index}/depth?camera=observation.images.camera_depth`;
    try {
        const beforeUrl=`/api/datasets/${encodeURIComponent(flashLive.episodeDataset)}/episodes/0/frame/${index}/depth?camera=observation.images.camera_depth_before_filters`;
        const modelUrl=`/api/datasets/${encodeURIComponent(flashLive.episodeDataset)}/episodes/0/frame/${index}/depth?camera=observation.images.camera_depth_model`;
        const [response,beforeResponse,modelResponse]=await Promise.all([fetch(url),
            flashLive.episodeHasPreFilterDepth?fetch(beforeUrl):Promise.resolve(null),
            flashLive.episodeHasModelDepth?fetch(modelUrl):Promise.resolve(null)]);
        if(!response.ok)throw new Error(`Depth frame ${index}: HTTP ${response.status}`);
        if(beforeResponse && !beforeResponse.ok)throw new Error(`Pre-filter depth frame ${index}: HTTP ${beforeResponse.status}`);
        if(modelResponse && !modelResponse.ok)throw new Error(`Model depth frame ${index}: HTTP ${modelResponse.status}`);
        const raw=new Uint16Array(await response.arrayBuffer());
        const rawBefore=beforeResponse?new Uint16Array(await beforeResponse.arrayBuffer()):null;
        const rawModel=modelResponse?new Uint16Array(await modelResponse.arrayBuffer()):null;
        if(request!==flashLive.episodeDepthRequest || flashLive.episodePlayer!==player)return;
        const camera=flashLive.episodeCamera || flashLive.overlay?.camera;
        if(!camera)return;
        const depth=new Float32Array(raw.length);
        for(let i=0;i<raw.length;i++)depth[i]=raw[i]*0.001;
        const depthBeforeFilters=rawBefore?new Float32Array(rawBefore.length):null;
        if(rawBefore)for(let i=0;i<rawBefore.length;i++)depthBeforeFilters[i]=rawBefore[i]*0.001;
        const modelDepth=rawModel && rawModel.some(value=>value>0)?new Float32Array(rawModel.length):null;
        if(modelDepth)for(let i=0;i<rawModel.length;i++)modelDepth[i]=rawModel[i]*0.001;
        flashLive.previewDepth={depth,camera,
            depthBeforeFilters,
            modelDepth,
            tracking_depth_source:flashLive.historyFrames[index]?.tracking_depth_source,
            depth_processing:flashLive.historyFrames[index]?.depth_processing||null,
            depth_filter_revision:flashLive.historyFrames[index]?.depth_filter_revision};
        void drawFlashLive();
    }catch(error){liveElement('history-info').textContent=error.message;}
}
async function loadFlashHistory(index) {
    const frames=flashLive.historyFrames, row=frames?.[index];if(!row)return;
    if(flashLive.episodePlayer){flashLive.episodePlayer.seek(index);return;}
    flashLive.historyActive=true;
    updateFlashLiveMode();
    liveElement('history-position').value=index;
    const first=flashLive.historyFrames[0]?.time_s || row.time_s;
    liveElement('play-time').textContent=`${(row.time_s-first).toFixed(1)} s`;
    const token=(flashLive.historyRequest||0)+1;flashLive.historyRequest=token;
    const generation=flashLive.generation;
    const recording=liveElement('recording').value;
    try {
        const response=await fetch(`/api/run/flash/tracking/history/scene?recording=${encodeURIComponent(recording)}&frame=${row.frame}`);
        if(!response.ok)throw new Error('Frame expired; refresh recent history.');
        const buffer=await response.arrayBuffer();
        if(token!==flashLive.historyRequest || !flashLive.historyActive)return;
        const n=new DataView(buffer).getUint32(0,true);
        const header=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,n)));
        if(header.result==null){
            const prior=frames.slice(0,index).reverse().find(item=>Object.keys(item.objects||{}).length);
            if(prior && (flashLive.overlayRecording!==recording || flashLive.overlay?.frame!==prior.frame)){
                const source=await fetch(`/api/run/flash/tracking/history/scene?recording=${encodeURIComponent(recording)}&frame=${prior.frame}`);
                if(!source.ok)throw new Error('Earlier tracking result is unavailable.');
                const earlier=await source.arrayBuffer(), length=new DataView(earlier).getUint32(0,true);
                if(token!==flashLive.historyRequest || !flashLive.historyActive)return;
                rememberFlashOverlay(JSON.parse(new TextDecoder().decode(new Uint8Array(earlier,4,length))),recording);
            }else if(!prior){flashLive.overlay=null;liveElement('objects').textContent='';}
        }
        await displayFlashLivePacket(buffer,generation,token,true);
    } catch(error) {liveElement('history-info').textContent=error.message;}
}
async function selectFlashHistory(recording) {
    stopFlashHistoryPlayback();
    closeFlashEpisode();
    flashLive.episodeOverlaySource=undefined;
    flashLive.historyRequest=(flashLive.historyRequest||0)+1;
    if(recording)liveElement('recording').value=recording;
    flashLive.historyActive=true;flashLive.historyUpdated=0;flashLive.packet=null;
    flashLive.bitmap?.close();flashLive.bitmap=null;flashLive.frame=null;flashLive.previewDepth=null;
    flashLive.overlay=null;flashLive.overlayRecording=null;liveElement('objects').textContent='';
    flashLive.depthColorRange=null;
    updateFlashLiveMode();
    await refreshFlashHistory();
    const last=flashLive.historyFrames.length-1;
    liveElement('history-position').value=last;
    if(last>=0 && recording!=='recent'){
        try{await openFlashEpisode(recording);}
        catch(error){liveElement('history-info').textContent=error.message;}
    }else if(last>=0)await loadFlashHistory(last);
}
function followFlashLive() {
    stopFlashHistoryPlayback();
    closeFlashEpisode();
    flashLive.historyRequest=(flashLive.historyRequest||0)+1;
    flashLive.historyActive=false;flashLive.processed=null;flashLive.depthColorRange=null;
    flashLive.overlay=null;flashLive.overlayRecording=null;liveElement('objects').textContent='';
    liveElement('recording').value='recent';flashLive.historyUpdated=0;
    updateFlashLiveMode();
}
function updateFlashLiveMode() {
    liveElement('playback').hidden=!flashLive.historyActive;
    liveElement('return-camera').hidden=!flashLive.historyActive || !flashLive.running;
    liveElement('pick').disabled=flashLive.historyActive || !flashLive.running || flashLive.view!=='2d';
    liveElement('select').disabled=flashLive.historyActive || !flashLive.running;
    if(flashLive.historyActive && flashLive.picking)toggleFlashLivePicker();
}
function stopFlashHistoryPlayback() {
    flashLive.playing=false;clearTimeout(flashLive.playTimer);
    flashLive.episodePlayer?.pause();
    if(liveElement('play'))liveElement('play').textContent='Play';
}
async function playFlashHistory() {
    if(flashLive.playing){stopFlashHistoryPlayback();return;}
    if(!flashLive.historyFrames?.length)return;
    if(flashLive.episodePlayer){
        if(flashLive.episodePlayer.frame()>=flashLive.historyFrames.length-1)flashLive.episodePlayer.seek(0);
        flashLive.playing=true;liveElement('play').textContent='Pause';flashLive.episodePlayer.play();return;
    }
    flashLive.playing=true;liveElement('play').textContent='Pause';
    if(Number(liveElement('history-position').value)>=flashLive.historyFrames.length-1)
        await loadFlashHistory(0);
    const frames=flashLive.historyFrames;
    const firstTime=frames[0].time_s;
    const startedAt=performance.now()-(frames[Number(liveElement('history-position').value)].time_s-firstTime)*1000;
    const advance=async()=>{
        if(!flashLive.playing || !flashLive.historyActive)return;
        const index=Number(liveElement('history-position').value), next=index+1;
        if(next>=flashLive.historyFrames.length){stopFlashHistoryPlayback();return;}
        await loadFlashHistory(next);
        if(flashLive.playing)
            flashLive.playTimer=setTimeout(advance,next+1<frames.length
                ?Math.max(0,startedAt+(frames[next+1].time_s-firstTime)*1000-performance.now()):0);
    };
    const next=Number(liveElement('history-position').value)+1;
    flashLive.playTimer=setTimeout(advance,Math.max(0,startedAt+(frames[next].time_s-firstTime)*1000-performance.now()));
}
async function deleteFlashHistory(recording) {
    try {
        stopFlashHistoryPlayback();
        const response=await fetch('/api/run/flash/tracking/history?recording='+encodeURIComponent(recording),{method:'DELETE'});
        if(!response.ok){const error=await response.json();throw new Error(error.detail || 'Could not delete recording');}
        if(recording===liveElement('recording').value){
            liveElement('recording').value='recent';
            flashLive.historyFrames=[];
            if(flashLive.historyActive)followFlashLive();
        }
        flashLive.historyUpdated=0;
        await refreshFlashHistory();
    } catch(error){liveElement('history-info').textContent=error.message;}
}
async function saveFlashHistory() {
    try {
        liveElement('save-history').disabled=true;
        const saved=await liveRequest('history/save',{});
        await selectFlashHistory(saved.id);
    } catch(error) {liveElement('history-info').textContent=error.message;}
}

document.addEventListener('DOMContentLoaded', () => {
    const handle=document.querySelector('.flash-live-section-resize');
    const setup=liveElement('setup'), sidebar=liveElement('sidebar');
    if(!handle || !setup || !sidebar)return;
    const resize=height=>{setup.style.flexBasis=Math.max(150,Math.min(height,sidebar.clientHeight-160))+'px';};
    handle.addEventListener('pointerdown', event=>{
        event.preventDefault();handle.setPointerCapture(event.pointerId);
        const startY=event.clientY, startHeight=setup.getBoundingClientRect().height;
        const move=e=>resize(startHeight+e.clientY-startY);
        const finish=()=>{handle.removeEventListener('pointermove',move);handle.removeEventListener('pointerup',finish);};
        handle.addEventListener('pointermove',move);handle.addEventListener('pointerup',finish);
    });
    handle.addEventListener('keydown',event=>{
        if(event.key==='ArrowUp' || event.key==='ArrowDown'){
            event.preventDefault();resize(setup.getBoundingClientRect().height+(event.key==='ArrowUp'?-20:20));
        }
    });
});
