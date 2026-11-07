/* Live acquisition/display only. The worker uses SurfaceFrameSession, as experiments do. */
const flashLive = {visible: false, timer: null, generation: 0, packet: null, bitmap: null,
    viewer: null, three: false, frame: null, settings: null, processed: null, busy: false,
    historyActive: false, picking: false, playing: false};
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
        for(const id of ['matching','matcher','camera','history-seconds']) liveElement(id).disabled = true;
        const data = await liveRequest('shared-defaults');
        const current = await liveRequest('state');
        if (!flashLive.visible || generation !== flashLive.generation) return;
        flashLive.defaults = structuredClone(data.settings);
        if(current.running && current.config?.engine === 'experiment') {
            data.settings = current.config.pipeline_settings;
        }
        flashLive.settings = data.settings;
        liveElement('matching').value = data.settings.shared_matching?.backend || 'lk';
        liveElement('matcher').replaceChildren(...Object.entries(data.matchers).map(([value,text]) => new Option(text,value)));
        liveElement('matcher').value = data.settings.config.shape.registration.backend;
        renderFlashPipelineSettings(data.fields, data.settings);
        liveElement('status').textContent = 'Open camera, then enter an object name or choose Pick object.';
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
    flashLive.bitmap?.close(); flashLive.bitmap = null; flashLive.packet = null;
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
        flashLive.starting = liveRequest('start', {engine: 'experiment', pipeline_settings: settings,
            serial: picker.value, width:1280, height:720, fps:30,
            history_seconds:Number(liveElement('history-seconds').value)});
        await flashLive.starting;
        if (!flashLive.visible || generation !== flashLive.generation) return;
        flashLive.processed = null; flashLive.packet = null; flashLive.frame = null;
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
function toggleFlashLive3D(three) {
    flashLive.three = three;
    if(three && flashLive.picking) toggleFlashLivePicker();
    const control=liveElement('view-mode');
    control.dataset.mode=three?'3d':'2d';
    for(const button of control.querySelectorAll('button'))
        button.setAttribute('aria-pressed', String(button.dataset.mode===(three?'3d':'2d')));
    liveElement('camera-view').hidden=!three;
    liveElement('pick').disabled=three || !flashLive.running || flashLive.historyActive;
    drawFlashLive();
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
    const state=flashLive, packet=state.packet, image=state.bitmap;
    if(!image || !state.visible) return;
    const shownGroups = packet ? flashSceneGroups(packet.scene_group_tracks, packet.objects,
        flashGroupMode('flash-live-groups')) : [];
    liveElement('group-count').textContent = packet
        ? `${packet.scene_group_tracks?.length || 0} groups · ${shownGroups.length} shown` : '';
    liveElement('groups').title = 'Connected measured points; group predictions remain in the recording.';
    const three=state.three && !!packet;
    liveElement('image').hidden=three; liveElement('scene').hidden=!three;
    if(three) {
        const module=await import('/static/rgbd_scene.js?v=7');
        if(!state.visible || state.packet!==packet) return;
        state.viewer ||= module.createRgbdScene(liveElement('scene'));
        const descriptions=[];
        for(const obj of Object.values(packet.objects)) {
            descriptions.push({layer:'estimate', points:obj.surface, pose:obj.pose, axisPoints:obj.axis_points, color:0xffa343});
        }
        const displayGroups=flashSceneGroups(packet.scene_group_tracks, packet.objects, flashGroupMode('flash-live-groups'));
        for(const group of displayGroups)
            descriptions.push({layer:'groups', points:flashGroupWirePoints(group).map(index =>
                (group.camera_xyz || group.display_camera_xyz || [])[index]),
                axes:false, pointSize:3, color:flashGroupHex(group.id || 0),
                pose:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]});
        state.viewer.update(packet.depth,image,packet.camera,descriptions);
        state.viewer.setOptions({estimate:liveElement('surface').checked,
            groups:flashGroupMode('flash-live-groups') !== 'none',axes:true});
        return;
    }
    const canvas=liveElement('image'), ctx=canvas.getContext('2d');
    canvas.width=image.width; canvas.height=image.height;
    ctx.drawImage(image,0,0);
    if(!packet) return;
    for(const obj of Object.values(packet.objects)) {
        if(liveElement('mask').checked) {
            ctx.strokeStyle='#65dcba'; ctx.lineWidth=2;
            for(const contour of obj.contours || []) {ctx.beginPath();contour.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke();}
        }
        if(liveElement('surface').checked && obj.pose) {
            ctx.fillStyle='#ffa343';
            for(const p of projectLive(obj.surface,obj.pose,packet.camera.intrinsics)) if(p) ctx.fillRect(p[0]-1,p[1]-1,2,2);
            const axes=projectLive(obj.axis_points,obj.pose,packet.camera.intrinsics);
            ['#ff6969','#6fe08a','#75a8ff'].forEach((color,i)=>{
                if(!axes[0] || !axes[i+1])return;
                ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(...axes[0]);ctx.lineTo(...axes[i+1]);ctx.stroke();
            });
        }
    }
    for(const group of flashSceneGroups(packet.scene_group_tracks, packet.objects, flashGroupMode('flash-live-groups'))) {
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
        flashLive.running=running; updateFlashPipelineControls(running);
        for(const id of ['matching','matcher','camera','history-seconds']) liveElement(id).disabled=running;
        liveElement('start').disabled=data.running || flashLive.busy;
        liveElement('stop').disabled=!running;
        liveElement('pick').disabled=!running || flashLive.historyActive || flashLive.three;
        liveElement('select').disabled=!running || flashLive.historyActive;
        liveElement('clear').disabled=!running;
        const timing=estimate.timings;
        if(running) liveElement('status').textContent=estimate.message || (timing ?
            `${timing.processed_fps.toFixed(1)} fps · mask ${timing.mask_ms.toFixed(0)} ms · tracking ${(timing.processing_ms-timing.mask_ms).toFixed(0)} ms · frame age ${timing.age_ms.toFixed(0)} ms` : estimate.status);
        else liveElement('status').textContent=data.estimate?.status==='error' ? data.estimate.message : 'Camera stopped. Open it to start a new tracking session.';
        if(!flashLive.historyActive && running && estimate.status==='tracking' && estimate.frame_id!==flashLive.processed) {
            const request=flashLive.historyRequest||0;
            const response=await fetch('/api/run/flash/tracking/scene');
            if(!response.ok) throw new Error('Processed scene unavailable');
            await displayFlashLivePacket(await response.arrayBuffer(), generation, request);
        } else if(!flashLive.historyActive && running && ['preview','frozen','loading_models','error'].includes(estimate.status)) {
            const frame=estimate.status==='frozen' || estimate.status==='loading_models' ? estimate.frame : data.preview;
            if(frame && (frame.frame_id!==flashLive.frame?.frame_id || flashLive.packet)) {
                const request=flashLive.historyRequest||0;
                const image=new Image();image.src=frame.image;await image.decode();
                const bitmap=await createImageBitmap(image);
                if(!flashLive.visible || generation!==flashLive.generation || request!==(flashLive.historyRequest||0)){bitmap.close();return;}
                flashLive.bitmap?.close();flashLive.bitmap=bitmap;flashLive.packet=null;
                flashLive.frame={...frame,frozen:estimate.status==='frozen'};await drawFlashLive();
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

async function displayFlashLivePacket(buffer, generation, request) {
    const n=new DataView(buffer).getUint32(0,true);
    const packet=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,n))), start=4+n;
    packet.depth=new Float32Array(buffer.slice(start,start+packet.depth_bytes));
    const bitmap=await createImageBitmap(new Blob([buffer.slice(start+packet.depth_bytes)],{type:'image/png'}));
    if(!flashLive.visible || generation!==flashLive.generation || request!==(flashLive.historyRequest||0)){bitmap.close();return;}
    flashLive.bitmap?.close();flashLive.bitmap=bitmap;flashLive.packet=packet;flashLive.processed=packet.frame;
    flashLive.frame={frame_id:packet.frame, frozen:false};
    liveElement('objects').textContent=Object.values(packet.objects).map(o=>
        `${o.name}: ${o.status}${o.reason ? ` — ${o.reason}` : ''}`).join(' · ');
    await drawFlashLive();
}
async function refreshFlashHistory() {
    const recording=liveElement('recording').value||'recent';
    const data=await liveRequest('history?recording='+encodeURIComponent(recording));
    flashLive.historyUpdated=Date.now();
    if(recording!==liveElement('recording').value)return;
    flashLive.recordings=data.recordings;
    const list=liveElement('recordings');
    list.replaceChildren();
    for(const item of [{id:'recent', label:'Recent history', frames:data.recording==='recent'?data.frames.filter(f=>!f.context_only).length:null},
        ...data.recordings.map(r=>({id:r.id,label:`${(r.objects || []).join(', ') || 'Saved clip'} · ${new Date(r.saved_at*1000).toLocaleString()}`,frames:r.frames}))]) {
        const row=document.createElement('div');row.className='flash-live-recording-row';row.setAttribute('role','listitem');
        const select=document.createElement('button');select.type='button';
        select.textContent=item.frames==null?item.label:`${item.label} · ${item.frames} frames`;
        select.setAttribute('aria-current',String(item.id===recording));
        select.onclick=()=>selectFlashHistory(item.id);
        const remove=document.createElement('button');remove.type='button';remove.className='flash-live-delete';
        remove.textContent=item.id==='recent'?'Clear':'Delete';
        remove.setAttribute('aria-label',item.id==='recent'?'Clear recent history':`Delete recording ${item.label}`);
        remove.onclick=()=>deleteFlashHistory(item.id);
        row.append(select,remove);list.append(row);
    }
    flashLive.historyFrames=data.frames.filter(f=>!f.context_only);
    const frames=flashLive.historyFrames, slider=liveElement('history-position');
    slider.max=Math.max(0,frames.length-1);slider.disabled=!frames.length;
    liveElement('save-history').disabled=!frames.length || recording!=='recent';
    if(!flashLive.historyActive)slider.value=slider.max;
    else {const index=frames.findIndex(f=>f.frame===flashLive.frame?.frame_id);if(index>=0)slider.value=index;}
    const span=frames.length>1?frames.at(-1).time_s-frames[0].time_s:0;
    const missing=frames.filter(f=>f.missing_dependencies?.length).length;
    const gaps=(data.dropped_frames?` · ${data.dropped_frames} frames not logged`:'') +
        (missing?` · older source frames unavailable for ${missing} frames`:'');
    liveElement('history-info').textContent=`${frames.length} frames · ${span.toFixed(1)} seconds retained · ${((data.bytes||0)/1048576).toFixed(0)} MiB${gaps}${data.error?' · '+data.error:''}`;
    if(flashLive.historyActive && !flashLive.packet && frames.length)await loadFlashHistory(frames.length-1);
}
async function loadFlashHistory(index) {
    const row=flashLive.historyFrames?.[index];if(!row)return;
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
        await displayFlashLivePacket(buffer,generation,token);
    } catch(error) {liveElement('history-info').textContent=error.message;}
}
async function selectFlashHistory(recording) {
    stopFlashHistoryPlayback();
    if(recording)liveElement('recording').value=recording;
    flashLive.historyActive=true;flashLive.historyUpdated=0;flashLive.packet=null;
    updateFlashLiveMode();
    await refreshFlashHistory();
    const last=flashLive.historyFrames.length-1;
    liveElement('history-position').value=last;
    if(last>=0)await loadFlashHistory(last);
}
function followFlashLive() {
    stopFlashHistoryPlayback();
    flashLive.historyRequest=(flashLive.historyRequest||0)+1;
    flashLive.historyActive=false;flashLive.processed=null;
    liveElement('recording').value='recent';flashLive.historyUpdated=0;
    updateFlashLiveMode();
}
function setFlashLiveMode(mode) {
    if(mode==='live')followFlashLive();
    else selectFlashHistory(liveElement('recording').value || 'recent');
}
function updateFlashLiveMode() {
    const mode=flashLive.historyActive?'replay':'live', control=liveElement('mode');
    control.dataset.mode=mode;
    for(const button of control.querySelectorAll('button'))
        button.setAttribute('aria-pressed',String(button.dataset.mode===mode));
    liveElement('playback').hidden=mode!=='replay';
    liveElement('pick').disabled=mode==='replay' || !flashLive.running || flashLive.three;
    liveElement('select').disabled=mode==='replay' || !flashLive.running;
    if(mode==='replay' && flashLive.picking)toggleFlashLivePicker();
}
function stopFlashHistoryPlayback() {
    flashLive.playing=false;clearTimeout(flashLive.playTimer);
    if(liveElement('play'))liveElement('play').textContent='Play';
}
async function playFlashHistory() {
    if(flashLive.playing){stopFlashHistoryPlayback();return;}
    if(!flashLive.historyFrames?.length)return;
    flashLive.playing=true;liveElement('play').textContent='Pause';
    const advance=async()=>{
        if(!flashLive.playing || !flashLive.historyActive)return;
        const index=Number(liveElement('history-position').value), next=index+1;
        if(next>=flashLive.historyFrames.length){stopFlashHistoryPlayback();return;}
        const dt=flashLive.historyFrames[next].time_s-flashLive.historyFrames[index].time_s;
        await loadFlashHistory(next);
        flashLive.playTimer=setTimeout(advance,Math.max(16,Math.min(500,dt*1000)));
    };
    if(Number(liveElement('history-position').value)>=flashLive.historyFrames.length-1)
        await loadFlashHistory(0);
    flashLive.playTimer=setTimeout(advance,16);
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
