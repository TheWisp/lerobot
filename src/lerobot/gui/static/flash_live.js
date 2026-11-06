/* Live acquisition/display only. The worker uses SurfaceFrameSession, as experiments do. */
const flashLive = {visible: false, timer: null, generation: 0, packet: null, bitmap: null,
    viewer: null, three: false, frame: null, settings: null, processed: null, busy: false};
const liveElement = id => document.getElementById('flash-live-' + id);
async function liveRequest(path, body) {
    const response = await fetch('/api/run/flash/tracking/' + path, body === undefined ? {} :
        {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    if (!response.ok) { const error = await response.json(); throw new Error(typeof error.detail === 'string' ? error.detail : JSON.stringify(error.detail)); }
    return response.json();
}
async function showFlashLive() {
    try {
        document.getElementById('flash-experiment-video').pause();
        flashLive.visible = true;
        document.getElementById('flash-experiment-playback').hidden = true;
        liveElement('panel').hidden = false;
        liveElement('status').textContent = 'Loading shared pipeline settings…';
        flashLive.busy = true;
        liveElement('start').disabled = true;
        for(const id of ['method','matching','matcher','config','camera']) liveElement(id).disabled = true;
        document.getElementById('flash-experiment-title').textContent = 'Live tracking';
        document.getElementById('flash-experiment-detail').hidden = true;
        const data = await liveRequest('shared-defaults');
        const current = await liveRequest('state');
        if(current.running && current.config?.engine === 'experiment') {
            data.settings = current.config.pipeline_settings;
            liveElement('method').value = current.config.method;
        } else liveElement('method').value = 'group_growing';
        flashLive.settings = data.settings;
        liveElement('matching').value = data.settings.shared_matching?.backend || 'lk';
        liveElement('matcher').replaceChildren(...Object.entries(data.matchers).map(([value,text]) => new Option(text,value)));
        liveElement('matcher').value = data.settings.config.shape.registration.backend;
        liveElement('config').value = JSON.stringify(data.settings, null, 2);
        liveElement('status').textContent = 'Open camera, then enter an object name or freeze and click. Shared pipeline defaults: LK and the current surface fitter.';
        flashLive.busy = false;
        await pollFlashLive();
    } catch (error) { liveElement('status').textContent = error.message; }
    finally { flashLive.busy = false; }
}
async function hideFlashLive(restore = true) {
    try { await liveRequest('stop', {}); }
    catch(error) { liveElement('status').textContent = error.message; return; }
    flashLive.visible = false; flashLive.generation++; clearTimeout(flashLive.timer);
    flashLive.viewer?.dispose(); flashLive.viewer = null;
    flashLive.bitmap?.close(); flashLive.bitmap = null; flashLive.packet = null;
    liveElement('panel').hidden = true;
    document.getElementById('flash-experiment-playback').hidden = false;
    document.getElementById('flash-experiment-detail').hidden = false;
    const selected = flashExperiments.items?.find(item => item.id === flashExperiments.selected);
    if(restore && selected) await selectFlashExperiment(selected);
}
async function startFlashLive() {
    if(flashLive.busy) return;
    flashLive.busy = true;
    try {
        const settings = JSON.parse(liveElement('config').value);
        settings.config.shape.registration.backend = liveElement('matcher').value;
        if(settings.shared_matching) settings.shared_matching.backend = liveElement('matching').value;
        else if(liveElement('matching').value === 'lk') throw new Error('Enable shared_matching in settings to use LK');
        flashLive.settings = settings;
        const cameras = await liveRequest('cameras', {});
        const picker = liveElement('camera'), saved = picker.value;
        picker.replaceChildren(...cameras.cameras.map(c => new Option(c.name || c.id, c.id)));
        if(saved) picker.value = saved;
        if(!picker.value) throw new Error('No RGB-D camera found');
        await liveRequest('start', {engine: 'experiment', pipeline_settings: settings,
            method: liveElement('method').value, serial: picker.value, width:1280, height:720, fps:30});
        flashLive.processed = null; flashLive.packet = null; flashLive.frame = null;
        liveElement('status').textContent = 'Opening camera…';
    } catch(error) { liveElement('status').textContent = error.message; }
    finally { flashLive.busy = false; }
}
async function stopFlashLive() {
    try { await liveRequest('stop', {}); }
    catch(error) { liveElement('status').textContent = error.message; }
}
async function controlFlashLive(action, point) {
    try {
        if(point && flashLive.frame?.frozen !== true) throw new Error('Freeze the image before clicking');
        const command = {id: crypto.randomUUID(), action, prompt:liveElement('name').value,
            ...(point ? {point, frame_id:flashLive.frame.frame_id} : {})};
        await liveRequest('control', command);
        liveElement('status').textContent = action === 'freeze' ? 'Freezing…' : 'Selecting object…';
    } catch(error) { liveElement('status').textContent = error.message; }
}
function toggleFlashLive3D() {
    flashLive.three = !flashLive.three;
    liveElement('3d').setAttribute('aria-pressed', String(flashLive.three));
    drawFlashLive();
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
    const three=state.three && !!packet;
    liveElement('image').hidden=three; liveElement('scene').hidden=!three;
    if(three) {
        const module=await import('/static/rgbd_scene.js?v=6');
        if(!state.visible || state.packet!==packet) return;
        state.viewer ||= module.createRgbdScene(liveElement('scene'));
        const descriptions=[];
        for(const obj of Object.values(packet.objects)) {
            descriptions.push({layer:'estimate', points:obj.surface, pose:obj.pose, axisPoints:obj.axis_points, color:0xffa343});
            const group=obj.group_tracks;
            if(group) descriptions.push({layer:'groups', points:group.display_camera_xyz || group.camera_xyz || [],
                links:group.links || [], axes:false, color:0xdf8cff, pose:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]});
        }
        state.viewer.update(packet.depth,image,packet.camera,descriptions);
        state.viewer.setOptions({estimate:liveElement('surface').checked,groups:liveElement('groups').checked,axes:true});
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
        if(liveElement('groups').checked) {
            const group=obj.group_tracks || {}, pixels=group.display_pixels || group.pixels || [];
            ctx.strokeStyle='#df8cff';ctx.lineWidth=1.5;ctx.beginPath();
            for(const [a,b] of group.links || []) {if(pixels[a]&&pixels[b]){ctx.moveTo(...pixels[a]);ctx.lineTo(...pixels[b]);}}ctx.stroke();
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
        for(const id of ['method','matching','matcher','config','camera']) liveElement(id).disabled=running;
        liveElement('start').disabled=data.running || flashLive.busy;
        liveElement('stop').disabled=!running;
        liveElement('freeze').disabled=!running;
        liveElement('select').disabled=!running;
        liveElement('clear').disabled=!running;
        const timing=estimate.timings;
        if(running) liveElement('status').textContent=estimate.message || (timing ?
            `${timing.processed_fps.toFixed(1)} fps · mask ${timing.mask_ms.toFixed(0)} ms · tracking ${(timing.processing_ms-timing.mask_ms).toFixed(0)} ms · frame age ${timing.age_ms.toFixed(0)} ms` : estimate.status);
        else liveElement('status').textContent=data.estimate?.status==='error' ? data.estimate.message : 'Camera stopped. Open it to start a new tracking session.';
        if(running && estimate.status==='tracking' && estimate.frame_id!==flashLive.processed) {
            const response=await fetch('/api/run/flash/tracking/scene');
            if(!response.ok) throw new Error('Processed scene unavailable');
            const buffer=await response.arrayBuffer(), n=new DataView(buffer).getUint32(0,true);
            const packet=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,n))), start=4+n;
            packet.depth=new Float32Array(buffer.slice(start,start+packet.depth_bytes));
            const bitmap=await createImageBitmap(new Blob([buffer.slice(start+packet.depth_bytes)],{type:'image/png'}));
            if(!flashLive.visible || generation!==flashLive.generation){bitmap.close();return;}
            flashLive.bitmap?.close();flashLive.bitmap=bitmap;flashLive.packet=packet;flashLive.processed=packet.frame;
            flashLive.frame={frame_id:packet.frame, frozen:false};
            liveElement('objects').textContent=Object.values(packet.objects).map(o=>`${o.name}: ${o.status}`).join(' · ');
            await drawFlashLive();
        } else if(running && ['preview','frozen','loading_models'].includes(estimate.status)) {
            const frame=estimate.status==='frozen' || estimate.status==='loading_models' ? estimate.frame : data.preview;
            if(frame && (frame.frame_id!==flashLive.frame?.frame_id || flashLive.packet)) {
                const image=new Image();image.src=frame.image;await image.decode();
                const bitmap=await createImageBitmap(image);
                if(!flashLive.visible || generation!==flashLive.generation){bitmap.close();return;}
                flashLive.bitmap?.close();flashLive.bitmap=bitmap;flashLive.packet=null;
                flashLive.frame={...frame,frozen:estimate.status==='frozen'};await drawFlashLive();
            } else if(frame)flashLive.frame.frozen=estimate.status==='frozen';
        }
    } catch(error) { liveElement('status').textContent=error.message; }
    finally { if(flashLive.visible)flashLive.timer=setTimeout(pollFlashLive,150); }
}
function clickFlashLive(event) {
    if(!flashLive.frame?.frozen)return;
    const canvas=liveElement('image'),rect=canvas.getBoundingClientRect();
    // object-fit: contain can letterbox; discard clicks in the bars.
    const scale=Math.min(rect.width/canvas.width,rect.height/canvas.height);
    const x=(event.clientX-rect.left-(rect.width-canvas.width*scale)/2)/scale;
    const y=(event.clientY-rect.top-(rect.height-canvas.height*scale)/2)/scale;
    if(x>=0 && y>=0 && x<canvas.width && y<canvas.height)controlFlashLive('select',[x,y]);
}
