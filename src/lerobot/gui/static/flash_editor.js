/* Demo authoring only: Run owns hardware, ChunkPlayer owns episode playback. */
const flashEditor = {visible:false, generation:0, dataset:null, document:null, source:null,
    player:null, frame:0, dirty:false, picking:null, reference:'world', rangeStart:null, busy:false, recording:null, mode:'live', library:{demos:[]}, rollouts:[], entry:null, view:'2d'};
const editorElement = id => document.getElementById('flash-editor-'+id);
async function editorRequest(url, body, method='POST') {
    const response=await fetch(url, body===undefined?{}:{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const value=await response.json();
    if(!response.ok)throw new Error(typeof value.detail==='string'?value.detail:JSON.stringify(value.detail||value));
    return value;
}
function editorStatus(text) {editorElement('status').textContent=text;}
async function editFlashDemoProfile(kind){
    editorElement('settings-dialog').close();
    switchTab('robot');
    await robotTabInit();
    const name=editorElement(kind).value;
    if(name)await selectProfile(kind,name);
}
function flashDemoSelection(doc){return JSON.stringify([doc.source,doc.camera,doc.reference_frame,doc.objects]);}
function editorChanged() {flashEditor.dirty=true;editorElement('save-state').textContent='Unsaved';renderFlashEditor();}
async function showFlashEditor() {
    if(flashLive.visible && !flashEditor.previewHome && !(await hideFlashLive(false)))return;
    document.getElementById('flash-experiment-video').pause();
    document.getElementById('flash-examples-workspace').hidden=true;
    document.getElementById('flash-live-workspace').hidden=true;
    editorElement('workspace').hidden=false;
    for(const id of ['examples','live','editor'])document.getElementById('flash-source-'+id).setAttribute('aria-pressed',String(id==='editor'));
    flashEditor.visible=true;
    history.replaceState(null,'','#/flash/editor');
    try {
        const [setup,defaults]=await Promise.all([
            editorRequest('/api/run/flash/editor/settings'),liveRequest('shared-defaults')]);
        flashEditor.setup=setup;flashEditor.defaults=defaults;
        await loadFlashEditorSettings();
        await refreshFlashDemoLibrary();
        await pollFlashDemo();
        if(flashEditor.mode==='demo'&&flashEditor.dataset)void pollFlashDemoTracking(true);
    }catch(error){editorStatus(error.message);}
    renderFlashEditor();
}
async function loadFlashEditorSettings(){
    const setup=flashEditor.setup;
    for(const [id,endpoint] of [['robot','profiles'],['teleop','teleop-profiles']]){
        const profiles=await editorRequest('/api/robot/'+endpoint);
        editorElement(id).replaceChildren(new Option('Select '+(id==='robot'?'robot':'teleoperator'),''),
            ...profiles.map(p=>new Option(`${p.name} (${p.type})`,p.name)));
        editorElement(id).value=setup[id]||'';
    }
    editorElement('fps').value=setup.fps;
    editorElement('depth-source').value=setup.depth_source;
    editorElement('matcher').replaceChildren(...Object.entries(flashEditor.defaults.matchers).map(([value,label])=>new Option(label,value)));
    editorElement('matcher').value=setup.pipeline_settings.config.shape.registration.backend;
    editorElement('matching').value=setup.pipeline_settings.shared_matching?.backend||'lk';
    flashEditor.settingsForm=createFlashPipelineForm(editorElement('settings-form'),flashEditor.defaults.fields,setup.pipeline_settings,()=>({
        matcher:editorElement('matcher').value,matching:editorElement('matching').value,live:true,running:!!flashEditor.runActive
    }));
    await checkFlashDemoGeometry();
}
async function openFlashEditorSettings(){
    await loadFlashEditorSettings();
    editorElement('settings-status').textContent='';editorElement('settings-dialog').showModal();
}
async function saveFlashEditorSettings(){
    try{
        const fps=editorElement('fps');if(!fps.reportValidity())return;
        const setup={robot:editorElement('robot').value,teleop:editorElement('teleop').value,
            camera:editorElement('setup-camera').value,fps:Number(fps.value),
            depth_source:editorElement('depth-source').value,pipeline_settings:flashEditor.settingsForm.read()};
        flashEditor.setup=await editorRequest('/api/run/flash/editor/settings',setup,'PUT');
        editorElement('settings-dialog').close();
        editorStatus('Setup saved.');renderFlashEditor();
        if(flashEditor.mode==='demo')void pollFlashDemoTracking(true);
    }catch(error){editorElement('settings-status').textContent=error.message;}
}
async function checkFlashDemoGeometry(){
    const name=editorElement('robot').value;
    flashEditor.geometryReady=false;
    editorElement('geometry').textContent=name?'Checking camera calibration…':'Select a calibrated robot.';
    editorElement('setup-camera').replaceChildren();
    if(!name)return;
    try{
        const profile=await _getProfileData('robot',name);
        const status=await editorRequest('/api/run/flash/editor/robot-geometry',profile);
        if(name!==editorElement('robot').value)return;
        flashEditor.geometryReady=status.ready;
        editorElement('geometry').textContent=status.ready?`Calibrated: ${status.cameras.join(', ')}`:status.reason;
        editorElement('setup-camera').replaceChildren(...(status.cameras||[]).map(c=>new Option(c,c)));
        if(status.cameras?.includes(flashEditor.setup.camera))editorElement('setup-camera').value=flashEditor.setup.camera;
    }catch(error){editorElement('geometry').textContent=error.message;}
    renderFlashEditor();
}
function hideFlashEditor() {
    flashEditor.visible=false;flashEditor.generation++;
    editorElement('workspace').hidden=true;
    flashEditor.player?.pause();
    clearTimeout(flashEditor.trackingTimer);
    restoreFlashDemoViewer();detachFlashEditorPreview();
}
function detachFlashEditorPreview(){
    if(!flashEditor.previewHome)return;
    liveElement('resolution-overlay').hidden=true;
    setFlashLiveTrajectory(null);
    flashLive.visible=false;flashLive.generation++;clearTimeout(flashLive.timer);
    stopFlashHistoryPlayback();closeFlashEpisode();
    flashEditor.previewHome.replaceWith(liveElement('panel'));flashEditor.previewHome=null;
    editorElement('preview-host').hidden=true;
}
async function stopFlashEditorPreview(){
    await editorRequest('/api/run/flash/editor/preview/stop',{});detachFlashEditorPreview();
}
function mountFlashEditorPreview(){
    liveElement('resolution-overlay').hidden=true;
    if(!flashEditor.previewHome){
        flashEditor.previewHome=document.createComment('shared FLASH live panel');
        const panel=liveElement('panel');panel.replaceWith(flashEditor.previewHome);
        editorElement('preview-host').append(panel);
    }
    setFlashLiveTrajectory(null);
    flashLive.visible=true;flashLive.historyActive=false;
    flashLive.bitmap?.close();flashLive.bitmap=null;flashLive.frame=null;
    flashLive.packet=null;flashLive.overlay=null;flashLive.processed=null;
    flashLive.processedMode=null;flashLive.previewDepth=null;flashLive.depthColorRange=null;
    editorElement('preview-host').hidden=false;editorElement('empty').hidden=true;
    flashLive.settings=flashEditor.setup.pipeline_settings;
    liveElement('matching').value=flashEditor.setup.pipeline_settings.shared_matching?.backend||'lk';
    liveElement('matcher').replaceChildren(...Object.entries(flashEditor.defaults.matchers).map(([value,label])=>new Option(label,value)));
    liveElement('matcher').value=flashEditor.setup.pipeline_settings.config.shape.registration.backend;
    renderFlashPipelineSettings(flashEditor.defaults.fields,flashEditor.setup.pipeline_settings);
    stopFlashHistoryPlayback();closeFlashEpisode();
    void pollFlashLive();
}
async function startFlashEditorPreview(){
    if(flashEditor.busy||flashEditor.recording)return;
    if(!flashEditor.setup.robot||!flashEditor.setup.camera){await openFlashEditorSettings();return;}
    flashEditor.busy=true;renderFlashEditor();
    try{
        await editorRequest('/api/run/flash/editor/preview',{});
        mountFlashEditorPreview();editorStatus('');
    }catch(error){editorStatus(error.message);}
    finally{flashEditor.busy=false;await pollFlashDemo();}
}
async function selectFlashEditorLive(){
    if(flashEditor.dirty && !(await saveFlashDemo()))return;
    flashEditor.player?.pause();flashEditor.mode='live';flashEditor.entry=null;renderFlashEditor();
    if(flashEditor.execution?.running)mountFlashEditorPreview();
    else if(!flashEditor.recording)await startFlashEditorPreview();
}
async function refreshFlashDemoLibrary(){
    [flashEditor.library,flashEditor.rollouts]=await Promise.all([
        editorRequest('/api/run/flash/editor/library'),editorRequest('/api/run/flash/editor/rollouts')]);
    renderFlashDemoLibrary();
}
function renderFlashDemoLibrary(){
    const library=flashEditor.library;
    const key=JSON.stringify([library,flashEditor.rollouts,flashEditor.mode,flashEditor.entry?.id,!!flashEditor.recording]);
    if(flashEditor.libraryRenderKey===key)return;
    flashEditor.libraryRenderKey=key;
    editorElement('live').setAttribute('aria-current',flashEditor.mode==='live'?'page':'false');
    editorElement('demos').replaceChildren();
    for(const entry of library.demos){
        const button=editorButton(entry.name,()=>openFlashDemoEntry(entry));button.className='flash-demo-nav';
        button.setAttribute('aria-current',flashEditor.entry?.id===entry.id&&flashEditor.mode==='demo'?'page':'false');
        button.disabled=!!flashEditor.recording;
        const date=document.createElement('small');date.textContent=new Date(entry.created_at*1000).toLocaleString();button.append(date);
        editorElement('demos').append(button);
    }
    if(!library.demos.length){const hint=document.createElement('p');hint.textContent='Record with teleop. Stop saves your demo here.';editorElement('demos').append(hint);}
    editorElement('rollouts').replaceChildren();
    for(const entry of flashEditor.rollouts){
        const button=editorButton(entry.name,()=>openFlashRollout(entry));button.className='flash-demo-nav';
        button.setAttribute('aria-current',flashEditor.entry?.id===entry.id&&flashEditor.mode==='rollout'?'page':'false');
        button.disabled=!!flashEditor.recording;
        const date=document.createElement('small');date.textContent=`${new Date(entry.created_at*1000).toLocaleString()}`;
        button.append(date);editorElement('rollouts').append(button);
    }
    if(!flashEditor.rollouts.length){const hint=document.createElement('p');hint.textContent='Applied runs are saved here automatically.';editorElement('rollouts').append(hint);}
    const selected=editorElement('apply-demo').value;
    editorElement('apply-demo').replaceChildren(new Option('Choose a demo',''),...library.demos.map(d=>new Option(d.name,d.id)));
    editorElement('apply-demo').value=selected;
}
async function openFlashDemoEntry(entry){
    if(!entry||flashEditor.recording)return;
    if(flashEditor.dirty && !(await saveFlashDemo()))return;
    try{
        if(flashEditor.execution?.running)detachFlashEditorPreview();else await stopFlashEditorPreview();
        if(await openFlashEditorDataset(entry.root,entry.episode)){
            flashEditor.entry=entry;
            if(flashEditor.document.name==='Untitled demo')flashEditor.document.name=entry.name;
            editorElement('name').value=flashEditor.document.name;renderFlashEditor();
        }
    }catch(error){editorStatus(error.message);}
}
async function keepFlashDemo(){
    if(!(await saveFlashDemo()))return;
    try{
        if(flashEditor.entry){
            flashEditor.library=await editorRequest('/api/run/flash/editor/library/save',{
                id:flashEditor.entry.id,name:flashEditor.document.name||'Untitled demo'});
            flashEditor.entry=flashEditor.library.demos.find(d=>d.id===flashEditor.entry.id);
            renderFlashEditor();
        }
    }catch(error){editorStatus(error.message);}
}
function restoreFlashDemoViewer(){
    if(flashEditor.viewerHome){
        flashEditor.viewerHome.replaceWith(document.getElementById('rerun-viewer'));
        flashEditor.viewerHome=null;
    }
}
async function showFlashDemoViewer(){
    if(flashEditor.viewerHome||!flashEditor.visible)return;
    await runTabInit();
    if(!flashEditor.visible||flashEditor.viewerHome)return;
    const viewer=document.getElementById('rerun-viewer');
    flashEditor.viewerHome=document.createComment('shared Run viewer');viewer.replaceWith(flashEditor.viewerHome);
    editorElement('live-host').append(viewer);editorElement('live-host').hidden=false;editorElement('image-host').hidden=true;
    startObsStreamViewer();
}
function flashEditorName() {if(flashEditor.document){flashEditor.document.name=editorElement('name').value.trim()||'Untitled demo';editorChanged();}}
async function renameFlashDemo(){flashEditorName();await keepFlashDemo();}
async function openFlashEditorDataset(path,episode=0) {
    if(flashEditor.dirty && !(await saveFlashDemo()))return;
    const generation=++flashEditor.generation;
    try {
        editorStatus('Opening recording…');
        const dataset=await editorRequest('/api/datasets',{local_path:path});
        if(generation!==flashEditor.generation)return;
        flashEditor.dataset=dataset;
        return await selectFlashEditorEpisode(episode);
    }catch(error){editorStatus(error.message);return false;}
}
async function selectFlashEditorEpisode(episode=0) {
    if(!flashEditor.dataset)return false;
    const generation=++flashEditor.generation;
    try {
        const data=await editorRequest(`/api/run/flash/editor/document?dataset_id=${encodeURIComponent(flashEditor.dataset.id)}&episode=${episode}`);
        if(generation!==flashEditor.generation)return false;
        flashEditor.viewer?.dispose();flashEditor.viewer=null;
        Object.assign(flashEditor,{mode:'demo',document:data.document,source:data.source,episode,dirty:false,frame:data.document.keyframes[0]?.frame||0,picking:null,reference:'world',rangeStart:null,overlay:null,view:'2d',displayResult:null,displaySchema:false,depthColorRange:null});
        editorElement('full-recording').checked=!data.document.keyframes.length;
        const preferredCamera='observation.images.'+flashEditor.setup?.camera;
        if(!data.document.revision && data.source.cameras[preferredCamera])data.document.camera=preferredCamera;
        editorElement('name').value=data.document.name==='Untitled demo'?'':data.document.name;
        const shape=data.source.cameras[data.document.camera]?.shape;
        const resolution=editorElement('resolution');
        resolution.textContent=shape?.[1]&&shape?.[0]?`${shape[1]} × ${shape[0]}`:'';
        resolution.hidden=!resolution.textContent;
        editorElement('save-state').textContent='';
        renderFlashEditor();const playable=openFlashDemoPlayer();
        void loadFlashDemoTrajectory();
        void pollFlashDemoTracking(true);
        if(playable)editorStatus(data.source.has_robot?'Select up to three objects, then mark keyframes.':'This replay contains no robot actions.');
        return true;
    }catch(error){editorStatus(error.message);return false;}
}
function openFlashDemoPlayer() {
    restoreFlashDemoViewer();
    flashEditor.player?.close();
    flashEditor.player=null;
    const dataset=flashEditor.dataset, source=flashEditor.source, camera=flashEditor.document.camera;
    const canvas=editorElement('image');canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
    editorElement('play').disabled=true;editorElement('position').disabled=true;
    editorElement('image-host').hidden=false;editorElement('live-host').hidden=true;
    if(!camera){editorStatus('This episode has no camera frames. Add a camera in Robot Setup before recording a demo.');return false;}
    if(source.cameras[camera].dtype!=='video'){
        editorStatus('This camera was stored as individual images. Record with video enabled to use the shared episode player.');return false;
    }
    const selection=flashDemoSelection(flashEditor.document), scene=flashEditor.view!=='2d';
    const player=ChunkPlayer.create({datasetId:dataset.id,fps:source.fps,length:source.length,cameras:[camera],profile:'native',
        loadFrameData:(start,count,signal,episode)=>loadFlashDemoDisplay(dataset.id,episode,start,count,scene,signal),
        tiles:()=>editorElement('image'),onPaint:(frame,episode,evidence)=>{
            if(flashEditor.player!==player)return;
            flashEditor.frame=frame;flashEditor.overlay=evidence;flashEditor.overlaySelection=selection;editorElement('position').value=frame;
            editorElement('time').textContent=`${(frame/source.fps).toFixed(2)} s · ${frame}`;
            showFlashDemoKeyframe();drawFlashDemoMarks();
        },onHold:held=>{
            if(flashEditor.player!==player)return;
            if(held){
                flashEditor.beforeBuffer=editorElement('status').textContent;
                editorStatus('Buffering recording…');
            }else if(editorElement('status').textContent==='Buffering recording…'){
                editorStatus(flashEditor.beforeBuffer||'');
            }
        },
        onLog:message=>console.debug('[FLASH editor]',message)});
    flashEditor.player=player;
    editorElement('position').disabled=false;editorElement('play').disabled=false;
    editorElement('image-host').hidden=false;editorElement('live-host').hidden=true;
    player.open(flashEditor.episode);updateFlashDemoPlaybackRange();player.seek(flashEditor.frame);
    return true;
}
function updateFlashDemoPlaybackRange(){
    const keys=flashEditor.document?.keyframes,source=flashEditor.source;
    if(!keys||!source)return;
    const full=editorElement('full-recording');
    full.disabled=!keys.length;
    const start=keys.length&&!full.checked?keys[0].frame:0;
    const end=keys.length&&!full.checked?keys.at(-1).frame:source.length-1;
    const slider=editorElement('position');slider.min=start;slider.max=end;
    editorElement('valid-span').textContent=keys.length?`Demo: ${(keys[0].frame/source.fps).toFixed(2)}–${(keys.at(-1).frame/source.fps).toFixed(2)} s`:'';
    flashEditor.player?.setRange(start,end+1);
    if(flashEditor.frame<start||flashEditor.frame>end){
        flashEditor.frame=Math.max(start,Math.min(end,flashEditor.frame));
        seekFlashDemo(flashEditor.frame);
    }
    drawFlashDemoTrajectory();
}
async function loadFlashDemoTrajectory(){
    const token=flashEditor.trajectoryRequest=(flashEditor.trajectoryRequest||0)+1;
    const {dataset,episode,document:doc}=flashEditor,camera=doc.camera;
    flashEditor.trajectory?.dispose();flashEditor.trajectory=null;
    const control=editorElement('trajectory'),label=editorElement('trajectory-tools');
    control.disabled=true;control.checked=true;label.title='Loading recorded actions…';
    try{
        const query=new URLSearchParams({dataset_id:dataset.id,episode,camera});
        const data=await editorRequest('/api/run/flash/editor/trajectory?'+query);
        if(token!==flashEditor.trajectoryRequest||flashEditor.dataset!==dataset||flashEditor.document?.camera!==camera)return;
        control.disabled=!data.points.length;
        label.title=data.points.length?'Demonstrated tool path; waypoint times are relative to playback':'No recorded actions';
        flashEditor.trajectory=new FlashTrajectory(editorElement('image-host'),data,time=>seekFlashDemo(Math.round(time*flashEditor.source.fps)));
        drawFlashDemoTrajectory();
    }catch(error){if(token===flashEditor.trajectoryRequest)label.title=error.message;}
}
function drawFlashDemoTrajectory(){
    const trajectory=flashEditor.trajectory;if(!trajectory||!flashEditor.source)return;
    const slider=editorElement('position');
    trajectory.render({enabled:editorElement('trajectory').checked&&!flashEditor.picking&&
        (flashEditor.view!=='3d'||!!flashEditor.overlay?.depth),view:flashEditor.view,
        camera:flashEditor.overlay?.camera||trajectory.data.camera,time:flashEditor.frame/flashEditor.source.fps,
        timeRange:[Number(slider.min)/flashEditor.source.fps,Number(slider.max)/flashEditor.source.fps],viewer:flashEditor.viewer});
}
async function loadFlashDemoDisplay(dataset,episode,start,count,scene,signal){
    const query=new URLSearchParams({dataset_id:dataset,episode,start,count,scene});
    const response=await fetch('/api/run/flash/editor/display?'+query,{signal});
    if(!response.ok){const error=await response.json();throw new Error(error.detail||'Cannot load demo display');}
    if(response.headers.get('Content-Type')?.includes('application/json'))return response.json();
    const buffer=await response.arrayBuffer(), view=new DataView(buffer), rows=[];
    let offset=0;
    while(offset<buffer.byteLength){
        const length=view.getUint32(offset,true);offset+=4;
        if(!length){rows.push(null);continue;}
        const stream=new Blob([buffer.slice(offset,offset+length)]).stream().pipeThrough(new DecompressionStream('gzip'));
        offset+=length;
        const bytes=await new Response(stream).arrayBuffer();
        const headerLength=new DataView(bytes).getUint32(0,true), payload=4+headerLength;
        const packet=JSON.parse(new TextDecoder().decode(bytes.slice(4,payload)));
        packet.depth=new Float32Array(bytes.slice(payload,payload+packet.depth_bytes));
        packet.frame=start+rows.length;
        rows.push(packet);
    }
    if(rows.length!==count)throw new Error('Incomplete demo display chunk');
    return rows;
}
async function setFlashDemoView(view){
    if(view!=='2d'&&!flashEditor.displaySchema)return;
    flashEditor.player?.pause();editorElement('play').textContent='Play';
    if(view==='3d'&&!flashEditor.viewer){
        const module=await import('/static/rgbd_scene.js?v=12');
        flashEditor.viewer=module.createRgbdScene(editorElement('scene'));
        flashEditor.viewer.setOverlayRenderer(drawFlashDemoTrajectory);
    }
    flashEditor.view=view;flashEditor.overlay=null;
    drawFlashDemoMarks();openFlashDemoPlayer();
}
function playFlashDemo(){const p=flashEditor.player;if(!p)return;const playing=p.state().playing;playing?p.pause():p.play();editorElement('play').textContent=playing?'Play':'Pause';}
function seekFlashDemo(frame){flashEditor.player?.pause();editorElement('play').textContent='Play';flashEditor.player?.seek(frame);}
function setFlashDemoReference(){
    if(flashEditor.document.objects.length){editorStatus('Remove the selected objects before changing their image.');return;}
    flashEditor.document.reference_frame=flashEditor.frame;editorChanged();
}
function addFlashDemoObject(){
    const doc=flashEditor.document;if(!doc||doc.objects.length>=3)return;
    const name=editorElement('object-name').value.trim();
    if(!name||name.includes('.')||doc.objects.some(o=>o.name===name)){editorStatus('Use a distinct object name without periods.');return;}
    if(!doc.objects.length)doc.reference_frame=flashEditor.frame;
    const id=[1,2,3].map(n=>'object_'+n).find(id=>!doc.objects.some(o=>o.id===id));
    doc.objects.push({id,name,point:null});editorElement('object-name').value='';editorChanged();
}
function pickFlashDemoObject(event){
    const doc=flashEditor.document, object=doc?.objects.find(o=>o.id===flashEditor.picking);
    if(!object||flashEditor.frame!==doc.reference_frame)return;
    const canvas=editorElement('image'),rect=canvas.getBoundingClientRect();
    object.point=[(event.clientX-rect.left)*canvas.width/rect.width,(event.clientY-rect.top)*canvas.height/rect.height];
    flashEditor.picking=null;canvas.style.cursor='';editorChanged();drawFlashDemoMarks();
}
function markFlashDemoKeyframe(){
    const doc=flashEditor.document;if(!doc)return;
    flashEditor.player?.pause();
    if(doc.keyframes.some(k=>k.frame===flashEditor.frame)){editorStatus('This image is already a keyframe.');return;}
    doc.keyframes.push({frame:flashEditor.frame,reference:flashEditor.reference,label:''});
    doc.keyframes.sort((a,b)=>a.frame-b.frame);editorChanged();
}
function startFlashDemoRange(){
    if(!flashEditor.document)return;
    flashEditor.player?.pause();
    flashEditor.rangeStart={frame:flashEditor.frame,reference:flashEditor.reference};
    renderFlashEditor();
}
function finishFlashDemoRange(){
    const doc=flashEditor.document,start=flashEditor.rangeStart,end=flashEditor.frame;
    if(!doc||!start)return;
    flashEditor.player?.pause();
    if(end<start.frame){editorStatus('Choose an end frame at or after the range start.');return;}
    const existing=new Map(doc.keyframes.map(k=>[k.frame,k]));
    for(let frame=start.frame;frame<=end;frame++){
        if(existing.has(frame)&&existing.get(frame).reference!==start.reference){
            editorStatus('This range overlaps keyframes with another reference. Remove that range first.');return;
        }
    }
    for(let frame=start.frame;frame<=end;frame++){
        if(!existing.has(frame))doc.keyframes.push({frame,reference:start.reference,label:''});
    }
    doc.keyframes.sort((a,b)=>a.frame-b.frame);flashEditor.rangeStart=null;editorChanged();
    editorStatus(`Marked every frame from ${start.frame} through ${end}.`);
}
function flashDemoKeyRanges(keys){
    const ranges=[];
    for(const key of keys){
        const last=ranges.at(-1);
        if(last&&last.end+1===key.frame&&last.reference===key.reference){last.end=key.frame;}
        else ranges.push({start:key.frame,end:key.frame,reference:key.reference});
    }
    return ranges;
}
function flashDemoReferenceName(reference){
    return reference==='world'?'World (robot base)':flashEditor.document.objects.find(o=>o.id===reference)?.name||reference;
}
function showFlashDemoKeyframe(){
    const frame=flashEditor.frame, key=flashEditor.document?.keyframes.find(k=>k.frame===frame);
    editorElement('current-key').textContent=key?`Keyframe ${frame} · Relative to ${flashDemoReferenceName(key.reference)}`:`Frame ${frame} · Not a keyframe`;
    for(const row of editorElement('keys').children){
        row.firstChild.setAttribute('aria-current',String(frame>=Number(row.dataset.start)&&frame<=Number(row.dataset.end)));
    }
}
function editorButton(text,callback){const b=document.createElement('button');b.type='button';b.textContent=text;b.onclick=callback;return b;}
function renderFlashEditor(){
    const doc=flashEditor.document, busy=flashEditor.busy;
    const live=flashEditor.mode==='live', rollout=flashEditor.mode==='rollout';
    editorElement('execution').hidden=!live||(!flashEditor.execution?.running&&flashEditor.execution?.returncode===undefined);
    editorElement('workspace').classList.toggle('flash-editor-rollout',rollout);
    editorElement('live-actions').hidden=!live;editorElement('demo-actions').hidden=live||rollout;
    editorElement('edit-view').hidden=live||rollout||!doc;
    editorElement('run-log').hidden=!rollout;
    editorElement('output-divider').hidden=!rollout||!editorElement('run-log').open;
    editorElement('live-host').hidden=!live||!flashEditor.recording;
    editorElement('preview-host').hidden=(!live&&!rollout)||!flashEditor.previewHome||!!flashEditor.recording;
    editorElement('empty').hidden=!live||!!flashEditor.recording||!!flashEditor.previewHome;
    editorElement('title').textContent=live?'Live':flashEditor.entry?.name||'Demo';
    editorElement('record').disabled=busy||!!flashEditor.recording||(!flashEditor.setup?.robot)||(!flashEditor.setup?.teleop)||!!flashEditor.trackingPid;
    editorElement('start').hidden=!!flashEditor.previewRunning||!!flashEditor.recording||!!flashEditor.execution?.running;
    editorElement('start').disabled=busy||!!flashEditor.trackingPid;
    editorElement('apply').disabled=busy||!!flashEditor.recording||!!flashEditor.trackingPid||!!flashEditor.execution?.running;
    editorElement('run-demo').disabled=editorElement('apply').disabled||!doc?.keyframes.length||!flashEditor.library.demos.some(d=>d.id===flashEditor.entry?.id);
    editorElement('run-demo').title='Apply this demo directly to the robot';
    if(flashEditor.execution?.running)editorElement('record').disabled=true;
    editorElement('save').disabled=!doc||busy||!flashEditor.dirty;
    editorElement('save').textContent='Save changes';
    renderFlashDemoLibrary();
    editorElement('add-object').disabled=!doc||!flashEditor.player||doc.objects.length>=3||busy;
    editorElement('track').disabled=!doc||!doc.objects.length||busy||!!flashEditor.trackingPid;
    editorElement('mark').disabled=!doc||!flashEditor.player||busy;
    editorElement('range-start').disabled=!doc||!flashEditor.player||busy;
    editorElement('range-end').disabled=!doc||!flashEditor.player||busy||!flashEditor.rangeStart;
    editorElement('range-cancel').hidden=!flashEditor.rangeStart;
    editorElement('range-hint').textContent=flashEditor.rangeStart?`Start: frame ${flashEditor.rangeStart.frame}. Seek to the end, then mark it.`:'Both endpoints included. Every frame is retained.';
    editorElement('reference').disabled=!doc||!flashEditor.player||!!doc.objects.length;
    editorElement('objects').replaceChildren();editorElement('keys').replaceChildren();editorElement('references').replaceChildren();
    if(!doc)return;

    updateFlashDemoPlaybackRange();

    editorElement('object-count').textContent=doc.objects.length+' / 3';
    editorElement('reference-hint').textContent=`Object image: frame ${doc.reference_frame}. Names detect objects; Pick selects an instance.`;
    for(const object of doc.objects){
        const row=document.createElement('div');row.className='flash-editor-object';
        const name=document.createElement('span');name.textContent=object.name;name.style.color=['#ffb45b','#69d4ee','#ad9fff'][Number(object.id.at(-1))-1];
        const state=document.createElement('small');state.dataset.objectState=object.id;name.append(state);
        const pick=editorButton(object.point?'Picked':'Pick',async()=>{if(flashEditor.view!=='2d')await setFlashDemoView('2d');flashEditor.picking=object.id;editorElement('full-recording').checked=true;updateFlashDemoPlaybackRange();seekFlashDemo(doc.reference_frame);editorElement('image').style.cursor='crosshair';editorStatus('Click '+object.name+' on the object image.');});
        const remove=editorButton('×',()=>{doc.objects=doc.objects.filter(o=>o!==object);if(flashEditor.reference===object.id)flashEditor.reference='world';editorChanged();});
        remove.setAttribute('aria-label','Remove '+object.name);remove.disabled=doc.keyframes.some(k=>k.reference===object.id);
        remove.title=remove.disabled?'Remove its keyframes first':'Remove object';row.append(name,pick,remove);editorElement('objects').append(row);
    }
    for(const choice of [{id:'world',name:'World'},...doc.objects]){
        const button=editorButton(choice.name,()=>{flashEditor.reference=choice.id;renderFlashEditor();});
        button.setAttribute('aria-pressed',String(flashEditor.reference===choice.id));editorElement('references').append(button);
    }
    editorElement('key-count').textContent=`${doc.keyframes.length} frames`;
    for(const range of flashDemoKeyRanges(doc.keyframes)){
        const row=document.createElement('div');row.className='flash-editor-key';
        row.dataset.start=range.start;row.dataset.end=range.end;
        const name=flashDemoReferenceName(range.reference);
        const first=(range.start/flashEditor.source.fps).toFixed(2),last=(range.end/flashEditor.source.fps).toFixed(2);
        const span=range.start===range.end?`${first} s`:`${first}–${last} s`;
        const seek=editorButton(span,()=>seekFlashDemo(range.start));
        const reference=document.createElement('strong');reference.textContent=`Relative to ${name}`;
        const count=document.createElement('small');count.textContent=range.start===range.end?`Keyframe ${range.start}`:`Frames ${range.start}–${range.end} · ${range.end-range.start+1} keyframes`;
        seek.append(reference,count);
        const remove=editorButton('×',()=>{doc.keyframes=doc.keyframes.filter(k=>k.frame<range.start||k.frame>range.end);editorChanged();});
        remove.setAttribute('aria-label',range.start===range.end?'Remove keyframe '+range.start:`Remove keyframes ${range.start} through ${range.end}`);
        row.append(seek,remove);editorElement('keys').append(row);
    }
    showFlashDemoKeyframe();drawFlashDemoMarks();
}
function drawFlashDemoMarks(){
    const canvas=editorElement('image'),marks=editorElement('marks'),doc=flashEditor.document;
    const view=flashEditor.view, scene=editorElement('scene'), depthCanvas=editorElement('depth');
    drawFlashDemoTrajectory();
    for(const mode of ['2d','3d','depth']){
        const button=editorElement('view-'+mode);
        button.setAttribute('aria-pressed',String(view===mode));button.disabled=mode!=='2d'&&!flashEditor.displaySchema;
        button.title=button.disabled?'Track objects to save processed depth and surfaces':'';
    }
    canvas.hidden=marks.hidden=view!=='2d';scene.hidden=view!=='3d';depthCanvas.hidden=view!=='depth';
    editorElement('scene-host').hidden=view!=='3d';
    editorElement('reset-view').hidden=view!=='3d';editorElement('depth-info').hidden=view!=='depth';
    editorElement('outline').closest('label').hidden=view!=='2d';
    for(const id of ['surface','pose','rejected'])editorElement(id).closest('label').hidden=view==='depth';
    if(marks.width!==canvas.width||marks.height!==canvas.height){marks.width=canvas.width;marks.height=canvas.height;}
    const host=editorElement('image-host').getBoundingClientRect();
    if(host.width && host.height && canvas.width && canvas.height){
        const scale=Math.min(host.width/canvas.width,host.height/canvas.height);
        canvas.style.width=(canvas.width*scale)+'px';canvas.style.height=(canvas.height*scale)+'px';
    }
    const rect=canvas.getBoundingClientRect();
    Object.assign(marks.style,{width:rect.width+'px',height:rect.height+'px',left:(rect.left-host.left)+'px',top:(rect.top-host.top)+'px'});
    const ctx=marks.getContext('2d');ctx.clearRect(0,0,marks.width,marks.height);
    const overlay=doc&&flashEditor.overlaySelection===flashDemoSelection(doc)&&flashEditor.overlay?.frame===flashEditor.frame?flashEditor.overlay:null;
    const groups=overlay?flashSceneGroups(overlay.scene_group_tracks,overlay.objects,
        flashGroupMode('flash-editor-groups')):[];
    for(const state of editorElement('objects').querySelectorAll('[data-object-state]')){
        const evidence=overlay?.objects[state.dataset.objectState];
        state.textContent=evidence?' · '+(evidence.status==='observed_candidate'?'observed':evidence.status):'';
        state.title=evidence?.reason||'';
    }
    if(view==='depth'){
        if(overlay?.depth)drawFlashDepth(depthCanvas.getContext('2d'),overlay,depthCanvas,{state:flashEditor,info:editorElement('depth-info'),highlight:false});
        else {depthCanvas.getContext('2d').clearRect(0,0,depthCanvas.width,depthCanvas.height);editorElement('depth-info').textContent='No processed depth on this frame';}
        return;
    }
    if(view==='3d'){
        scene.hidden=editorElement('scene-host').hidden=!overlay?.depth;
        if(!overlay?.depth||!flashEditor.viewer)return;
        const descriptions=[];
        for(const object of doc.objects){
            const evidence=overlay.objects[object.id];if(!evidence)continue;
            const color=['#ffb45b','#69d4ee','#ad9fff'][Number(object.id.at(-1))-1];
            descriptions.push({layer:'estimate',points:evidence.surface||[],surface:editorElement('surface').checked,pose:evidence.pose,
                axisPoints:evidence.axis_points,color:parseInt(color.slice(1),16),label:object.name});
            if(editorElement('rejected').checked)for(const [kind,pose] of Object.entries(flashRejectedPoses(evidence))){
                if(pose)descriptions.push({layer:kind==='fit'?'rejectedFit':'rejectedPrediction',points:evidence.surface_used||[],pose,
                    axes:false,color:kind==='fit'?0xff52d7:0x49bcff,label:`${object.name} (rejected ${kind})`});
            }
        }
        descriptions.push(...flashGroupSceneDescriptions(groups));
        flashEditor.viewer.update(overlay.depth,canvas,overlay.camera,descriptions);
        flashEditor.viewer.setOptions({scene:true,estimate:editorElement('surface').checked||editorElement('pose').checked,
            axes:editorElement('pose').checked,rejectedFit:editorElement('rejected').checked,
            rejectedPrediction:editorElement('rejected').checked,groups:flashGroupMode('flash-editor-groups')!=='none'});
        return;
    }
    if(overlay){
        const intrinsics=overlay.camera?.intrinsics||overlay.intrinsics;
        for(const object of doc.objects){
            const evidence=overlay.objects[object.id];if(!evidence)continue;
            const color=['#ffb45b','#69d4ee','#ad9fff'][Number(object.id.at(-1))-1];
            ctx.fillStyle=color;ctx.globalAlpha=.2;
            for(const [start,length] of editorElement('outline').checked?overlay.masks_rle?.[object.id]||[]:[]){
                let index=start,left=length;
                while(left>0){const x=index%marks.width,y=Math.floor(index/marks.width),count=Math.min(left,marks.width-x);ctx.fillRect(x,y,count,1);index+=count;left-=count;}
            }
            ctx.globalAlpha=1;
            if(editorElement('outline').checked){
                ctx.strokeStyle=color;ctx.lineWidth=2;
                for(const contour of evidence.contours||[]){ctx.beginPath();contour.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke();}
            }
            drawFlashObjectPose(ctx,evidence,intrinsics,object.name,{color,surface:editorElement('surface').checked,pose:editorElement('pose').checked});
            if(editorElement('rejected').checked)drawFlashRejectedPoses(ctx,evidence,intrinsics,object.name);
        }
        drawFlashGroups2D(ctx,groups);
    }
    if(!doc||flashEditor.frame!==doc.reference_frame)return;
    for(const object of doc.objects){if(!object.point)continue;const [x,y]=object.point;
        ctx.strokeStyle=ctx.fillStyle=['#ffb45b','#69d4ee','#ad9fff'][Number(object.id.at(-1))-1];
        ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,7,0,2*Math.PI);ctx.stroke();ctx.font='16px sans-serif';ctx.fillText(object.name,x+12,y-8);
    }
}
async function saveFlashDemo(){
    if(!flashEditor.document||flashEditor.busy)return false;
    flashEditor.busy=true;renderFlashEditor();
    // Keep the submitted snapshot separate: typing during a save must not be overwritten.
    const submitted=JSON.stringify(flashEditor.document),generation=flashEditor.generation;
    try{
        const saved=await editorRequest('/api/run/flash/editor/document',{dataset_id:flashEditor.dataset.id,episode:flashEditor.episode,document:JSON.parse(submitted)},'PUT');
        if(generation!==flashEditor.generation)return false;
        const changed=JSON.stringify(flashEditor.document)!==submitted;
        if(!changed)flashEditor.document=saved;else flashEditor.document.revision=saved.revision;
        flashEditor.dirty=changed;editorElement('save-state').textContent=changed?'Unsaved':'Saved';
        return !changed;
    }catch(error){editorStatus(error.message);return false;}
    finally{flashEditor.busy=false;renderFlashEditor();}
}
async function recordFlashDemo(){
    if(flashEditor.busy)return;
    if(flashEditor.dirty && !(await saveFlashDemo()))return;
    if(!flashEditor.setup?.robot||!flashEditor.setup?.teleop){await openFlashEditorSettings();return;}
    flashEditor.busy=true;renderFlashEditor();
    try{
        await stopFlashEditorPreview();
        // The normal recorder becomes the sole hardware owner, including its cameras.
        const response=await editorRequest('/api/run/flash/editor/record-current',{});
        flashEditor.player?.pause();flashEditor.mode='live';
        flashEditor.recording=response;
        await showFlashDemoViewer();editorStatus('Recording. Use teleoperation, then press Stop.');
    }catch(error){editorStatus(error.message);}
    finally{flashEditor.busy=false;void pollFlashDemo();}
}
async function stopFlashDemo(){
    editorElement('stop').disabled=true;
    try{
        if(flashEditor.rebuildPid){
            flashEditor.generation++;
            await editorRequest('/api/run/flash/editor/cancel-tracking',{pid:flashEditor.rebuildPid});
            editorStatus('Tracking stopped.');
        }else if(flashEditor.recording){
            editorStatus('Saving demo…');
            await editorRequest('/api/run/flash/editor/stop',{pid:flashEditor.recording.pid});
        }else{await stopFlashEditorPreview();editorStatus('Camera stopped.');}
    }catch(error){editorStatus(error.message);}
    await pollFlashDemo();
}
async function pollFlashDemo(){
    clearTimeout(flashEditor.timer);
    try{
        const data=await editorRequest('/api/run/flash/editor/recording'),status=data.run,own=data.recording;
        flashEditor.runActive=!!status.running;
        flashEditor.recording=own?.running?own:null;
        const wasExecuting=flashEditor.execution?.running;
        flashEditor.previewRunning=!!data.preview_running;
        flashEditor.execution=await editorRequest('/api/run/flash/editor/execution');
        await refreshFlashDemoLibrary();
        const execution=flashEditor.execution;
        liveElement('panel').classList.toggle('flash-demo-execution',!!execution.running);
        if(editorElement('status').textContent==='Failed to fetch')editorStatus('Connected.');
        editorElement('execution').hidden=flashEditor.mode!=='live'||(!execution.running&&execution.returncode===undefined);
        const executionState=execution.state==='complete'?'path complete — check placement':execution.state;
        editorElement('execution').textContent=execution.running?
            `${execution.mode==='robot'?'Robot':'Test run · no motion'} · ${executionState} · ${Math.min((execution.keyframe||0)+1,execution.total||1)} / ${execution.total||'…'}`:
            (execution.returncode===0?'Run stopped.':execution.reason||'Run ended.');
        if(wasExecuting&&!execution.running&&flashEditor.mode==='live')editorStatus('Rollout saved. Select it under Rollouts to inspect the run.');
        editorElement('stop').disabled=!own?.running&&!data.preview_running&&!execution.running&&!flashEditor.rebuildPid;
        for(const id of ['robot','teleop','setup-camera','fps','matching','matcher','depth-source','settings-save'])editorElement(id).disabled=!!status.running;
        flashEditor.settingsForm?.update();
        if(own?.running){flashEditor.mode='live';await showFlashDemoViewer();}
        else if(data.preview_running && flashEditor.visible && flashEditor.mode==='live' && !flashEditor.previewHome)mountFlashEditorPreview();
        if(own && !own.running && flashEditor.completedPid!==own.pid){
            flashEditor.completedPid=own.pid;restoreFlashDemoViewer();
            if(own.returncode===0){await openFlashDemoEntry(flashEditor.library.demos.find(d=>d.id===own.id));}
            else editorStatus('Recording failed. Open Run to see the error.');
        }
        renderFlashEditor();
    }catch(error){editorStatus(error.message);}
    if(flashEditor.visible||flashEditor.recording)flashEditor.timer=setTimeout(pollFlashDemo,1000);
}
window.addEventListener('resize',drawFlashDemoMarks);

async function runSelectedFlashDemo(){
    if(flashEditor.busy||!flashEditor.entry)return;
    if(flashEditor.dirty&&!(await saveFlashDemo()))return;
    await applyFlashDemo(flashEditor.entry.id);
}
async function applyFlashDemo(id=editorElement('apply-demo').value){
    if(flashEditor.busy)return;
    if(!id){editorStatus('Choose a demo to apply.');return;}
    if(flashEditor.dirty&&!(await saveFlashDemo()))return;
    flashEditor.busy=true;renderFlashEditor();editorStatus('Checking demo tracking…');
    const generation=flashEditor.generation;
    try{
        if(flashEditor.execution?.running)throw new Error('Stop the current run before applying another demo.');
        await stopFlashEditorPreview();
        const entry=flashEditor.library.demos.find(d=>d.id===id);
        if(!entry)throw new Error('Choose a saved demo.');
        const dataset=await editorRequest('/api/datasets',{local_path:entry.root});
        const {document:doc}=await editorRequest(`/api/run/flash/editor/document?dataset_id=${encodeURIComponent(dataset.id)}&episode=${entry.episode}`);
        let status=await editorRequest('/api/run/flash/editor/track',{
            dataset_id:dataset.id,episode:entry.episode,revision:doc.revision,ensure_current:true});
        while(['loading','running'].includes(status.state)){
            flashEditor.rebuildPid=status.pid;editorElement('stop').disabled=false;
            editorElement('cancel-track').hidden=false;
            editorStatus(`Updating demo tracking: ${status.done||0}${status.total?' / '+status.total:''} frames…`);
            await new Promise(resolve=>setTimeout(resolve,1000));
            if(!flashEditor.visible||generation!==flashEditor.generation)throw new Error('Apply cancelled. Click Apply again when ready.');
            status=await editorRequest(`/api/run/flash/editor/tracking?dataset_id=${encodeURIComponent(dataset.id)}&episode=${entry.episode}`);
        }
        if(status.state!=='complete'||status.stale)throw new Error(status.stale_reason||status.reason||'Tracking must complete. Open the demo and Track again.');
        if(!flashEditor.visible||generation!==flashEditor.generation)throw new Error('Apply cancelled. Click Apply again when ready.');
        editorStatus('Starting robot…');
        await editorRequest('/api/run/flash/editor/run-demo',{id});
        flashEditor.player?.pause();flashEditor.mode='live';flashEditor.entry=null;renderFlashEditor();
        editorElement('apply-demo').value=id;
        mountFlashEditorPreview();
        editorStatus('Running robot. Rollout is being saved. Stop ends execution.');
    }catch(error){editorStatus(error.message);}
    finally{flashEditor.rebuildPid=null;flashEditor.busy=false;renderFlashEditor();void pollFlashDemo();}
}

async function openFlashRollout(entry){
    if(flashEditor.busy||flashEditor.recording)return;
    if(flashEditor.dirty&&!(await saveFlashDemo()))return;
    flashEditor.busy=true;
    try{
        if(!flashEditor.execution?.running)await stopFlashEditorPreview();
        flashEditor.player?.pause();
        flashEditor.mode='rollout';flashEditor.entry=entry;
        editorElement('terminal').textContent='Loading output…';
        mountFlashEditorPreview();renderFlashEditor();
        flashLive.streamMode='processed';setFlashLiveView('2d');syncFlashLiveControls();
        await selectFlashHistory(entry.id);
        editorStatus(`Recorded ${new Date(entry.created_at*1000).toLocaleString()}`);
        const results=await Promise.allSettled([
            editorRequest(`/api/run/flash/editor/rollouts/${entry.id}/trajectory`),
            editorRequest(`/api/run/flash/editor/rollouts/${entry.id}/log`)]);
        if(!flashEditor.visible||!flashEditor.previewHome||flashEditor.mode!=='rollout'||flashEditor.entry?.id!==entry.id)return;
        if(results[0].status==='fulfilled')setFlashLiveTrajectory(results[0].value);
        else {liveElement('trajectory-tools').hidden=false;liveElement('trajectory').disabled=true;
            liveElement('trajectory-tools').title=results[0].reason.message;}
        const line=document.createElement('div');line.className='terminal-line';
        line.textContent=results[1].status==='fulfilled'?results[1].value.text:results[1].reason.message;
        editorElement('terminal').replaceChildren(line);
    }catch(error){editorStatus(error.message);}
    finally{flashEditor.busy=false;renderFlashEditor();}
}

function initFlashOutputDivider(){
    const handle=editorElement('output-divider'),log=editorElement('run-log');
    const panel=log.querySelector('.run-terminal-panel'),main=editorElement('main');
    const storage='flash-editor-output-height';
    let preferred=Number(localStorage.getItem(storage))||170;
    panel.style.height=`${preferred}px`;
    const resize=(height,persist=true)=>{
        if(log.hidden||!log.open)return;
        const viewer=liveElement('image').parentElement;
        const maximum=Math.max(100,panel.clientHeight+viewer.clientHeight-120);
        const value=Math.max(100,Math.min(height,maximum));
        panel.style.height=`${value}px`;
        handle.setAttribute('aria-valuemin','100');
        handle.setAttribute('aria-valuemax',String(Math.round(maximum)));
        handle.setAttribute('aria-valuenow',String(Math.round(value)));
        if(persist){preferred=value;localStorage.setItem(storage,String(value));}
    };
    log.addEventListener('toggle',()=>{
        handle.hidden=log.hidden||!log.open;
        resize(preferred,false);
    });
    handle.addEventListener('pointerdown',event=>{
        event.preventDefault();handle.setPointerCapture(event.pointerId);handle.classList.add('dragging');
        const y=event.clientY,height=panel.clientHeight;
        const move=e=>resize(height+y-e.clientY);
        const finish=()=>{
            handle.classList.remove('dragging');
            handle.removeEventListener('pointermove',move);
            handle.removeEventListener('pointerup',finish);handle.removeEventListener('pointercancel',finish);
        };
        handle.addEventListener('pointermove',move);handle.addEventListener('pointerup',finish);handle.addEventListener('pointercancel',finish);
    });
    handle.addEventListener('keydown',event=>{
        if(['ArrowUp','ArrowDown','Home','End'].includes(event.key)){
            event.preventDefault();
            resize(event.key==='Home'?100:event.key==='End'?Infinity:panel.clientHeight+(event.key==='ArrowUp'?20:-20));
        }
    });
    new ResizeObserver(()=>resize(preferred,false)).observe(main);
}

document.addEventListener('DOMContentLoaded',()=>{
    initFlashOutputDivider();
    const handle=editorElement('library-divider'),section=editorElement('demo-section'),sidebar=editorElement('sidebar');
    const resize=height=>{
        const maximum=Math.max(100,sidebar.clientHeight-220),value=Math.max(100,Math.min(height,maximum));
        section.style.flex=`0 0 ${value}px`;
        handle.setAttribute('aria-valuemin','100');handle.setAttribute('aria-valuemax',String(Math.round(maximum)));
        handle.setAttribute('aria-valuenow',String(Math.round(value)));
        localStorage.setItem('flash-editor-demo-section-height',String(value));
    };
    const saved=Number(localStorage.getItem('flash-editor-demo-section-height'));
    if(saved>0){section.style.flex=`0 0 ${saved}px`;handle.setAttribute('aria-valuemin','100');handle.setAttribute('aria-valuemax',String(Math.max(saved,100)));handle.setAttribute('aria-valuenow',String(saved));}
    handle.addEventListener('pointerdown',event=>{
        event.preventDefault();handle.setPointerCapture(event.pointerId);
        const y=event.clientY,height=section.getBoundingClientRect().height;
        const move=e=>resize(height+e.clientY-y);
        const finish=()=>{handle.removeEventListener('pointermove',move);handle.removeEventListener('pointerup',finish);handle.removeEventListener('pointercancel',finish);};
        handle.addEventListener('pointermove',move);handle.addEventListener('pointerup',finish);handle.addEventListener('pointercancel',finish);
    });
    handle.addEventListener('keydown',event=>{
        if(['ArrowUp','ArrowDown'].includes(event.key)){
            event.preventDefault();resize(section.getBoundingClientRect().height+(event.key==='ArrowUp'?-20:20));
        }
    });
});

async function trackFlashDemo(){
    if(!flashEditor.document||flashEditor.trackingPid)return;
    if(flashEditor.dirty && !(await saveFlashDemo()))return;
    if(!flashEditor.document.revision && !(await saveFlashDemo()))return;
    try{
        const result=await editorRequest('/api/run/flash/editor/track',{dataset_id:flashEditor.dataset.id,episode:flashEditor.episode,revision:flashEditor.document.revision});
        flashEditor.trackingPid=result.pid;flashEditor.overlay=null;
        await pollFlashDemoTracking();
    }catch(error){editorElement('tracking-status').textContent=error.message;}
    renderFlashEditor();
}
async function pollFlashDemoTracking(ensureCurrent=false){
    clearTimeout(flashEditor.trackingTimer);
    if(!flashEditor.dataset)return;
    const generation=flashEditor.generation;
    try{
        let status=await editorRequest(`/api/run/flash/editor/tracking?dataset_id=${encodeURIComponent(flashEditor.dataset.id)}&episode=${flashEditor.episode}`);
        if(generation!==flashEditor.generation)return;
        if(ensureCurrent&&!flashEditor.dirty&&flashEditor.document?.revision&&flashEditor.document.objects.length&&
            (status.stale||status.state==='none')){
            editorElement('tracking-status').textContent=(status.stale_reason||'Tracking missing')+'. Updating…';
            flashEditor.overlay=null;flashEditor.displaySchema=false;flashEditor.displayResult=null;
            openFlashDemoPlayer();
            status=await editorRequest('/api/run/flash/editor/track',{
                dataset_id:flashEditor.dataset.id,episode:flashEditor.episode,revision:flashEditor.document.revision,ensure_current:true});
            if(generation!==flashEditor.generation)return;
        }
        const running=['loading','running'].includes(status.state);
        if((status.stale||status.state!=='complete')&&flashEditor.displayResult){
            flashEditor.overlay=null;flashEditor.displayResult=null;openFlashDemoPlayer();
        }
        flashEditor.trackingPid=running?status.pid:null;
        editorElement('cancel-track').hidden=!running;
        editorElement('tracking-status').textContent=status.stale?`${status.stale_reason}. Updating required.`:status.state==='complete'?`Tracking complete · pipeline v${status.pipeline_version}. Scrub to inspect masks and poses.`:status.state==='failed'?status.reason:running?`Tracking ${status.done||0}${status.total?' / '+status.total:''} frames…`:'';
        flashEditor.displaySchema=status.state==='complete'&&!status.stale&&!!status.display_schema;
        if(status.state==='complete'&&!status.stale&&flashEditor.displayResult!==status.result_id){
            flashEditor.displayResult=status.result_id;openFlashDemoPlayer();
        }
        renderFlashEditor();
        if(running && flashEditor.visible)flashEditor.trackingTimer=setTimeout(()=>pollFlashDemoTracking(),1000);
    }catch(error){editorElement('tracking-status').textContent=error.message;}
}
async function cancelFlashDemoTracking(){
    const pid=flashEditor.rebuildPid||flashEditor.trackingPid;
    if(!pid)return;
    if(flashEditor.rebuildPid)flashEditor.generation++;
    try{await editorRequest('/api/run/flash/editor/cancel-tracking',{pid});await pollFlashDemoTracking();}
    catch(error){editorElement('tracking-status').textContent=error.message;}
}
