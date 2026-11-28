/* Same labeled numeric/switch controls as the training form; used by both FLASH sources. */
function createFlashPipelineForm(form, fields, values, context) {
    form.replaceChildren(); form.classList.add('flash-pipeline-form');
    for(const section of [...new Set(fields.map(f=>f.section))]) {
        const group=document.createElement('section'), title=document.createElement('h3');
        title.textContent=section; group.append(title);
        const grid=document.createElement('div');grid.className='training-field-row';group.append(grid);
        for(const field of fields.filter(f=>f.section===section)) {
            const label=document.createElement('label');label.className='training-field';
            const text=document.createElement('span');text.className='training-field-label';text.textContent=field.label;
            const input=document.createElement('input');input.name=field.key;input.setAttribute('aria-label',field.label);
            input.type=field.type==='bool'?'checkbox':'number';
            const stored=field.key.split('.').reduce((v,key)=>v?.[key],values);
            const value=stored===undefined?field.default:stored;
            if(field.type==='bool') {input.checked=value;label.classList.add('training-field-bool');}
            else {
                input.value=field.nullable&&value==null?'':Number((value*(field.scale||1)).toPrecision(10));
                input.required=!field.nullable;
                for(const key of ['min','max','step'])if(field[key]!==undefined)input[key]=field[key];
                if(field.type==='float')input.step='any';
            }
            const hint=document.createElement('span');hint.className='training-field-hint';hint.textContent=field.description;
            label.append(text,input,hint);grid.append(label);
        }
        form.append(group);
    }
    const update=()=>{
        const current=context();
        for(const field of fields) {
            const input=form.elements.namedItem(field.key);
            const applicable=(!field.live_only || current.live) && field.exclude_matcher!==current.matcher &&
                (!field.image_backend || field.image_backend===current.matching);
            input.disabled=!!current.running || !applicable;
            input.closest('label').hidden=!applicable;
        }
    };
    update();
    return {update, read:()=>{
        const result=structuredClone(values);
        for(const field of fields) {
            const input=form.elements.namedItem(field.key);
            if(input.disabled)continue;
            if(!input.checkValidity()) {
                const dialog=form.closest('dialog');if(!dialog.open)dialog.showModal();
                input.reportValidity();throw new Error(`Check ${field.label}`);
            }
            const keys=field.key.split('.'), key=keys.pop();
            const parent=keys.reduce((v,k)=>v[k]??=( {} ),result);
            parent[key]=field.type==='bool'?input.checked:
                field.nullable&&input.value===''?null:Number(input.value)/(field.scale||1);
        }
        const current=context();
        result.config.shape.registration.backend=current.matcher;
        if(current.matching)result.shared_matching={...result.shared_matching,backend:current.matching};
        return result;
    }};
}
function renderFlashPipelineSettings(fields, values) {
    flashLive.fields=fields;
    flashLive.settingsForm=createFlashPipelineForm(liveElement('settings-form'),fields,values,()=>({
        matcher:liveElement('matcher').value,matching:liveElement('matching').value,live:true,running:flashLive.running
    }));
    updateFlashPipelineControls();
}
function updateFlashPipelineControls(running=flashLive.running) {
    flashLive.running=running;
    flashLive.settingsForm?.update();
    liveElement('settings-reset').disabled=!!running;
}
function readFlashPipelineSettings() {return flashLive.settingsForm.read();}
function resetFlashPipelineSettings() {
    liveElement('matcher').value=flashLive.defaults.config.shape.registration.backend;
    liveElement('matching').value=flashLive.defaults.shared_matching.backend;
    renderFlashPipelineSettings(flashLive.fields,flashLive.defaults);
}
