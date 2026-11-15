/* A screen-space overlay shared by calibrated 2D and orbitable 3D views.
 * The curve and waypoints show dispatched targets or demonstrated tool poses.
 */
class FlashTrajectory {
    constructor(host, data, seek) {
        this.host=host;this.data=data;this.seek=seek;
        this.svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
        this.svg.classList.add('flash-trajectory');
        this.svg.setAttribute('aria-label','End-effector trajectory');
        this.tip=document.createElement('div');this.tip.className='flash-trajectory-tooltip';
        this.tip.setAttribute('role','tooltip');this.tip.hidden=true;
        host.append(this.svg,this.tip);
        const element=(name,cls)=>{
            const node=document.createElementNS(this.svg.namespaceURI,name);node.setAttribute('class',cls);
            this.svg.append(node);return node;
        };
        this.curves=data.spans.map(()=>['halo','future','past'].map(kind=>element('path','trajectory-'+kind)));
        this.dots=data.points.map((point,i)=>{
            const dot=element('circle','trajectory-waypoint');dot.setAttribute('r','2.2');
            dot.dataset.waypoint=String(i);dot.setAttribute('tabindex',i===0?'0':'-1');
            dot.setAttribute('role','button');
            dot.setAttribute('aria-label',`Waypoint ${i+1}`);
            dot.addEventListener('pointerenter',()=>this.inspect(i));
            dot.addEventListener('pointerleave',()=>{this.hover=null;this.tip.hidden=true;});
            dot.addEventListener('focus',()=>this.inspect(i));
            dot.addEventListener('blur',()=>{this.hover=null;this.tip.hidden=true;});
            dot.addEventListener('click',()=>seek(point.time_s));
            dot.addEventListener('keydown',event=>{
                if(event.key==='Enter'||event.key===' '){event.preventDefault();seek(point.time_s);}
                if(['ArrowLeft','ArrowRight'].includes(event.key)){
                    event.preventDefault();const step=event.key==='ArrowRight'?1:-1;
                    let next=i+step;
                    while(next>=0&&next<data.points.length&&!this.pixels?.[next])next+=step;
                    if(next<0||next>=data.points.length)return;
                    dot.setAttribute('tabindex','-1');this.dots[next].setAttribute('tabindex','0');this.dots[next].focus();
                }
            });return dot;
        });
        this.current=element('circle','trajectory-current');this.current.setAttribute('r','4.5');
        this.resize=new ResizeObserver(()=>this.render());this.resize.observe(host);
    }
    relativeTime(index){
        const delta=Math.round((this.data.points[index].time_s-this.options.time)*100)/100;
        return `${delta>0?'+':''}${delta.toFixed(2)}s`;
    }
    updateTime(time){
        if(this.options)this.options.time=time;
        if(this.hover!=null&&this.options?.enabled&&this.options.view!=='depth')this.inspect(this.hover);
    }
    inspect(index){
        this.hover=index;const position=this.pixels?.[index];if(!position){this.tip.hidden=true;return;}
        const offset=this.relativeTime(index);
        this.tip.textContent=`${offset} · Waypoint ${index+1}`;
        this.dots[index].setAttribute('aria-label',`Waypoint ${index+1}, ${offset}`);
        this.tip.hidden=false;
        this.tip.style.left=`${Math.max(6,Math.min(this.host.clientWidth-this.tip.offsetWidth-6,position[0]+12))}px`;
        this.tip.style.top=`${Math.max(6,Math.min(this.host.clientHeight-this.tip.offsetHeight-6,position[1]-34))}px`;
    }
    render(options){
        if(options)this.options=options;
        if(!this.options)return;
        const {enabled,view,camera,time,viewer,timeRange}=this.options;
        const inRange=at=>!timeRange||(at>=timeRange[0]&&at<=timeRange[1]);
        this.svg.style.display=enabled&&view!=='depth'?'':'none';
        if(!enabled||view==='depth'){this.tip.hidden=true;return;}
        const w=this.host.clientWidth,h=this.host.clientHeight;
        this.svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
        if(!camera||!w||!h)return;
        const scale=Math.min(w/camera.width,h/camera.height),k=camera.intrinsics;
        const project=points=>view==='3d'?viewer?.project(points)||points.map(()=>null):points.map(([x,y,z])=>
            z<=0?null:[(w-camera.width*scale)/2+(k[0][0]*x/z+k[0][2])*scale,
                (h-camera.height*scale)/2+(k[1][1]*y/z+k[1][2])*scale]);
        const path=(positions,times,pastOnly=false)=>{
            let value='',connected=false;
            positions.forEach((p,i)=>{
                if(!p||!inRange(times[i])||(pastOnly&&times[i]>time)){connected=false;return;}
                value+=`${connected?'L':'M'}${p[0].toFixed(2)},${p[1].toFixed(2)} `;connected=true;
            });return value;
        };
        this.data.spans.forEach((span,i)=>{
            const pixels=project(span.positions),all=path(pixels,span.times);
            this.curves[i][0].setAttribute('d',all);this.curves[i][1].setAttribute('d',all);
            this.curves[i][2].setAttribute('d',path(pixels,span.times,true));
        });
        this.pixels=project(this.data.points.map(point=>point.position)).map((p,i)=>inRange(this.data.points[i].time_s)?p:null);
        let current=-1;
        this.dots.forEach((dot,i)=>{
            const p=this.pixels[i],past=this.data.points[i].time_s<=time;
            dot.style.display=p?'':'none';dot.classList.toggle('future',!past);
            dot.setAttribute('aria-label',`Waypoint ${i+1}, ${this.relativeTime(i)}`);
            if(p){dot.setAttribute('cx',p[0]);dot.setAttribute('cy',p[1]);}
            if(p&&past)current=i;
        });
        if(!this.dots.some((dot,i)=>this.pixels[i]&&dot.getAttribute('tabindex')==='0')){
            const first=this.pixels.findIndex(Boolean);
            this.dots.forEach((dot,i)=>dot.setAttribute('tabindex',i===first?'0':'-1'));
        }
        const p=this.pixels[current];this.current.style.display=p?'':'none';
        if(p){this.current.setAttribute('cx',p[0]);this.current.setAttribute('cy',p[1]);}
        if(this.hover!=null)this.inspect(this.hover);
    }
    dispose(){this.resize.disconnect();this.svg.remove();this.tip.remove();}
}
