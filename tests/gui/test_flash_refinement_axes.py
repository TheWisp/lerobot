"""Run the review's rendering code with both surface layers hidden."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).parents[2] / "src/lerobot/gui/static/flash_refinement_review.html"


def test_refinement_axes_do_not_depend_on_surface_visibility():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to exercise the review renderer")
    script = r"""
const fs=require('fs'),vm=require('vm');
let code=fs.readFileSync(process.argv[1],'utf8').split('<script type="module">')[1].split('</script>')[0];
code=code.replace(/^import .*;$/mg,'');
// Skip network bootstrap; supply a packet through the same rendering path.
code=code.slice(0,code.lastIndexOf("try{const r=await fetch"));
const lines=[],labels=[],nodes={};let layers=[],options={};
const g={drawImage(){},beginPath(){},moveTo(...p){this.start=p},lineTo(...p){lines.push([this.start,p])},stroke(){},fillText(t){labels.push(t)}};
const document={getElementById(id){return nodes[id]??={checked:id==='axes',getContext(){return g},setAttribute(){}}}};
const ctx=vm.createContext({document,URLSearchParams,location:{search:''},console});
vm.runInContext(code,ctx);
ctx.viewer={update(d,b,k,l){layers=l},setOptions(o){options=o}};
vm.runInContext(`state.bitmap={width:100,height:100};state.packet={camera:{intrinsics:[[100,0,50],[0,100,50],[0,0,1]]},display_pose:[[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]],axes:[[0,0,0],[0.1,0,0],[0,0.1,0],[0,0,0.1]],before:[[0,0,0]],after:[[0,0,0]]};render2d();state.mode='3d';state.viewer=viewer;render3d()`,ctx);
const shown=layers.filter(l=>options[l.layer]!==false&&options.axes&&l.axes!==false&&l.axisPoints);
const on={lines:lines.length,labels:[...labels],axes:shown.map(l=>({points:l.axisPoints,pose:l.pose}))};
nodes.axes.checked=false;lines.length=0;labels.length=0;
vm.runInContext("state.mode='2d';render2d();state.mode='3d';render3d()",ctx);
const off={lines:lines.length,axes:layers.filter(l=>options[l.layer]!==false&&options.axes&&l.axes!==false&&l.axisPoints).length};
console.log(JSON.stringify({on,off}));
"""
    result = subprocess.run([node, "-e", script, str(PAGE)], text=True, capture_output=True, check=True)
    data = json.loads(result.stdout)
    assert data["on"]["lines"] == 3
    assert data["on"]["labels"] == ["X", "Y", "Z"]
    assert len(data["on"]["axes"]) == 1
    assert data["on"]["axes"][0]["pose"][2][3] == 1
    assert data["off"] == {"lines": 0, "axes": 0}
