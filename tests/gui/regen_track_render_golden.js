// Regenerate the timeline row golden. Run deliberately, and READ THE DIFF:
//   node tests/gui/regen_track_render_golden.js
// Every changed entry is a row type whose pixels moved. If that was not the
// intent, the change is a regression and the diff is the report.
const fs = require("fs");
const { render, GOLDEN } = require("./track_render_golden.js");
fs.writeFileSync(GOLDEN, JSON.stringify(render(), null, 2) + "\n");
console.log("wrote", GOLDEN);
