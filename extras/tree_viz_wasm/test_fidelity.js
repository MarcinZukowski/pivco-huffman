/* Fidelity A/B driver: runs the wasm module over every distribution in
 * figures/tree_viz_data.js, emitting the same report format as
 * native_dump.c.  Usage: node test_fidelity.js <tree_viz_wasm.js> \
 *   <tree_viz_data.js> [--dump-freqs]  */
const fs = require("fs");

const wasmJs = fs.readFileSync(process.argv[2], "utf8");
const dataJs = fs.readFileSync(process.argv[3], "utf8");
const dumpFreqs = process.argv.includes("--dump-freqs");

const window = {};
global.window = window;
global.atob = (b) => Buffer.from(b, "base64").toString("binary");
eval(dataJs);
eval(wasmJs);

const hex = (l) => Array.from(l, (x) => x.toString(16)).join("");

window.PIVCO_WASM.then((api) => {
  for (const d of window.PIVCO_DISTRIBUTIONS) {
    if (dumpFreqs) {
      console.log(d.name.replace(/\s/g, "_") + " " + d.freq.join(" "));
      continue;
    }
    console.log("== " + d.name.replace(/\s/g, "_"));
    console.log("plain " + hex(api.plainLengths(d.freq)));
    for (let e = 1; e <= 3; e++)
      for (let f = 0; f <= 1; f++) {
        const r = api.jointLengths(d.freq, e, f);
        console.log(
          `joint_e${e}_f${f}_a${r.adopted ? 1 : 0} ` + hex(r.lengths)
        );
      }
  }
});
