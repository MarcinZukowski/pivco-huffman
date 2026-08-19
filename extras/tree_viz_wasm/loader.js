/* Loader half of tree_viz_wasm.js (appended after the base64 blob by
 * build.sh).  Exposes window.PIVCO_WASM: a promise resolving to
 *   { plainLengths(freq256) -> Uint8Array(256) | throws,
 *     jointLengths(freq256, effort, fseEnabled) ->
 *         { lengths: Uint8Array(256), adopted: bool } }
 * freq256: any array-like of 256 non-negative numbers. */
window.PIVCO_WASM = (function () {
  const b64 = window.PIVCO_WASM_B64;
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  return WebAssembly.instantiate(bytes, {
    env: { log2: Math.log2, ceil: Math.ceil },
  })
    .then(({ instance }) => {
      const e = instance.exports;
      const mem = () => e.memory.buffer;
      function setFreq(freq) {
        const buf = new Float64Array(mem(), e.viz_freq_buf(), 256);
        for (let i = 0; i < 256; i++) buf[i] = freq[i] || 0;
      }
      const lengths = () =>
        new Uint8Array(mem(), e.viz_len_buf(), 256).slice();
      return {
        plainLengths(freq) {
          setFreq(freq);
          const rc = e.viz_plain_lengths();
          if (rc !== 0) throw new Error("pivco build_table rc=" + rc);
          return lengths();
        },
        jointLengths(freq, effort, fseEnabled) {
          setFreq(freq);
          const rc = e.viz_joint_lengths(effort | 0, fseEnabled ? 1 : 0);
          if (rc < 0) throw new Error("pivco build_table rc=" + rc);
          return { lengths: lengths(), adopted: rc === 1 };
        },
      };
    });
})();
