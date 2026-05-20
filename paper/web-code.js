/* paper/web-code.js — runtime extras for the HTML build of the paper.
 * Loaded by conf.typ via html.elem("script", read("web-code.js")) when
 * the html target is active.  PDF builds never see this file.
 *
 * Current contents:
 *   - Floating top-right toggle that shows/hides div.htmlonly blocks
 *     (the "Author's comments" yellow boxes).  Default ON; state
 *     persists in localStorage across page loads.
 */

(function () {
    'use strict';

    var STORAGE_KEY = 'phShowAuthorComments';

    function init() {
        var btn = document.createElement('button');
        btn.id = 'author-toggle';
        btn.type = 'button';
        document.body.appendChild(btn);

        /* Default ON.  Only flip OFF if user has explicitly toggled. */
        var on = localStorage.getItem(STORAGE_KEY) !== '0';

        function apply() {
            document.body.classList.toggle('hide-author', !on);
            btn.dataset.state = on ? 'on' : 'off';
            btn.setAttribute('aria-pressed', on ? 'true' : 'false');
            btn.textContent = (on ? "Hide" : "Show") + " author's comments";
        }
        apply();

        btn.addEventListener('click', function () {
            on = !on;
            localStorage.setItem(STORAGE_KEY, on ? '1' : '0');
            apply();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
