/* paper/web-code.js -- runtime extras for the HTML build of the paper.
 * Loaded by conf.typ via html.elem("script", read("web-code.js")) when
 * the html target is active.  PDF builds never see this file.
 *
 * Current contents:
 *   - Floating top-right "tools" panel holding:
 *       * a "Show/Hide TOC" toggle and a hierarchical document
 *         outline.  Current section bold; under the current top-
 *         level section, subsections expand.  Click any entry to
 *         jump.  Position is tracked by scroll (rAF-throttled,
 *         no IntersectionObserver -- simpler and dependable).
 *       * an "Author's notes" toggle that shows/hides
 *         div.anote blocks.
 *     Both buttons default ON; state persists via localStorage.
 */

(function () {
    'use strict';

    var KEY_ANOTE = 'phShowAuthorNotes';
    var KEY_TOC    = 'phShowToc';

    /* ====== TOC build ====== */

    function buildHeadingTree() {
        /* h2 onward -- skip the document title (h1).  Adjust if the
         * paper ever uses h1 for sections. */
        var all = document.querySelectorAll('h2, h3, h4, h5, h6');
        var headings = [];
        for (var i = 0; i < all.length; i++) {
            var h = all[i];
            if (h.closest('#ph-tools')) continue;       /* skip our own UI */
            if (!h.id) {
                h.id = 'h-auto-' + i;                    /* fallback anchor */
            }
            headings.push(h);
        }
        /* Appendix folding: Typst emits every appendix entry (A..G) as h2,
         * same level as the "Appendices" heading itself, so by default they
         * sit flat in the TOC.  Bump every heading between "Appendices" and
         * the next top-level non-appendix heading by 1 level, so A..G fold
         * under "Appendices" and their h3 subsections fold under each A..G.
         * Sentinel: "Comments" (runtime-added end-of-doc widget heading)
         * ends the appendix range. */
        var inAppendix = false;
        var levels = headings.map(function (h) {
            var L = parseInt(h.tagName.substring(1), 10);
            var txt = h.textContent.trim();
            if (txt === 'Appendices') {
                inAppendix = true;
            } else if (inAppendix) {
                if (L === 2 && txt === 'Comments') inAppendix = false;
                else L = L + 1;
            }
            return L;
        });
        var root = { level: 1, children: [] };
        var stack = [root];
        headings.forEach(function (h, idx) {
            var level = levels[idx];
            while (stack[stack.length - 1].level >= level) stack.pop();
            var node = {
                level: level,
                text:  h.textContent.trim(),
                id:    h.id,
                el:    h,
                children: []
            };
            stack[stack.length - 1].children.push(node);
            stack.push(node);
        });
        return { root: root, flat: headings };
    }

    function renderTree(node) {
        if (!node.children.length) return null;
        var ul = document.createElement('ul');
        node.children.forEach(function (child) {
            var li = document.createElement('li');
            li.dataset.headingId = child.id;
            li.dataset.level     = String(child.level);
            var a = document.createElement('a');
            a.href = '#' + child.id;
            a.dataset.tocHref = child.id;
            a.textContent = child.text;
            li.appendChild(a);
            var sub = renderTree(child);
            if (sub) li.appendChild(sub);
            ul.appendChild(li);
        });
        return ul;
    }

    /* ====== current-section tracking ====== */

    function makeCurrentTracker(headings, panel) {
        var lookup = {};
        panel.querySelectorAll('li[data-heading-id]').forEach(function (li) {
            lookup[li.dataset.headingId] = li;
        });
        var lastId = null;

        function apply(currentId) {
            if (currentId === lastId) return;
            lastId = currentId;
            panel.querySelectorAll('li.toc-current, li.toc-expanded')
                  .forEach(function (li) {
                      li.classList.remove('toc-current', 'toc-expanded');
                  });
            if (!currentId) return;
            var li = lookup[currentId];
            if (!li) return;
            li.classList.add('toc-current', 'toc-expanded');
            var p = li.parentElement;
            while (p && p.id !== 'toc-panel') {
                if (p.tagName === 'LI') p.classList.add('toc-expanded');
                p = p.parentElement;
            }
        }

        var raf = 0;
        function update() {
            raf = 0;
            var offset = 120;     /* px below viewport top */
            var current = null;
            for (var i = 0; i < headings.length; i++) {
                if (headings[i].getBoundingClientRect().top <= offset) {
                    current = headings[i];
                } else {
                    break;
                }
            }
            apply(current ? current.id : (headings[0] && headings[0].id));
        }
        function schedule() {
            if (!raf) raf = requestAnimationFrame(update);
        }
        window.addEventListener('scroll', schedule, { passive: true });
        window.addEventListener('resize', schedule);
        update();
    }

    /* ====== panel construction ====== */

    function makeButton(id, initialOn, onLabel, offLabel, key, onToggle) {
        var btn = document.createElement('button');
        btn.id = id;
        btn.type = 'button';
        var on = localStorage.getItem(key) !== '0';
        if (initialOn !== undefined && localStorage.getItem(key) === null) {
            on = initialOn;
        }
        function apply() {
            btn.dataset.state = on ? 'on' : 'off';
            btn.setAttribute('aria-pressed', on ? 'true' : 'false');
            btn.textContent = on ? onLabel : offLabel;
            onToggle(on);
        }
        btn.addEventListener('click', function () {
            on = !on;
            localStorage.setItem(key, on ? '1' : '0');
            apply();
        });
        apply();
        return btn;
    }

    function init() {
        /* --- Comments heading + Hyvor widget relocation ---
         * Typst emits <section role="doc-endnotes"> as the very last body
         * child, so the Hyvor element from the Typst source lands BEFORE
         * the endnotes.  We move it (with a fresh "Comments" h2) past
         * the endnotes here, BEFORE building the TOC, so the heading is
         * picked up like any other section. */
        var hyvor    = document.querySelector('hyvor-talk-comments');
        var endnotes = document.querySelector('section[role="doc-endnotes"]');
        if (hyvor && endnotes) {
            var commentsHdr = document.createElement('h2');
            commentsHdr.id = 'comments';
            commentsHdr.textContent = 'Comments';
            endnotes.after(commentsHdr, hyvor);
        }

        /* --- floating container --- */
        var tools = document.createElement('div');
        tools.id = 'ph-tools';
        document.body.appendChild(tools);

        var buttons = document.createElement('div');
        buttons.id = 'ph-tools-buttons';
        tools.appendChild(buttons);

        /* --- TOC panel (built first; visibility toggled by button) --- */
        var tocPanel = document.createElement('div');
        tocPanel.id = 'toc-panel';
        tools.appendChild(tocPanel);

        var info = buildHeadingTree();
        var tree = renderTree(info.root);
        if (tree) tocPanel.appendChild(tree);

        /* --- click-to-jump (smooth, updates URL hash) --- */
        tocPanel.addEventListener('click', function (e) {
            var a = e.target.closest('a[data-toc-href]');
            if (!a) return;
            e.preventDefault();
            var id = a.dataset.tocHref;
            var target = document.getElementById(id);
            if (!target) return;
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            history.replaceState(null, '', '#' + id);
        });

        /* --- buttons --- */
        var tocBtn = makeButton(
            'toc-toggle', true, 'Hide TOC', 'Show TOC', KEY_TOC,
            function (on) { document.body.classList.toggle('hide-toc', !on); });
        buttons.appendChild(tocBtn);

        var anoteBtn = makeButton(
            'anote-toggle', true,
            "Hide author's notes", "Show author's notes",
            KEY_ANOTE,
            function (on) { document.body.classList.toggle('hide-anote', !on); });
        buttons.appendChild(anoteBtn);

        /* --- live current-section tracking --- */
        if (info.flat.length) {
            makeCurrentTracker(info.flat, tocPanel);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
