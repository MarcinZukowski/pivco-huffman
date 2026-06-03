# Build the PIVCO-Huffman paper as PDF and HTML.
#
#   make           builds both PDF and HTML
#   make pdf       PDF only
#   make html      HTML only
#   make watch     live-rebuild PDF on save (typst's built-in watcher)
#   make watchh    live-rebuild HTML on save
#   make clean     remove generated artifacts
#
# style.css is inlined into the HTML by ph.typ itself (via
# `html.elem("style", read("style.css"))` gated on `--input target=html`),
# so no post-processing is needed and the HTML is self-contained.
# The Typst HTML backend is experimental — expect quirks (e.g. <h2>
# emitted inside a <p>); style.css compensates with :has() selectors.

TYPST  ?= typst
SRC    := ph.typ
INCS   := conf.typ \
	ans.typ \
	appendix.typ \
	bottom-up.typ \
	enc.typ \
	intro.typ \
	related.typ \
	sideways.typ \

STYLE  := style.css
OUT    := out
PDF    := $(OUT)/ph.pdf
HTML   := $(OUT)/ph.html

.PHONY: all pdf html watch watchh clean

all: $(PDF) $(HTML)

$(OUT):
	@mkdir -p $(OUT)

pdf: $(PDF)
$(PDF): $(SRC) $(INCS) | $(OUT)
	$(TYPST) compile $(SRC) $(PDF)

html: $(HTML)
$(HTML): $(SRC) $(INCS) $(STYLE) web-code.js | $(OUT)
	$(TYPST) compile --creation-timestamp $$(date +%s) -f html --features html --input target=html $(SRC) $(HTML)

watch: | $(OUT)
	$(TYPST) watch --features html $(SRC) $(PDF)

watchh: | $(OUT)
	$(TYPST) watch -f html --features html --input target=html \
		$(SRC) $(HTML)

clean:
	rm -rf $(OUT)

SIMD_ANNOTATE=../../simd.dev/simd-annotate/simd-annotate

dist: $(PDF) $(HTML)
	mkdir -p out/figures
	cp ../figures/tree_viz.html ../figures/tree_viz_data.js out/figures/
	# Bake figures.json into fig-web.html (replaces `const CFG = null;`
	# with a literal const CFG = {...};) so the dist HTML is self-contained
	# and the figures.json fetch is skipped.  json.dumps does the JS-safe
	# encoding for us; doing this with sed mishandles "/" and "&".
	# Paths: fig-web.html + figures.json live in paper/figures/; tree_viz.*
	# files (copied above) live at project-root figures/.
	python3 -c "import json; \
	  cfg = json.load(open('figures/figures.json')); \
	  html = open('figures/fig-web.html').read(); \
	  assert html.count('const CFG = null;') == 1, \
	    'fig-web.html: expected exactly one CFG sentinel'; \
	  open('out/figures/fig-web.html','w').write( \
	    html.replace('const CFG = null;', \
	                 'const CFG = ' + json.dumps(cfg) + ';'))"
	python3 $(SIMD_ANNOTATE) -o /tmp/ph.html out/ph.html
	mv /tmp/ph.html out/ph.html

zip: out dist
	cd out && zip -r ../ph.zip .

WEB_PAPER=../web/paper-1.0
web: clean dist
	rm -rf $(WEB_PAPER)
	cp -R out $(WEB_PAPER)
