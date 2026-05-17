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
	intro.typ \
	sideways.typ \
	updown.typ \
	enc.typ \
	ans.typ \
	related.typ
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
$(HTML): $(SRC) $(INCS) $(STYLE) | $(OUT)
	$(TYPST) compile --creation-timestamp $$(date +%s) -f html --features html --input target=html $(SRC) $(HTML)

watch: | $(OUT)
	$(TYPST) watch $(SRC) $(PDF)

watchh: | $(OUT)
	$(TYPST) watch -f html --features html --input target=html \
		$(SRC) $(HTML)

clean:
	rm -rf $(OUT)
