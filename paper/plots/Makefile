# Render paper plots to standalone SVGs (one per host) from ../data/fair.csv.
# Root is the paper dir (..) so "/data/fair.csv" resolves; run from here:
#   make            build all plot SVGs
#   make clean      remove generated SVGs
TYPST ?= typst
ROOT  := ..
HOSTS := m4 c8i
PLOTS := enc-bars dec-bw

SVGS := $(foreach p,$(PLOTS),$(foreach h,$(HOSTS),$(p)-$(h).svg))

.PHONY: all clean
all: $(SVGS)

# pattern: <plot>-<host>.svg from <plot>.typ
define RULE
$(1)-$(2).svg: $(1).typ common.typ $(ROOT)/data/fair.csv
	$(TYPST) compile --root $(ROOT) --input host=$(2) $(1).typ $$@
endef
$(foreach p,$(PLOTS),$(foreach h,$(HOSTS),$(eval $(call RULE,$(p),$(h)))))

clean:
	rm -f $(SVGS)
