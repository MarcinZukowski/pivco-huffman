# Render paper plots to standalone SVGs (one per host) from ../data/fair.csv.
# Root is the paper dir (..) so "/data/fair.csv" resolves; run from here:
#   make            build all plot SVGs
#   make clean      remove generated SVGs
TYPST ?= typst
ROOT  := ..
HOSTS := m4 c8i
# tree_modes is named with _ to match a legacy file convention; the typst
# source is tree_modes.typ and outputs tree_modes_<host>.svg.
PLOTS_DASH  := enc-bars dec-bw
PLOTS_UNDER := tree_modes

SVGS := $(foreach p,$(PLOTS_DASH),$(foreach h,$(HOSTS),$(p)-$(h).svg)) \
        $(foreach p,$(PLOTS_UNDER),$(foreach h,$(HOSTS),$(p)_$(h).svg))

.PHONY: all clean
all: $(SVGS)

# dash naming: <plot>-<host>.svg from <plot>.typ
define RULE_DASH
$(1)-$(2).svg: $(1).typ common.typ $(ROOT)/data/fair.csv ../style.typ
	$(TYPST) compile --root $(ROOT) --input host=$(2) $(1).typ $$@
endef
$(foreach p,$(PLOTS_DASH),$(foreach h,$(HOSTS),$(eval $(call RULE_DASH,$(p),$(h)))))

# underscore naming: <plot>_<host>.svg from <plot>.typ
define RULE_UNDER
$(1)_$(2).svg: $(1).typ common.typ $(ROOT)/data/fair.csv ../style.typ
	$(TYPST) compile --root $(ROOT) --input host=$(2) $(1).typ $$@
endef
$(foreach p,$(PLOTS_UNDER),$(foreach h,$(HOSTS),$(eval $(call RULE_UNDER,$(p),$(h)))))

clean:
	rm -f $(SVGS)
