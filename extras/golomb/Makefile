CFLAGS ?= -O3 -Wall -Wextra -std=c11
LDLIBS ?= -lm

# Auto-enable AVX-512 VBMI2 on a host that supports it; otherwise build scalar+NEON.
EXTRA_CFLAGS :=
ifeq ($(shell uname -m),x86_64)
  ifneq ($(shell grep -q -E 'vbmi2' /proc/cpuinfo 2>/dev/null && echo y),)
    EXTRA_CFLAGS += -march=x86-64-v4 -mavx512vbmi -mavx512vbmi2
  endif
endif

golomb: pivco-golomb.c
	$(CC) $(CFLAGS) $(EXTRA_CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f golomb

.PHONY: clean
