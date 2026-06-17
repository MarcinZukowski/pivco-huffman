# `sweep_uarch` — regenerating the "PH over Huff0 across CPU generations" figure

This directory produces **Figure `fig-trends`** in the paper
(`paper/cao.typ`, caption *"PH performance over Huff0 on 3 CPU families on AWS
(across datasets)"*), rendered from `paper/plots/sweep_uarch_dec_op.svg`.

The figure plots ph's **decode `dec_op`** speedup over stock Huff0
(`HUF_decompress`) as a function of CPU release year, one panel per family
(Intel / AMD / Graviton), with a min/mean/max band across datasets. `dec_op`
is used because stock Huff0 has no prebuilt-table API, so it is the only
apples-to-apples decode column.

## Pieces

| file | role |
|------|------|
| `hosts.tsv` | host inventory: `alias  type  arch  family  year  uarch  region  existing`. The `family`/`year` columns drive the plot axes; `existing=no` rows are what `provision.sh` launches. **Keep `existing` in sync with `myaws` before provisioning.** |
| `provision.sh` | launches every `existing=no` host (us-west-2, build-tools user-data). Idempotent on the `existing` flag, *not* on duplicate Name tags — fix `existing` first or you get duplicates. |
| `sweep.sh` | rsync → clean build → `pivco_fair_bench` (no args: default dist set, all engines, default block size) on each host via `taskset -c 0`; per-host logs to `results/sweep_uarch/<date>/<alias>.txt`. |
| `plot.py` | parses the per-host logs + `hosts.tsv`, computes ph/huf0 `dec_op` ratios, writes `results/sweep_uarch/<date>/ph_vs_huf0_by_year_dec_op.svg` and mirrors it to `paper/plots/sweep_uarch_dec_op.svg`. |

## Prerequisites

- `aws` CLI with working credentials (`aws sts get-caller-identity`).
- `myaws` (`~/bin/myaws`) for status/ssh-config sync.
- SSH keys referenced by `~/.ssh/config` (us-west-1 → N-California, us-west-2 → Oregon).
- `python3` with `matplotlib` for `plot.py`.

## Regenerate (full procedure)

```sh
cd extras/sweep_uarch

# 1. Sync the inventory with reality.  List running instances:
myaws
#    Edit hosts.tsv: set existing=yes for every running alias, existing=no for
#    the generations you still need.  Add any new alias (e.g. m9g) with its
#    family/year/uarch.

# 2. Launch the missing hosts (only existing=no rows).
./provision.sh
#    Gotchas seen in practice (see "AMI / compiler matrix" below):
#    - a1.large (Graviton1) is NOT offered in the hardcoded us-west-2b subnet;
#      relaunch it in us-west-2a/c (subnet-20bc9357).
#    - a1 KERNEL-PANICS on the AL2023 ARM AMI (AL2023 dropped Graviton1).
#      Launch a1 on an Amazon Linux 2 arm64 AMI instead.
#    provision.sh aborts (set -e) if any launch errors — fix and re-run, or
#    launch the stragglers by hand with aws ec2 run-instances.

# 3. Register the new instances in ~/.ssh/config + known_hosts:
myaws ssh_update
#    If a reused DNS has a CONFLICTING old host key, ssh still fails after
#    ssh_update (it renames, doesn't clear mismatches).  Fix that one host:
#      ssh-keygen -R <dns>; ssh-keyscan -t ed25519,ecdsa,rsa <dns> >> ~/.ssh/known_hosts

# 4. Put a RECENT compiler on every host (the figure is compiler-sensitive;
#    use the same one everywhere it's available).
#    AL2023 (all x86 + Graviton2+): sudo dnf install -y clang19   -> /usr/bin/clang-19
#    AL2 (a1/Graviton1 only):       sudo yum install -y clang cmake3   (clang 11, the newest AL2 offers)

# 5. Run the sweep.  Build each host with its recent compiler, then bench.
#    sweep.sh does a clean default-compiler build; for clang either edit it to
#    pass -DCMAKE_C_COMPILER=clang-19 / -DCMAKE_CXX_COMPILER=clang++-19, or run
#    per-host (this is what the 2026-06-16 regen did):
#      cmake -B build-clang19 -DCMAKE_BUILD_TYPE=Release \
#        -DCMAKE_C_COMPILER=/usr/bin/clang-19 -DCMAKE_CXX_COMPILER=/usr/bin/clang++-19
#      cmake --build build-clang19 --target pivco_fair_bench -jN
#      taskset -c 0 ./build-clang19/pivco_fair_bench   > results/.../test-<alias>.txt
#    a1 (AL2, clang-11, cmake3) additionally needs -pthread (older glibc does
#    not auto-link it) and only ph+huf0 are needed for the plot:
#      cmake3 -B build-clang11 -DCMAKE_BUILD_TYPE=Release \
#        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
#        -DCMAKE_C_FLAGS=-pthread -DCMAKE_CXX_FLAGS=-pthread -DCMAKE_EXE_LINKER_FLAGS=-pthread
#      taskset -c 0 ./build-clang11/pivco_fair_bench --engines=ph,pha,huf0,huf0_4x2

# 6. Render the figure (also mirrors into paper/plots/).
./plot.py results/sweep_uarch/<date>
#    -> paper/plots/sweep_uarch_dec_op.svg   (committed; the paper includes it)

# 7. Stop/terminate the spun-up instances to save cost:
#    myaws ec2_stop <alias>     (or ec2_term to delete)
```

## AMI / compiler matrix

| host class | AMI | compiler used |
|------------|-----|---------------|
| x86 + Graviton2+ (everything but a1) | Amazon Linux 2023 arm64/x86 | **clang-19** (`dnf install clang19`) |
| a1 / Graviton1 | Amazon Linux **2** arm64 (AL2023 panics) | **clang-11** + cmake3 (AL2 max), `-pthread` |

a1 is the one host that can't match the fleet compiler (Graviton1 is too old for
AL2023 / a recent toolchain). It is built with clang-11 and flagged as such; its
ph/huf0 ratio is still same-compiler-both-engines, so internally consistent.

## Notes / gotchas

- **No positional repeats arg.** Current `pivco_fair_bench` self-paces and
  rejects unknown args (since `f3e3ef3`); `sweep.sh` calls it with no args.
- **Block size.** The sweep runs at each build's default `PIVCO_BLOCK_SIZE`
  (32K x86 / 16K Apple-arm; see `docs/BLOCK_SIZE.md`), so the figure reflects ph
  *as shipped today*, not the paper-era 8K.
- **Metric.** `plot.py` defaults to `dec_op`; pass a 2nd arg `dec_pb` only when
  comparing against `huf0_4x2` (which has a prebuilt column). The paper figure
  is `dec_op`.
- **Huff0 refuses `image_jpeg`**, so that dataset drops out of the ratio (8
  datasets/host, not 9) — expected.
- **Cost.** These are `.large` (2 vCPU) on-demand instances across two regions;
  stop/terminate them after the sweep.
