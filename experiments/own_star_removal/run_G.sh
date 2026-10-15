#!/usr/bin/env bash
# Run G: the run E recipe with "agreed" bright stamps (own_star_removal.py stamps --stamp-variant agreed),
# for the projects given as "<name>" (inputs: D:/astro_gpu_diag/own_star_removal/<name>/snapshot, read-only copies of the
# project files; outputs: D:/astro_gpu_diag/own_star_removal/<name>/runG). Second GPU only.
#   bash experiments/own_star_removal/run_F.sh FlamingStar Horsehead
PY=<user folder>/sandbox/astro_GPU/.venv/Scripts/python.exe
HERE="$(cd "$(dirname "$0")" && pwd)"
S="$HERE/own_star_removal.py"
ALL=D:/astro_gpu_diag/own_star_removal/runG_driver.log
export CUDA_VISIBLE_DEVICES=1
for name in "$@"; do
  out=D:/astro_gpu_diag/own_star_removal/$name
  P=$out/snapshot
  lg=$out/runG_driver.log
  step() {
    local label=$1; shift
    echo "$(date +%T) $name: $label" | tee -a "$ALL" >> "$lg"
    if ! "$@" >> "$lg" 2>&1; then
      echo "$(date +%T) $name: $label FAILED" | tee -a "$ALL" >> "$lg"
      return 1
    fi
  }
  [ -f "$out/prep/library_nohalo_agreed.pt" ] || step stamps "$PY" "$S" stamps --stamp-variant agreed --project "$P" --out "$out" || continue
  step train "$PY" "$S" train --run runG --minus-halo --noisy-target --bright-aug --core-mask --stamps agreed \
    --minutes 32 --batch 6 --project "$P" --out "$out" || continue
  step infer "$PY" "$S" infer --run runG --project "$P" --out "$out" || continue
  step eval "$PY" "$S" eval --run runG --inject-stamps "" --project "$P" --out "$out" || continue
  step sheets "$PY" "$HERE/compare_runs.py" runE,runF,runG --project "$P" --out "$out"
  echo "$(date +%T) $name: done" | tee -a "$ALL" >> "$lg"
done
echo "$(date +%T) ALL DONE" >> "$ALL"
