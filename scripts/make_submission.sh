#!/bin/bash
# Assemble submission/lab2_src_06/ from the current sources and check that it builds and passes on its own.
# Usage: ./scripts/make_submission.sh   (run from the repository root)
set -e

OUT=submission/lab2_src_06
rm -rf "$OUT"
mkdir -p "$OUT/scripts"
cp -R src Makefile "$OUT/"
cp scripts/import_lab1_data.py scripts/plot_layer_times.py scripts/tf_inference_time.py "$OUT/scripts/"

# Build a throw-away copy from scratch, exactly as a grader would, and run it against data/
CHECK=$(mktemp -d)
cp -R "$OUT/." "$CHECK/"
ln -s "$PWD/data" "$CHECK/data"
(cd "$CHECK" && make build > build.log 2>&1) || { cat "$CHECK/build.log"; echo "BUILD FAILED"; exit 1; }
if grep -q -E "warning:| error" "$CHECK/build.log"; then grep -E "warning:| error" "$CHECK/build.log"; echo "BUILD HAS WARNINGS"; exit 1; fi
(cd "$CHECK" && ./build/ml > run.log 2>&1) || { tail -5 "$CHECK/run.log"; echo "RUN FAILED"; exit 1; }
echo "compiler: $(${CXX:-g++} --version | head -1)"
echo "build: OK, no warnings"
echo "tests passed: $(grep -c 'True' "$CHECK/run.log") (expected 41), failed: $(grep -c 'False' "$CHECK/run.log") (expected 0)"
echo "files in $OUT: $(find "$OUT" -type f | wc -l | tr -d ' ')"
echo "To make the Canvas archive: (cd submission && zip -qr lab2_src_06.zip lab2_src_06)"
rm -rf "$CHECK"
