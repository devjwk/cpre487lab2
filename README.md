<div align="center">

<img src="assets/banner.svg" alt="CNN INFERENCE IN C++ — Every layer written by hand and checked against TensorFlow, one layer at a time." width="100%">

![Language](https://img.shields.io/badge/Language-C%2B%2B-1D4ED8?style=flat-square&labelColor=172554)
![Build](https://img.shields.io/badge/Build-Make-1E3A8A?style=flat-square&labelColor=172554)
![Board](https://img.shields.io/badge/Board-ZedBoard-6366F1?style=flat-square&labelColor=172554)
![Stage](https://img.shields.io/badge/Stage-Starting-0891B2?style=flat-square&labelColor=172554)

Iowa State University · CprE 487/587 · Lab 2 · Team 06

[Why](#why) · [Where this lab fits](#where-this-lab-fits) · [Progress](#progress) · [Results](#results) · [Build and run](#build-and-run) · [Limitations](#limitations-and-next-steps)

</div>

---

> **Where it stands — Starting**  
> The repository holds the unmodified course framework. No layer has been implemented or verified in this re-run yet.

<!-- TEMPLATE: fill these four cells from measured output only. Leave "—" until a number exists. -->
| Layers matching TensorFlow | Max error vs. TensorFlow | Lab PC, per image | ZedBoard, per image |
| :---: | :---: | :---: | :---: |
| **0 / 12** | **—** | **—** | **—** |

| | |
|---|---|
| Period | Original team submission September 2026 · individual re-run started October 7, 2026 |
| Team | 2 — Zach Dixon, Jongwoo Kim |
| This repository | My re-run of Lab 2 from the course framework, checked step by step against the handout. The earlier team source and report are kept in `previous_lab_data/` |
| Stack | C++, Make, TensorFlow/Keras outputs as the reference, ZedBoard |
| Deliverables | <!-- TEMPLATE: link the report PDF and source archive once they exist --> not yet |
| Related labs | [Lab 1 — TensorFlow baseline](https://github.com/devjwk/cpre487lab1), [Lab 3 — MAC units](https://github.com/devjwk/487lab3), [Lab 4 — quantization](https://github.com/devjwk/cpre487lab4), [Lab 5 — hardware integration](https://github.com/devjwk/cpre487lab5) |

## Why

TensorFlow hides what inference costs. Lab 1 measured the model from the outside; this lab rewrites each layer as plain C++ loops so that every multiply-add is visible, countable, and ready to move onto hardware in Labs 3–5. Each layer is checked against the TensorFlow output of the same layer.

## Where this lab fits

<img src="assets/lab_flow.svg" alt="Lab 1 · Train in TensorFlow → Lab 2 · C++ framework → Lab 3 · MAC units → Lab 4 · Quantization → Lab 5 · Hardware integration" width="100%">

## Starting point

- **Framework:** the course template (`course` remote), unmodified. Its own documentation is in [`FRAMEWORK_README.md`](FRAMEWORK_README.md).
- **Weights:** `data/model/` ships with the template. The files checked so far (`conv1`, `conv2`, `conv6`, `dense1`, `dense2`) are byte-identical to the weights exported in Lab 1, under different names (`conv1_weights.bin` here, `conv2d_weights.bin` there).
- **Test images:** `data/image_{0,1,2}.bin` are float32 values in [0, 1] (49,152 bytes each). They are the course's images, not the three images used in the Lab 1 re-run.
- **Reference outputs:** `data/image_N_data/layer_{0..11}_output.bin`, one file per layer.

## Progress

<!-- TEMPLATE: replace this table with the handout's own checklist once the handout sections are read; one row per required item. -->
| Step | Status |
|---|---|
| Build the unmodified framework on the lab machine | not started |
| Implement and verify each layer against its reference output | not started |
| Verify whole-model inference | not started |
| Measure and profile on the lab PC | not started |
| Run on the ZedBoard | not started |
| Report | not started |

## Results

<!-- TEMPLATE: add tables here as measurements come in. Suggested: per-layer error vs. TensorFlow, per-layer time and MAC count, PC vs. ZedBoard. -->
No results yet.

## Build and run

```bash
make build      # 'make help' lists all targets; run 'make clean' first after changing a header
./build/ml      # runs the framework's checks
```

ZedBoard instructions are in [`FRAMEWORK_README.md`](FRAMEWORK_README.md).

## Repository layout

```
src/                    framework source; layers are in src/layers/
data/                   weights, test images and per-layer reference outputs (from the course template)
scripts/, zedboard/     ZedBoard build, flash and file-transfer tools
CprE487_587_Lab2.pdf    lab handout
FRAMEWORK_README.md     the framework's original README
previous_lab_data/      first team implementation (src/), its report, and the README of the old repository
assets/                 README banner and figure
```

## My role

<!-- TEMPLATE: write this yourself once the work is done. -->

## What I learned

<!-- TEMPLATE: write this yourself once the work is done. -->

## Limitations and next steps

<!-- TEMPLATE: list what was not done or not measured, with the reason. -->
- Nothing is implemented in this re-run yet.
