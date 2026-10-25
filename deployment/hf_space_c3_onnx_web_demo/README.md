---
title: CAS2023 C3 On-Device Segmentation Demo
emoji: 🧠
colorFrom: indigo
colorTo: pink
sdk: static
pinned: false
---

# CAS2023 C3 Knowledge-Distillation -- On-Device Demo

This page downloads the compressed C3 (combined-KD) student model as ONNX
and runs segmentation **entirely in your browser** using ONNX Runtime Web
(WASM) -- nothing is sent to a server. It measures and reports the actual
inference time on the device you're viewing it from.

Input patches are precomputed crops (64x192x192 voxels) from already
nnU-Net-preprocessed CAS2023 test scans, since full DICOM/NIfTI
preprocessing in the browser is out of scope for this demo (per the
project's own "phone demo" design note: full scans are hundreds of MB, so
the demo uses pre-cropped patches).

This is a research/course demonstration, not a clinical decision-support
tool.
