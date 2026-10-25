# CAS2023 teacher/student pipeline — disconnect-proof version

## What changed vs. your notebook
Your `nnUNet_raw` and `nnUNet_preprocessed` were pointed at `/content/...` (local
Colab disk), which gets wiped on every disconnect — that's why you kept losing
preprocessing and had to redo it. These scripts point **everything**
(`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) at your Drive
(`WORK_DIR`) instead. Slightly slower I/O, but a disconnect only ever costs you
unsaved training progress since the last epoch checkpoint — never the
conversion or preprocessing again.

Every training call always passes `--c` (continue), which is a no-op if a run
already finished and otherwise resumes from the last checkpoint — so it is
always safe to just re-run a stage after a disconnect.

## One-time setup
1. Upload this whole `nnunet_pipeline/` folder into your Drive at
   `/content/drive/MyDrive/cas2023/pipeline/` (any path is fine, just update
   the `!python ...` path below to match).
2. Open `00_settings.py` and check `CAS_DIR`, `WORK_DIR`, `DATASET_ID`,
   `TRAINER` etc. match your original settings cell. `SEED`/`TEST_FRACTION`
   only matter if `WORK_DIR/data_split.json` doesn't already exist — if it
   does (from your teacher run), it's reused automatically so the student
   gets the identical test split.

## Running it
In a fresh Colab notebook, just two cells:

```python
!pip install -q nnunetv2 SimpleITK tqdm
```

```python
!python /content/drive/MyDrive/cas2023/pipeline/run_all.py
```

That runs every stage in order: convert → preprocess teacher → train teacher →
predict+evaluate teacher → teacher stats → build student plan → preprocess
student → train student → predict+evaluate student → student stats.

**If Colab disconnects partway through:** just re-run the exact same cell
again. Finished stages are skipped instantly; an interrupted training stage
resumes from its last checkpoint instead of restarting.

### Running one stage at a time (optional)
```python
!python /content/drive/MyDrive/cas2023/pipeline/run_all.py --stage train_student
```
Stage names: `convert`, `preprocess_teacher`, `train_teacher`,
`predict_eval_teacher`, `stats_teacher`, `make_student_plan`,
`preprocess_student`, `train_student`, `predict_eval_student`, `stats_student`.

### Resuming from a specific stage onward
```python
!python /content/drive/MyDrive/cas2023/pipeline/run_all.py --from train_student
```

## Outputs
- `WORK_DIR/data_split.json` — the train/test case IDs (shared by teacher & student)
- `WORK_DIR/report_assets/teacher_stats.json`, `student_stats.json` — params/checkpoint size
- `WORK_DIR/report_assets/teacher_test_summary.json`, `student_test_summary.json` — Dice/IoU per case
- `WORK_DIR/teacher_predictions/`, `student_predictions/` — predicted masks

## Student architecture
The student uses the **same architecture/stages** as the teacher, just with
`features_per_stage` scaled by `STUDENT_WIDTH_SCALE` (default `0.5`, i.e. half
the channel width at every stage, ~4x fewer conv parameters).

## The 5-model comparison (A/B/C1/C2/C3)
This matches "How will we know whether distillation helps?" in the prep deck:

| Tag | Model | Trained with |
|-----|-------|--------------|
| `teacher` | A — Teacher | expert masks (full-width network) |
| `student` | B — Scratch student | expert masks only, no teacher (baseline) |
| `c1` | C1 — Output-level KD | KL loss vs. teacher's soft output only (alpha=0) |
| `c2` | C2 — Feature-level KD | L2 loss vs. teacher's bottleneck features + GT loss |
| `c3` | C3 — Combined KD | GT loss + output-KD loss + optional clDice (best-effort) |

B is already covered by the original `train_student`/`preprocess_student`
stages. C1/C2/C3 reuse the exact same preprocessed data and plan as B (only
the loss function differs), so there's no extra preprocessing step for them.

**One-time per fresh runtime**, before training C1/C2/C3: the custom trainer
classes must be installed into the nnunetv2 package so `-tr` can find them
(pip installs don't persist across disconnects):
```python
!python /content/drive/MyDrive/cas2023/pipeline/run_all.py --stage install_trainers
```
(`run_all.py` with no `--stage` runs this automatically as part of the full
sequence, in the right order, after model B.)

Then C1/C2/C3 each train against the **frozen teacher checkpoint**
(`TEACHER_CHECKPOINT` in `00_settings.py` — set automatically once the teacher
stage has produced `checkpoint_final.pth`). Run them with:
```python
!python /content/drive/MyDrive/cas2023/pipeline/run_all.py --from train_c1
```
or one at a time with `--stage train_c1` / `train_c2` / `train_c3`.

### Trainer source files
`trainers/nnUNetTrainerKDStudent.py` is the shared base (output-level KD + GT
+ optional clDice). `trainers/nnUNetTrainerKD_C1_OutputOnly.py`,
`_C2_Feature.py`, `_C3_Combined.py` are thin presets selected via `-tr`.
`trainers/soft_cldice.py` is the differentiable clDice loss used by C3.

If a KD run's loss looks off (e.g. one term dominating), the relevant knobs
are env vars set in `09_train_kd_variant.py`: `KD_ALPHA`, `KD_TEMPERATURE`,
`KD_FEATURE_WEIGHT` (C2 only), `KD_CLDICE_WEIGHT` (C3 only).

### Final comparison
Once all 5 are done, `WORK_DIR/report_assets/` holds `*_stats.json` (params,
checkpoint size) and `*_test_summary.json` (Dice/IoU) for every tag —
everything needed for the final metrics table in the deck's roadmap (phase 5).
