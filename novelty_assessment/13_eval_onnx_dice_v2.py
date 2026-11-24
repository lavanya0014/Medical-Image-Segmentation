"""
Full-volume Dice of an ONNX model (fp32 or int8) on the same 15 test cases, using nnU-Net's own
preprocessing + sliding window, with ONNX Runtime (CPU) as the network. Use it to get the Dice
of the int8 model. Mirroring is OFF so it is quick; to compare fairly, run it for BOTH fp32 and int8.

    !python .../13_eval_onnx_dice_v2.py c1 fp32 --limit 2     # smoke test on 2 cases first
    !python .../13_eval_onnx_dice_v2.py c1 int8
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import novelty_env_v2  # noqa: F401
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_settings.py")).read())

import numpy as np, torch, onnxruntime as ort
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

ap = argparse.ArgumentParser()
ap.add_argument("which"); ap.add_argument("variant", choices=["fp32", "int8"])
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

onnx_path = Path(WORK_DIR) / "deployment" / f"{a.which}{RUN_TAG}_3d_fullres{'_int8' if a.variant == 'int8' else ''}.onnx"
model_dir_ = get_model_dir(a.which)

class OrtNet(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.keep = torch.nn.Parameter(torch.zeros(1), requires_grad=False)   # so nnU-Net finds a 'network'
    def forward(self, x):
        out = self.sess.run(["logits"], {"image": x.detach().cpu().numpy().astype(np.float32)})[0]
        return torch.from_numpy(out)

pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                       perform_everything_on_device=False, device=torch.device("cpu"),
                       verbose=False, verbose_preprocessing=False, allow_tqdm=True)
pred.initialize_from_trained_model_folder(model_dir_, use_folds=(FOLD,), checkpoint_name="checkpoint_final.pth")
net = OrtNet(onnx_path)
pred.network = net
pred.list_of_parameters = [net.state_dict()]       # nnU-Net reloads weights per fold; keep it a no-op

imgs = sorted(Path(RAW_DS, "imagesTs").glob("*_0000.nii.gz"))
if a.limit: imgs = imgs[:a.limit]
out_dir = Path(WORK_DIR) / f"{a.which}{RUN_TAG}_onnx_{a.variant}_predictions"
shutil.rmtree(out_dir, ignore_errors=True); out_dir.mkdir(parents=True)
pred.predict_from_files([[str(p)] for p in imgs],
                        [str(out_dir / p.name.replace("_0000.nii.gz", ".nii.gz")) for p in imgs],
                        save_probabilities=False, overwrite=True,
                        num_processes_preprocessing=2, num_processes_segmentation_export=2)

# evaluate only the cases we predicted
ref_dir = out_dir / "_labels"; ref_dir.mkdir()
for p in imgs:
    cid = p.name.replace("_0000.nii.gz", ".nii.gz")
    shutil.copy(Path(RAW_DS, "labelsTs", cid), ref_dir / cid)
subprocess.run(["nnUNetv2_evaluate_folder", str(ref_dir), str(out_dir),
                "-djfile", f"{model_dir_}/dataset.json", "-pfile", f"{model_dir_}/plans.json"], check=True)
summ = json.load(open(out_dir / "summary.json"))
m = summ["foreground_mean"]
print(f"\n{a.which} ONNX {a.variant}: Dice {m['Dice']:.4f}  IoU {m['IoU']:.4f}  ({len(imgs)} cases)")
res_dir = Path(WORK_DIR) / "report_assets"; res_dir.mkdir(exist_ok=True)
shutil.copy(out_dir / "summary.json", res_dir / f"{a.which}{RUN_TAG}_onnx_{a.variant}_summary.json")
