"""
sweep_eval.py - Barridos de evaluacion sobre un checkpoint de DP3 (sin reentrenar).

Ponerlo en  <repo>/3D-Diffusion-Policy/   (junto a train.py y eval.py) y ejecutarlo
desde esa carpeta con el entorno dp3 activo.

Tipos de barrido (+sweep.kind):
  steps    -> num_inference_steps de DDIM (p.ej. 1,2,3,5,10,20)
  noise    -> ruido gaussiano (sigma, en las unidades de la nube de puntos)
  drop     -> fraccion de puntos eliminados al azar (0..0.95), se rellena repitiendo
  npoints  -> numero de puntos tras FPS (p.ej. 512,256,128,64)
  occ      -> oclusion: se borra una esfera de radio r centrada en un punto al azar

Cada condicion se evalua con task.env_runner.eval_episodes episodios y se anota:
exito, IC de Wilson 95%, recompensa media y latencia de predict_action (ms).
Resultados en  <hydra.run.dir>/results.jsonl  y  results.csv
"""
import csv
import json
import math
import pathlib
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch3d.ops import sample_farthest_points

from train import TrainDP3Workspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ---------------------------------------------------------------- utilidades
def wilson(successes, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def refill(pc, keep_idx):
    """pc [N,C]; se queda con keep_idx y rellena hasta N repitiendo puntos conservados."""
    n = pc.shape[0]
    k = keep_idx.numel()
    if k < n:
        extra = keep_idx[torch.randint(0, k, (n - k,), device=pc.device)]
        keep_idx = torch.cat([keep_idx, extra])
    return pc[keep_idx]


def perturb_cloud(pc, kind, v):
    """pc [N,C] (xyz en las 3 primeras columnas). v = intensidad de la perturbacion."""
    n = pc.shape[0]
    if kind == "noise" and v > 0:
        pc = pc.clone()
        pc[:, :3] += v * torch.randn_like(pc[:, :3])
        return pc
    if kind == "drop" and v > 0:
        keep = max(1, int(round((1.0 - v) * n)))
        keep_idx = torch.randperm(n, device=pc.device)[:keep]
        return refill(pc, keep_idx)
    if kind == "npoints" and int(v) < n:
        _, idx = sample_farthest_points(pc[None, :, :3].contiguous(), K=int(v))
        return pc[idx[0]]
    if kind == "occ" and v > 0:
        c = pc[torch.randint(0, n, (1,), device=pc.device)[0], :3]
        keep_idx = torch.nonzero((pc[:, :3] - c).norm(dim=1) > v).squeeze(1)
        if keep_idx.numel() == 0:  # todo ocluido: conserva el punto mas lejano
            keep_idx = (pc[:, :3] - c).norm(dim=1).argmax().view(1)
        return refill(pc, keep_idx)
    return pc


class WrappedPolicy:
    """Envuelve la policy: perturba la nube antes de predict_action y mide la latencia."""

    def __init__(self, policy, kind, value):
        self.policy = policy
        self.kind = kind
        self.value = value
        self.latencies_ms = []
        self._printed = False

    def __getattr__(self, name):  # device, dtype, reset(), etc.
        return getattr(self.policy, name)

    def predict_action(self, obs_dict):
        obs = dict(obs_dict)
        pc = obs["point_cloud"]  # [B, T, N, C]
        if not self._printed:
            xyz = pc[..., :3].reshape(-1, 3)
            print(f"[sweep] point_cloud shape={tuple(pc.shape)}  xyz min={xyz.min().item():.3f} "
                  f"max={xyz.max().item():.3f} std={xyz.std().item():.3f}")
            self._printed = True
        if self.kind in ("noise", "drop", "npoints", "occ"):
            b, t = pc.shape[0], pc.shape[1]
            assert b == 1, "se espera batch 1 en evaluacion"
            pc = torch.stack([perturb_cloud(pc[0, i], self.kind, self.value) for i in range(t)])[None]
            obs["point_cloud"] = pc
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.policy.predict_action(obs)
        torch.cuda.synchronize()
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return out


# ---------------------------------------------------------------- principal
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).resolve().parent / "diffusion_policy_3d" / "config"),
)
def main(cfg):
    sw = cfg.sweep
    kind = str(sw.kind)
    values = list(sw["values"])
    tag = str(sw.get("tag", kind))
    assert kind in ("steps", "noise", "drop", "npoints", "occ"), f"kind desconocido: {kind}"

    ckpt = pathlib.Path(str(sw.ckpt))
    assert ckpt.is_file(), f"No existe el checkpoint: {ckpt}"

    workspace = TrainDP3Workspace(cfg)
    workspace.load_checkpoint(path=ckpt)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    policy.cuda()

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=workspace.output_dir)
    n_ep = int(cfg.task.env_runner.eval_episodes)
    default_steps = int(policy.num_inference_steps)
    out_dir = pathlib.Path(workspace.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] ckpt={ckpt}\n[sweep] kind={kind} values={values} episodes={n_ep} "
          f"use_ema={cfg.training.use_ema} num_inference_steps(base)={default_steps}")

    rows = []
    for v in values:
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        policy.num_inference_steps = int(v) if kind == "steps" else default_steps
        wrapped = WrappedPolicy(policy, kind, float(v))
        t0 = time.time()
        log = env_runner.run(wrapped)
        elapsed = time.time() - t0

        sr = log.get("mean_success_rates", log.get("test_mean_score"))
        sr = float(sr)
        succ = int(round(sr * n_ep))
        lo, hi = wilson(succ, n_ep)
        lat = np.array(wrapped.latencies_ms) if wrapped.latencies_ms else np.array([float("nan")])
        row = dict(
            tag=tag, kind=kind, value=float(v), episodes=n_ep, successes=succ,
            success_rate=sr, ci_low=lo, ci_high=hi,
            mean_traj_rewards=float(log.get("mean_traj_rewards", float("nan"))),
            latency_ms_mean=float(lat.mean()), latency_ms_p95=float(np.percentile(lat, 95)),
            n_calls=int(len(wrapped.latencies_ms)), seconds=elapsed, ckpt=str(ckpt),
        )
        rows.append(row)
        print(f"[sweep] {kind}={v}: success={sr:.3f} [{lo:.3f},{hi:.3f}]  "
              f"lat={row['latency_ms_mean']:.1f} ms (p95 {row['latency_ms_p95']:.1f})  {elapsed:.0f}s")
        with open(out_dir / "results.jsonl", "a") as f:  # se guarda cada condicion al acabar
            f.write(json.dumps(row) + "\n")

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n[sweep] Resumen")
    for r in rows:
        print(f"  {r['kind']}={r['value']:<8g} exito={r['success_rate']:.3f} "
              f"[{r['ci_low']:.3f},{r['ci_high']:.3f}]  lat={r['latency_ms_mean']:.1f} ms")
    print(f"[sweep] Guardado en {out_dir}")


if __name__ == "__main__":
    main()
