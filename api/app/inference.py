"""Motor de inferencia congelado de Fase 5 (ViT-B/16 + YOLO11n + concordancia + Grad-CAM/Rollout)."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image


CLASSES = ("health", "sick", "TB")
TB_CLASS_INDEX = 2
CONCORDANCE_STATES = {
    (True, True): "both_alert",
    (True, False): "classifier_only_review",
    (False, True): "detector_only_review",
    (False, False): "neither_alert",
}


class BundleIntegrityError(RuntimeError):
    """El paquete congelado de Fase 5 falta o no coincide con los hashes verificados."""


class InferenceExecutionError(RuntimeError):
    """Error durante la ejecución o verificación numérica de la inferencia."""


@dataclass(frozen=True)
class FrozenBundleSpec:
    bundle_dir: Path
    manifest: dict[str, Any]
    contract: dict[str, Any]
    data_config: dict[str, Any]
    classifier_weights_path: Path
    detector_weights_path: Path


@dataclass(frozen=True)
class LetterboxGeometry:
    orig_width: int
    orig_height: int
    target_size: int
    resized_width: int
    resized_height: int
    pad_x: int
    pad_y: int


@dataclass(frozen=True)
class InferenceOutput:
    predicted_class: str
    p_health: float
    p_sick: float
    p_tb: float
    classifier_alert: bool
    detector_box_count: int
    detector_max_conf: float
    detector_alert: bool
    concordance_state: str
    review_required: bool
    low_confidence: bool
    low_confidence_reasons: list[str]
    inference_ms: float
    explainability_verified: bool
    detections: list[dict[str, Any]]
    gradcam_png: bytes
    rollout_png: bytes


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle_files(
    bundle_dir: Path,
    *,
    verify_weight_hashes: bool = True,
) -> FrozenBundleSpec:
    manifest_path = bundle_dir / "deployment_manifest.json"
    if not manifest_path.is_file():
        raise BundleIntegrityError(f"Falta deployment_manifest.json en {bundle_dir}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "tbx_deployment_bundle_v1":
        raise BundleIntegrityError("Esquema inválido en deployment_manifest.json.")

    contract_path = bundle_dir / str(manifest["contract_file"])
    data_config_path = bundle_dir / str(manifest["data_config_file"])
    classifier_path = bundle_dir / str(manifest["classifier_weights"])
    detector_path = bundle_dir / str(manifest["detector_weights"])
    for required in (contract_path, data_config_path, classifier_path, detector_path):
        if not required.is_file():
            raise BundleIntegrityError(f"Archivo requerido ausente del paquete congelado: {required.name}")

    if file_sha256(contract_path) != manifest["contract_sha256"]:
        raise BundleIntegrityError("SHA-256 de phase5_frozen_contract.json no coincide.")
    if file_sha256(data_config_path) != manifest["data_config_sha256"]:
        raise BundleIntegrityError("SHA-256 de data_config.json no coincide.")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    data_config = json.loads(data_config_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "phase5_frozen_contract_v1" or contract.get("decisions_frozen") is not True:
        raise BundleIntegrityError("El contrato congelado no tiene decisions_frozen=true.")
    if contract["preprocessing"]["data_config_sha256"] != manifest["data_config_sha256"]:
        raise BundleIntegrityError("data_config_sha256 inconsistente entre contrato y manifiesto.")
    if contract["classifier"]["checkpoint_sha256"] != manifest["classifier_sha256"]:
        raise BundleIntegrityError("classifier_sha256 inconsistente entre contrato y manifiesto.")
    if contract["detector"]["weights_sha256"] != manifest["detector_sha256"]:
        raise BundleIntegrityError("detector_sha256 inconsistente entre contrato y manifiesto.")
    if tuple(data_config["classes"].keys()) != CLASSES:
        raise BundleIntegrityError("Mapa de clases distinto de health/sick/TB.")

    if verify_weight_hashes:
        if file_sha256(classifier_path) != manifest["classifier_sha256"]:
            raise BundleIntegrityError("SHA-256 de vit_b_16_best.pt no coincide con el contrato congelado.")
        if file_sha256(detector_path) != manifest["detector_sha256"]:
            raise BundleIntegrityError("SHA-256 de yolo11n_best.pt no coincide con el contrato congelado.")

    return FrozenBundleSpec(
        bundle_dir=bundle_dir,
        manifest=manifest,
        contract=contract,
        data_config=data_config,
        classifier_weights_path=classifier_path,
        detector_weights_path=detector_path,
    )


def compute_concordance(
    p_tb: float,
    boxes: Sequence[dict[str, Any]],
    *,
    classifier_threshold: float = 0.30,
    detector_threshold: float = 0.25,
) -> dict[str, Any]:
    """Regla congelada additive_4state_no_suppression_v1: no suprime alertas del clasificador."""
    ca = bool(p_tb >= classifier_threshold)
    da = any(float(box["confidence"]) >= detector_threshold for box in boxes)
    state = CONCORDANCE_STATES[(ca, da)]
    return {
        "classifier_alert": ca,
        "detector_alert": da,
        "review_required": bool(ca or da),
        "state": state,
    }


def evaluate_low_confidence(
    probabilities: Sequence[float],
    *,
    concordance_state: str,
    low_confidence_threshold: float = 0.70,
    operational_tb_threshold: float = 0.30,
) -> tuple[bool, list[str]]:
    """Regla fija de HU12 hasta habilitar parametrización administrativa (HU21)."""
    reasons: list[str] = []
    max_prob = max(float(p) for p in probabilities)
    p_tb = float(probabilities[TB_CLASS_INDEX])
    if max_prob < low_confidence_threshold:
        reasons.append(
            f"Probabilidad máxima de clase ({max_prob:.1%}) por debajo del umbral de referencia ({low_confidence_threshold:.0%})."
        )
    if 0.20 <= p_tb < 0.50:
        reasons.append(
            f"Probabilidad de TB ({p_tb:.1%}) cercana al punto de corte operativo ({operational_tb_threshold:.0%})."
        )
    if concordance_state == "classifier_only_review":
        reasons.append("Alerta del clasificador sin cajas focales de YOLO11n (>= 0,25); requiere revisión clínica sin supresión.")
    elif concordance_state == "detector_only_review":
        reasons.append("Detección focal de YOLO11n con p_TB por debajo de 0,30; requiere contraste radiológico.")
    return bool(reasons), reasons


def apply_letterbox(image: "Image.Image", target_size: int = 512) -> tuple["Image.Image", LetterboxGeometry]:
    from PIL import Image, ImageOps

    rgb = image.convert("RGB")
    orig_w, orig_h = rgb.size
    contained = ImageOps.contain(rgb, (target_size, target_size), method=Image.Resampling.LANCZOS)
    resized_w, resized_h = contained.size
    pad_x = int((target_size - resized_w) * 0.5)
    pad_y = int((target_size - resized_h) * 0.5)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    canvas.paste(contained, (pad_x, pad_y))
    return canvas, LetterboxGeometry(
        orig_width=orig_w,
        orig_height=orig_h,
        target_size=target_size,
        resized_width=resized_w,
        resized_height=resized_h,
        pad_x=pad_x,
        pad_y=pad_y,
    )


def project_letterbox_box(
    xyxy: Sequence[float],
    geom: LetterboxGeometry,
) -> tuple[list[float], list[float]]:
    lx1, ly1, lx2, ly2 = (float(v) for v in xyxy)
    scale_x = geom.orig_width / max(1, geom.resized_width)
    scale_y = geom.orig_height / max(1, geom.resized_height)
    ox1 = max(0.0, min(float(geom.orig_width), (lx1 - geom.pad_x) * scale_x))
    oy1 = max(0.0, min(float(geom.orig_height), (ly1 - geom.pad_y) * scale_y))
    ox2 = max(ox1, min(float(geom.orig_width), (lx2 - geom.pad_x) * scale_x))
    oy2 = max(oy1, min(float(geom.orig_height), (ly2 - geom.pad_y) * scale_y))
    orig_box = [round(ox1, 2), round(oy1, 2), round(ox2, 2), round(oy2, 2)]
    norm_box = [
        round(ox1 / max(1, geom.orig_width), 6),
        round(oy1 / max(1, geom.orig_height), 6),
        round(ox2 / max(1, geom.orig_width), 6),
        round(oy2 / max(1, geom.orig_height), 6),
    ]
    return orig_box, norm_box


def _jet_rgba_lut() -> "np.ndarray":
    import numpy as np

    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    alpha = np.where(x <= 0.06, 0.0, np.clip((x - 0.06) / 0.94, 0.0, 1.0) * 0.85)
    lut = np.stack([r, g, b, alpha], axis=1)
    return np.clip(np.rint(lut * 255.0), 0, 255).astype(np.uint8)


def colorize_heatmap_png(
    heatmap_2d: "np.ndarray",
    geom: LetterboxGeometry,
    *,
    max_display_dim: int = 1024,
) -> bytes:
    import io
    import numpy as np
    from PIL import Image

    arr = np.asarray(heatmap_2d, dtype=np.float32)
    cropped = arr[
        geom.pad_y : geom.pad_y + geom.resized_height,
        geom.pad_x : geom.pad_x + geom.resized_width,
    ]
    if cropped.size == 0:
        cropped = arr
    mn = float(cropped.min())
    mx = float(cropped.max())
    if mx > mn + 1e-12:
        norm = np.clip((cropped - mn) / (mx - mn), 0.0, 1.0)
    else:
        norm = np.zeros_like(cropped, dtype=np.float32)

    indices = np.clip(np.rint(norm * 255.0), 0, 255).astype(np.uint8)
    rgba = _jet_rgba_lut()[indices]
    pil_rgba = Image.fromarray(rgba, mode="RGBA")

    scale = min(1.0, max_display_dim / max(1, geom.orig_width, geom.orig_height))
    out_w = max(1, int(round(geom.orig_width * scale)))
    out_h = max(1, int(round(geom.orig_height * scale)))
    if pil_rgba.size != (out_w, out_h):
        pil_rgba = pil_rgba.resize((out_w, out_h), resample=Image.Resampling.BILINEAR)

    buffer = io.BytesIO()
    pil_rgba.save(buffer, format="PNG")
    return buffer.getvalue()


def _normalize_2d_tensor(tensor_2d) -> tuple["np.ndarray", bool]:
    import numpy as np
    import torch

    arr = tensor_2d.detach().cpu().numpy().astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if not np.isfinite(arr).all() or mx <= mn + 1e-12:
        return np.zeros_like(arr, dtype=np.float32), False
    return ((arr - mn) / (mx - mn)).astype(np.float32), True


def _upsample_2d(cam_2d: "np.ndarray", size: int) -> "np.ndarray":
    import numpy as np
    import torch
    from torch.nn import functional as F

    tensor = torch.from_numpy(np.asarray(cam_2d, dtype=np.float32))[None, None]
    up = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
    mn, mx = float(up.min()), float(up.max())
    if mx > mn + 1e-12:
        up = (up - mn) / (mx - mn)
    return up.astype(np.float32)


class _FrozenViTClassifier:
    """Wrapper compatible con el state_dict guardado en phase3_vit_b_16 (ViTClassifier.encoder)."""

    @staticmethod
    def build(image_size: int = 512):
        import torch.nn as nn
        from torchvision.models import vit_b_16

        class Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.image_size = int(image_size)
                self.encoder = vit_b_16(weights=None, image_size=self.image_size)
                self.encoder.heads.head = nn.Linear(self.encoder.heads.head.in_features, 3)

            def forward(self, x):
                return self.encoder(x)

        return Wrapper()


class FrozenPhase5Engine:
    """Motor lazy-loaded y thread-safe para inferencia bajo el contrato congelado de Fase 5."""

    def __init__(
        self,
        bundle_dir: Path,
        *,
        device: str = "auto",
        low_confidence_threshold: float = 0.70,
    ) -> None:
        self.bundle_dir = bundle_dir
        self.requested_device = device
        self.low_confidence_threshold = low_confidence_threshold
        self._lock = threading.Lock()
        self._loaded = False
        self._spec: FrozenBundleSpec | None = None
        self._device: str = "cpu"
        self._vit_model: Any = None
        self._yolo_model: Any = None

    def _resolve_device(self) -> str:
        import torch

        if self.requested_device == "cpu":
            return "cpu"
        if self.requested_device == "cuda":
            if not torch.cuda.is_available():
                raise InferenceExecutionError("APP_INFERENCE_DEVICE=cuda solicitado pero CUDA no está disponible.")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self) -> FrozenBundleSpec:
        if self._loaded and self._spec is not None:
            return self._spec

        import torch
        from ultralytics import YOLO

        spec = verify_bundle_files(self.bundle_dir, verify_weight_hashes=True)
        device = self._resolve_device()
        image_size = int(spec.data_config["image_size"])

        payload = torch.load(spec.classifier_weights_path, map_location="cpu", weights_only=True)
        cls_cfg = spec.contract["classifier"]
        if (
            payload.get("architecture") != cls_cfg["architecture"]
            or payload.get("run_id") != cls_cfg["run_id"]
            or payload.get("epoch") != cls_cfg["best_epoch"]
            or list(payload.get("classes", [])) != list(CLASSES)
        ):
            raise BundleIntegrityError("Metadatos internos de vit_b_16_best.pt no coinciden con el contrato.")

        vit = _FrozenViTClassifier.build(image_size=image_size)
        vit.load_state_dict(payload["model_state"], strict=True)
        vit = vit.to(device).eval()

        yolo = YOLO(str(spec.detector_weights_path))
        if yolo.names != {0: "tb_lesion"}:
            raise BundleIntegrityError(f"Clases inesperadas en detector YOLO11n: {yolo.names}")

        self._spec = spec
        self._device = device
        self._vit_model = vit
        self._yolo_model = yolo
        self._loaded = True
        return spec

    def analyze_image(self, image_path: Path) -> InferenceOutput:
        import numpy as np
        import torch
        from PIL import Image
        from torch.nn import functional as F

        with self._lock:
            spec = self._ensure_loaded()
            t0 = time.perf_counter()
            image_size = int(spec.data_config["image_size"])
            mean = np.asarray(spec.data_config["normalization"]["mean"], dtype=np.float32)[:, None, None]
            std = np.asarray(spec.data_config["normalization"]["std"], dtype=np.float32)[:, None, None]

            with Image.open(image_path) as im:
                letterboxed, geom = apply_letterbox(im, image_size)

            rgb_array = np.asarray(letterboxed, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.0)
            norm_array = np.ascontiguousarray((rgb_array - mean) / std, dtype=np.float32)
            x_tensor = torch.from_numpy(norm_array).unsqueeze(0).to(self._device)

            # 1. Clasificador ViT-B/16 + Grad-CAM sobre tokens espaciales (layers[-2]).
            captured: dict[str, Any] = {}
            self._vit_model.zero_grad(set_to_none=True)
            x_in = x_tensor.detach().clone().requires_grad_(True)

            def _save_tokens(_mod, _inp, output):
                tensor = output[0] if isinstance(output, tuple) else output
                tensor.retain_grad()
                captured["tokens"] = tensor

            hook = self._vit_model.encoder.encoder.layers[-2].register_forward_hook(_save_tokens)
            try:
                logits = self._vit_model(x_in)
                probs = logits.softmax(dim=1).detach().cpu().numpy()[0]
                logits[:, TB_CLASS_INDEX].sum().backward()
            finally:
                hook.remove()

            try:
                tokens = captured.get("tokens")
                if tokens is None or tokens.grad is None:
                    raise InferenceExecutionError("No se capturó el gradiente espacial de ViT-B/16.")
                if not bool(torch.isfinite(tokens).all().item()) or not bool(torch.isfinite(tokens.grad).all().item()):
                    raise InferenceExecutionError("Activaciones o gradientes no finitos en ViT-B/16.")
                grad_norm = float(tokens.grad.detach().norm().item())
                if grad_norm <= 0.0:
                    raise InferenceExecutionError("Gradiente nulo al calcular Grad-CAM para la clase TB.")

                grid = image_size // 16
                act_4d = tokens[:, 1:, :].transpose(1, 2).reshape(1, tokens.shape[-1], grid, grid)
                grad_4d = tokens.grad[:, 1:, :].transpose(1, 2).reshape(1, tokens.grad.shape[-1], grid, grid)
                weights = grad_4d.mean(dim=(2, 3), keepdim=True)
                raw_cam = F.relu((weights * act_4d).sum(dim=1))[0]
                cam_norm, cam_nontrivial = _normalize_2d_tensor(raw_cam)
                gradcam_512 = _upsample_2d(cam_norm, image_size)

                # 2. Attention Rollout en modo sin gradientes sobre el mismo encoder.
                with torch.no_grad():
                    rollout_512, rollout_nontrivial = self._compute_vit_rollout(x_in.detach(), image_size)
            finally:
                self._vit_model.zero_grad(set_to_none=True)

            # 3. Detector YOLO11n sobre la misma imagen 512x512 letterboxed.
            det_cfg = spec.contract["detector"]
            det_thresh = float(det_cfg["confidence_threshold"])
            det_opts = dict(det_cfg["inference_options"])
            det_opts.update(
                device=0 if self._device == "cuda" else "cpu",
                save=False,
                verbose=False,
            )
            results = self._yolo_model.predict(source=[letterboxed], **det_opts)
            if len(results) != 1:
                raise InferenceExecutionError("Salida inesperada del detector YOLO11n.")
            res = results[0]
            raw_boxes = [
                {"xyxy": xy, "confidence": float(score)}
                for xy, score in zip(res.boxes.xyxy.cpu().tolist(), res.boxes.conf.cpu().tolist())
            ]
            kept_boxes = sorted(
                (box for box in raw_boxes if box["confidence"] >= det_thresh),
                key=lambda item: -item["confidence"],
            )

            detections: list[dict[str, Any]] = []
            for idx, box in enumerate(kept_boxes):
                orig_xyxy, norm_xyxy = project_letterbox_box(box["xyxy"], geom)
                detections.append(
                    {
                        "box_index": idx,
                        "label": "tb_lesion",
                        "confidence": round(float(box["confidence"]), 6),
                        "xyxy": orig_xyxy,
                        "normalized_xyxy": norm_xyxy,
                        "letterbox_xyxy": [round(float(v), 2) for v in box["xyxy"]],
                    }
                )

            p_health = float(probs[0])
            p_sick = float(probs[1])
            p_tb = float(probs[2])
            predicted_class = CLASSES[int(probs.argmax())]
            cls_thresh = float(spec.contract["classifier"]["operational_tb_threshold"])
            conc = compute_concordance(
                p_tb,
                kept_boxes,
                classifier_threshold=cls_thresh,
                detector_threshold=det_thresh,
            )
            low_conf, low_reasons = evaluate_low_confidence(
                (p_health, p_sick, p_tb),
                concordance_state=conc["state"],
                low_confidence_threshold=self.low_confidence_threshold,
                operational_tb_threshold=cls_thresh,
            )

            gradcam_png = colorize_heatmap_png(gradcam_512, geom)
            rollout_png = colorize_heatmap_png(rollout_512, geom)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            return InferenceOutput(
                predicted_class=predicted_class,
                p_health=round(p_health, 6),
                p_sick=round(p_sick, 6),
                p_tb=round(p_tb, 6),
                classifier_alert=bool(conc["classifier_alert"]),
                detector_box_count=len(detections),
                detector_max_conf=round(max((b["confidence"] for b in raw_boxes), default=0.0), 6),
                detector_alert=bool(conc["detector_alert"]),
                concordance_state=str(conc["state"]),
                review_required=bool(conc["review_required"]),
                low_confidence=low_conf,
                low_confidence_reasons=low_reasons,
                inference_ms=elapsed_ms,
                explainability_verified=bool(cam_nontrivial and rollout_nontrivial),
                detections=detections,
                gradcam_png=gradcam_png,
                rollout_png=rollout_png,
            )

    def _compute_vit_rollout(self, x_tensor, image_size: int) -> tuple["np.ndarray", bool]:
        import torch

        vit_enc = self._vit_model.encoder
        tokens = vit_enc._process_input(x_tensor)
        batch = x_tensor.shape[0]
        cls_tok = vit_enc.class_token.expand(batch, -1, -1)
        seq = torch.cat([cls_tok, tokens], dim=1)
        seq = seq + vit_enc.encoder.pos_embedding
        seq = vit_enc.encoder.dropout(seq)

        n_tokens = seq.shape[1]
        eye = torch.eye(n_tokens, device=seq.device, dtype=seq.dtype).unsqueeze(0)
        rollout = eye.expand(batch, -1, -1).clone()

        for block in vit_enc.encoder.layers:
            hidden = block.ln_1(seq)
            attn_out, attn_weights = block.self_attention(
                hidden, hidden, hidden, need_weights=True, average_attn_weights=False
            )
            seq = seq + block.dropout(attn_out)
            seq = seq + block.mlp(block.ln_2(seq))
            attn_mean = attn_weights.mean(dim=1)
            aug = 0.5 * attn_mean + 0.5 * eye
            aug = aug / aug.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            rollout = torch.bmm(aug, rollout)

        grid = int(round((n_tokens - 1) ** 0.5))
        cls_to_patches = rollout[:, 0, 1:].reshape(batch, grid, grid)[0]
        norm_map, nontrivial = _normalize_2d_tensor(cls_to_patches)
        return _upsample_2d(norm_map, image_size), nontrivial
