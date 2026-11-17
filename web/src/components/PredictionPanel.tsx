import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ScanSearch,
  ShieldAlert,
} from "lucide-react";

import { formatDate, shortHash } from "../lib/format";
import {
  CLASS_LABELS,
  CONCORDANCE_DESCRIPTIONS,
  CONCORDANCE_LABELS,
  type ConcordanceState,
  type Prediction,
  type PredictionClass,
} from "../types";
import { Notice } from "./ui";
import { cardClass } from "./styles";

const CONCORDANCE_TONES: Record<ConcordanceState, string> = {
  both_alert: "border-rose-200 bg-rose-50 text-rose-900",
  classifier_only_review: "border-amber-200 bg-amber-50 text-amber-900",
  detector_only_review: "border-amber-200 bg-amber-50 text-amber-900",
  neither_alert: "border-emerald-200 bg-emerald-50 text-emerald-900",
};

const CLASS_ORDER: PredictionClass[] = ["TB", "sick", "health"];
const CLASS_BAR_COLORS: Record<PredictionClass, string> = {
  TB: "bg-rose-600",
  sick: "bg-amber-500",
  health: "bg-emerald-600",
};

export function PredictionPanel({
  prediction,
  selectedBoxIndex,
  onSelectBox,
}: {
  prediction: Prediction;
  selectedBoxIndex: number | null;
  onSelectBox: (boxIndex: number | null) => void;
}) {
  const probabilities: Record<PredictionClass, number> = {
    health: prediction.p_health,
    sick: prediction.p_sick,
    TB: prediction.p_tb,
  };

  return (
    <div className="space-y-4">
      <section className={`rounded-2xl border p-4 ${CONCORDANCE_TONES[prediction.concordance_state]}`}>
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <p className="text-xs font-semibold opacity-80">Concordancia operativa</p>
            <h3 className="mt-0.5 text-base font-bold">
              {CONCORDANCE_LABELS[prediction.concordance_state]}
            </h3>
          </div>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-white/85 px-2.5 py-1 text-xs font-semibold">
            {prediction.review_required ? (
              <ShieldAlert className="h-3.5 w-3.5" aria-hidden="true" />
            ) : (
              <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
            )}
            {prediction.review_required ? "Revisión prioritaria" : "Sin alerta combinada"}
          </span>
        </div>
        <p className="mt-2 text-sm leading-5">
          {CONCORDANCE_DESCRIPTIONS[prediction.concordance_state]}
        </p>
      </section>

      {prediction.low_confidence ? (
        <Notice variant="warning" title="Alerta de confianza baja o caso limítrofe">
          <ul className="list-disc space-y-1 pl-4 text-xs leading-5">
            {prediction.low_confidence_reasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
          <p className="mt-1.5 text-xs font-semibold">
            Se recomienda revisión prioritaria por especialista.
          </p>
        </Notice>
      ) : null}

      <section className={`${cardClass} p-5`}>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-xs font-semibold text-teal-700">
              Clasificación global ({prediction.model_version.classifier_display_name})
            </p>
            <h3 className="mt-0.5 text-base font-semibold text-slate-900">
              {CLASS_LABELS[prediction.predicted_class]}
            </h3>
          </div>
          <span
            className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${
              prediction.classifier_alert
                ? "border-rose-200 bg-rose-50 text-rose-800"
                : "border-slate-200 bg-slate-50 text-slate-700"
            }`}
          >
            p_TB {(prediction.p_tb * 100).toFixed(1)}%{" "}
            {prediction.classifier_alert ? "(alerta >= 30%)" : "(< 30%)"}
          </span>
        </div>

        <div className="mt-4 space-y-3">
          {CLASS_ORDER.map((cls) => {
            const prob = probabilities[cls];
            const pct = Math.max(0, Math.min(100, prob * 100));
            return (
              <div key={cls}>
                <div className="mb-1 flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700">{CLASS_LABELS[cls]}</span>
                  <span className="font-mono font-semibold text-slate-900">
                    {pct.toFixed(1)}%
                  </span>
                </div>
                <div className="relative h-2.5 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={`h-full rounded-full ${CLASS_BAR_COLORS[cls]}`}
                    style={{ width: `${pct}%` }}
                  />
                  {cls === "TB" ? (
                    <span
                      title="Corte operativo p_TB = 0,30"
                      className="absolute inset-y-0 w-0.5 bg-slate-900/70"
                      style={{ left: "30%" }}
                    />
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className={`${cardClass} p-5`}>
        <div className="flex items-center justify-between gap-2">
          <div>
            <p className="text-xs font-semibold text-teal-700">Localización focal (YOLO11n)</p>
            <h3 className="mt-0.5 text-base font-semibold text-slate-900">Zonas sospechosas</h3>
          </div>
          <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-semibold text-slate-700">
            {prediction.detector_box_count} detectada{prediction.detector_box_count === 1 ? "" : "s"}
          </span>
        </div>

        {prediction.detections.length === 0 ? (
          <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
            <div className="flex items-start gap-2.5">
              <ScanSearch className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" aria-hidden="true" />
              <p>
                Sin lesiones focales con confianza &gt;= 25% (máxima observada:{" "}
                <span className="font-mono font-semibold">
                  {(prediction.detector_max_conf * 100).toFixed(1)}%
                </span>
                ).
                {prediction.classifier_alert
                  ? " La alerta del clasificador global se mantiene activa."
                  : ""}
              </p>
            </div>
          </div>
        ) : (
          <ul className="mt-3 space-y-2">
            {prediction.detections.map((det) => {
              const [x1, y1, x2, y2] = det.xyxy;
              const selected = selectedBoxIndex === det.box_index;
              return (
                <li key={det.id}>
                  <button
                    type="button"
                    aria-pressed={selected}
                    onClick={() => onSelectBox(selected ? null : det.box_index)}
                    className={`flex w-full items-center justify-between rounded-xl border px-3.5 py-2.5 text-left text-xs transition ${
                      selected
                        ? "border-amber-400 bg-amber-50 text-amber-950"
                        : "border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100"
                    }`}
                  >
                    <span className="font-semibold">Zona #{det.box_index + 1}</span>
                    <span className="font-mono">
                      {(det.confidence * 100).toFixed(0)}% · [{Math.round(x1)}, {Math.round(y1)} →{" "}
                      {Math.round(x2)}, {Math.round(y2)}] px
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <details className={`${cardClass} group overflow-hidden`}>
        <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-5 py-3.5 text-xs font-semibold text-slate-700 hover:bg-slate-50">
          <span className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-teal-700" aria-hidden="true" />
            Trazabilidad del modelo ({prediction.model_version.classifier_display_name} +{" "}
            {prediction.model_version.detector_arch} · {(prediction.inference_ms / 1000).toFixed(1)} s)
          </span>
          <ChevronDown
            className="h-4 w-4 text-slate-400 transition group-open:rotate-180"
            aria-hidden="true"
          />
        </summary>
        <div className="border-t border-slate-100 px-5 py-4">
          <dl className="grid gap-2 text-xs sm:grid-cols-2">
            <div className="rounded-lg bg-slate-50 px-3 py-2">
              <dt className="text-slate-500">Clasificador</dt>
              <dd className="mt-0.5 font-mono font-semibold text-slate-900">
                {prediction.model_version.classifier_display_name} (época{" "}
                {prediction.model_version.classifier_epoch})
              </dd>
            </div>
            <div className="rounded-lg bg-slate-50 px-3 py-2">
              <dt className="text-slate-500">Detector</dt>
              <dd className="mt-0.5 font-mono font-semibold text-slate-900">
                {prediction.model_version.detector_arch} (conf. &gt;={" "}
                {prediction.model_version.detector_threshold})
              </dd>
            </div>
            <div className="rounded-lg bg-slate-50 px-3 py-2">
              <dt className="text-slate-500">Emitida</dt>
              <dd className="mt-0.5 font-semibold text-slate-900">
                {formatDate(prediction.created_at)}
              </dd>
            </div>
            <div className="rounded-lg bg-slate-50 px-3 py-2">
              <dt className="text-slate-500">Tiempo de inferencia</dt>
              <dd className="mt-0.5 font-mono font-semibold text-slate-900">
                {prediction.inference_ms.toFixed(0)} ms
              </dd>
            </div>
          </dl>
          <p className="mt-2 break-all font-mono text-[11px] text-slate-500">
            Contrato: {shortHash(prediction.model_version.contract_sha256)} · ViT:{" "}
            {shortHash(prediction.model_version.classifier_sha256)} · YOLO:{" "}
            {shortHash(prediction.model_version.detector_sha256)}
          </p>
          {!prediction.explainability_verified ? (
            <p className="mt-2 flex items-center gap-1.5 text-xs text-amber-700">
              <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
              El mapa de explicabilidad presentó activación uniforme en esta imagen.
            </p>
          ) : null}
        </div>
      </details>
    </div>
  );
}
