import {
  AlertTriangle,
  Eye,
  Flame,
  RotateCcw,
  ScanSearch,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { useState } from "react";

import type { Prediction, Radiograph } from "../types";
import { Spinner } from "./ui";

export type HeatmapKind = "none" | "gradcam" | "rollout";

interface RadiographViewerProps {
  radiograph: Radiograph;
  imageUrl: string | null;
  imageLoading: boolean;
  imageError: string | null;
  internalCode: string;
  prediction: Prediction | null;
  gradcamUrl: string | null;
  rolloutUrl: string | null;
  heatmapLoading: boolean;
  selectedBoxIndex: number | null;
  onSelectBox: (boxIndex: number | null) => void;
}

export function RadiographViewer({
  radiograph,
  imageUrl,
  imageLoading,
  imageError,
  internalCode,
  prediction,
  gradcamUrl,
  rolloutUrl,
  heatmapLoading,
  selectedBoxIndex,
  onSelectBox,
}: RadiographViewerProps) {
  const [zoom, setZoom] = useState(1);
  const [heatmapKind, setHeatmapKind] = useState<HeatmapKind>("gradcam");
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.5);
  const [showBoxes, setShowBoxes] = useState(true);

  const activeHeatmapUrl =
    heatmapKind === "gradcam" ? gradcamUrl : heatmapKind === "rollout" ? rolloutUrl : null;
  const strokeUnit = Math.max(2, Math.round(Math.max(radiograph.width, radiograph.height) / 220));
  const labelFontSize = Math.max(12, Math.round(Math.max(radiograph.width, radiograph.height) / 36));
  const maxWidthRem = (34 * radiograph.width) / Math.max(1, radiograph.height);

  return (
    <div className="space-y-3">
      <div
        role="toolbar"
        aria-label="Controles del visor radiográfico"
        className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2"
      >
        <div role="group" aria-label="Control de ampliación" className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={() => setZoom((current) => Math.max(1, Number((current - 0.25).toFixed(2))))}
            aria-disabled={zoom <= 1}
            className="inline-flex min-h-8 min-w-8 items-center justify-center rounded-lg border border-slate-200 bg-white px-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-teal-600 aria-disabled:opacity-45"
            aria-label="Reducir zoom"
            title="Reducir zoom"
          >
            <ZoomOut className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
          <span className="min-w-12 text-center font-mono text-xs font-semibold text-slate-700">
            {Math.round(zoom * 100)}%
          </span>
          <button
            type="button"
            onClick={() => setZoom((current) => Math.min(3, Number((current + 0.25).toFixed(2))))}
            aria-disabled={zoom >= 3}
            className="inline-flex min-h-8 min-w-8 items-center justify-center rounded-lg border border-slate-200 bg-white px-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-teal-600 aria-disabled:opacity-45"
            aria-label="Aumentar zoom"
            title="Aumentar zoom"
          >
            <ZoomIn className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
          {zoom > 1 ? (
            <button
              type="button"
              onClick={() => setZoom(1)}
              className="inline-flex min-h-8 items-center gap-1 rounded-lg border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-600 transition hover:bg-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-teal-600"
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
              Ajustar
            </button>
          ) : null}
        </div>

        {prediction ? (
          <div className="flex flex-wrap items-center gap-2.5">
            <div
              role="group"
              aria-label="Capa de explicabilidad visual"
              className="inline-flex rounded-lg border border-slate-200 bg-white p-0.5 text-xs"
            >
              {(
                [
                  { id: "none", label: "Original" },
                  { id: "gradcam", label: "Grad-CAM" },
                  { id: "rollout", label: "Atención ViT" },
                ] as const
              ).map((option) => {
                const pressed = heatmapKind === option.id;
                return (
                  <button
                    key={option.id}
                    type="button"
                    aria-pressed={pressed}
                    onClick={() => setHeatmapKind(option.id)}
                    className={`inline-flex min-h-7 items-center gap-1 rounded-md px-2.5 font-semibold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-teal-600 ${
                      pressed
                        ? "bg-teal-700 text-white"
                        : "text-slate-600 hover:bg-slate-100"
                    }`}
                  >
                    {option.id !== "none" ? <Flame className="h-3 w-3" aria-hidden="true" /> : null}
                    {option.label}
                  </button>
                );
              })}
            </div>

            {heatmapKind !== "none" ? (
              <label className="flex items-center gap-2 text-xs font-medium text-slate-600">
                <span>Opacidad</span>
                <input
                  type="range"
                  min={20}
                  max={90}
                  value={Math.round(heatmapOpacity * 100)}
                  onChange={(event) => setHeatmapOpacity(Number(event.target.value) / 100)}
                  className="h-1.5 w-20 accent-teal-700"
                />
              </label>
            ) : null}

            <button
              type="button"
              aria-pressed={showBoxes}
              onClick={() => setShowBoxes((current) => !current)}
              className={`inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-2.5 text-xs font-semibold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-teal-600 ${
                showBoxes
                  ? "border-amber-300 bg-amber-50 text-amber-900"
                  : "border-slate-200 bg-white text-slate-600 hover:bg-slate-100"
              }`}
            >
              <ScanSearch className="h-3.5 w-3.5" aria-hidden="true" />
              Zonas ({prediction.detections.length})
            </button>
          </div>
        ) : null}
      </div>

      <div className="relative max-h-[38rem] min-h-72 overflow-auto rounded-2xl border border-slate-800 bg-slate-950 p-3">
        {imageLoading ? (
          <div className="flex flex-col items-center justify-center gap-3 py-20 text-slate-400">
            <Spinner className="h-7 w-7" />
            <p className="text-sm">Cargando radiografía…</p>
          </div>
        ) : imageError ? (
          <div className="px-6 py-16 text-center">
            <AlertTriangle className="mx-auto h-8 w-8 text-amber-400" aria-hidden="true" />
            <p className="mt-3 text-sm text-slate-300">{imageError}</p>
          </div>
        ) : imageUrl ? (
          <div
            className="relative mx-auto select-none"
            style={{
              aspectRatio: `${radiograph.width} / ${radiograph.height}`,
              width: `calc(min(100%, ${maxWidthRem.toFixed(3)}rem) * ${zoom})`,
            }}
          >
            <img
              src={imageUrl}
              alt={`Radiografía de la evaluación ${internalCode}`}
              className="block h-full w-full rounded-lg object-fill"
              draggable={false}
            />

            {prediction && heatmapKind !== "none" && activeHeatmapUrl ? (
              <img
                src={activeHeatmapUrl}
                alt={
                  heatmapKind === "gradcam"
                    ? "Mapa de calor Grad-CAM orientado a tuberculosis"
                    : "Mapa de atención acumulada de ViT"
                }
                className="pointer-events-none absolute inset-0 h-full w-full rounded-lg object-fill"
                style={{ opacity: heatmapOpacity }}
                draggable={false}
              />
            ) : null}

            {prediction && showBoxes && prediction.detections.length > 0 ? (
              <svg
                viewBox={`0 0 ${radiograph.width} ${radiograph.height}`}
                className="absolute inset-0 h-full w-full"
                role="img"
                aria-label="Zonas sospechosas detectadas sobre la radiografía"
              >
                {prediction.detections.map((det) => {
                  const [x1, y1, x2, y2] = det.xyxy;
                  const isSelected = selectedBoxIndex === det.box_index;
                  const boxWidth = Math.max(1, x2 - x1);
                  const boxHeight = Math.max(1, y2 - y1);
                  const labelY = y1 > labelFontSize + 8 ? y1 - 6 : y1 + labelFontSize + 4;
                  return (
                    <g
                      key={det.id}
                      className="cursor-pointer"
                      onMouseEnter={() => onSelectBox(det.box_index)}
                      onMouseLeave={() => onSelectBox(null)}
                      onClick={() => onSelectBox(isSelected ? null : det.box_index)}
                    >
                      <rect
                        x={x1}
                        y={y1}
                        width={boxWidth}
                        height={boxHeight}
                        fill={isSelected ? "rgba(251, 146, 60, 0.16)" : "none"}
                        stroke={isSelected ? "#fde047" : "#fb923c"}
                        strokeWidth={isSelected ? strokeUnit * 1.6 : strokeUnit}
                        strokeDasharray={
                          isSelected ? undefined : `${strokeUnit * 3} ${strokeUnit * 1.5}`
                        }
                      />
                      <rect
                        x={x1}
                        y={Math.max(0, labelY - labelFontSize)}
                        width={labelFontSize * 5.2}
                        height={labelFontSize * 1.25}
                        rx={strokeUnit}
                        fill="rgba(15, 23, 42, 0.85)"
                      />
                      <text
                        x={x1 + strokeUnit * 1.5}
                        y={labelY}
                        fill={isSelected ? "#fde047" : "#fdba74"}
                        fontSize={labelFontSize}
                        fontWeight="700"
                        fontFamily="monospace"
                      >
                        {`TB ${(det.confidence * 100).toFixed(0)}%`}
                      </text>
                    </g>
                  );
                })}
              </svg>
            ) : null}

            {prediction && heatmapKind !== "none" && heatmapLoading ? (
              <div className="pointer-events-none absolute right-2 top-2 flex items-center gap-1.5 rounded-lg bg-slate-900/80 px-2.5 py-1 text-xs text-slate-200">
                <Spinner className="h-3.5 w-3.5" />
                Cargando mapa…
              </div>
            ) : null}
          </div>
        ) : null}
      </div>

      {prediction && heatmapKind !== "none" ? (
        <p className="flex items-center gap-1.5 text-xs text-slate-500">
          <Eye className="h-3.5 w-3.5 shrink-0 text-teal-700" aria-hidden="true" />
          {heatmapKind === "gradcam"
            ? "Grad-CAM (clase objetivo TB): señala regiones de mayor contribución al puntaje de tuberculosis."
            : "Mapa de atención acumulada (Rollout): refleja las regiones de mayor foco del transformador visual."}
        </p>
      ) : null}
    </div>
  );
}
