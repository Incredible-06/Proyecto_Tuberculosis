import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  ChevronDown,
  Cpu,
  FileImage,
  Lock,
  Play,
  RefreshCw,
  Sparkles,
  Upload,
} from "lucide-react";
import { useEffect, useState, type DragEvent } from "react";
import { Link, useParams } from "react-router-dom";

import { ApiError, api } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { ObservationsPanel } from "../components/ObservationsPanel";
import { PredictionPanel } from "../components/PredictionPanel";
import { RadiographViewer } from "../components/RadiographViewer";
import { buttonPrimary, buttonSecondary, cardClass } from "../components/styles";
import { LoadingPanel, Notice, PageHeader, Spinner, StatusBadge } from "../components/ui";
import { formatBytes, formatDate, shortHash } from "../lib/format";
import type { Evaluation, EvaluationStatus, Observation, Prediction } from "../types";

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/png", "image/jpeg"]);

const WORKFLOW_STEPS: Array<{ status: EvaluationStatus; label: string }> = [
  { status: "pendiente_imagen", label: "Registro" },
  { status: "imagen_cargada", label: "Imagen" },
  { status: "lista_para_analisis", label: "Calidad OK" },
  { status: "analizada", label: "Analizada" },
  { status: "revisada", label: "Revisada" },
];

function validateSelectedFile(file: File): string | null {
  if (!ALLOWED_TYPES.has(file.type)) {
    return "Solo se admiten imágenes PNG o JPG. El formato DICOM no está permitido.";
  }
  if (file.size === 0) {
    return "El archivo está vacío.";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return "La imagen supera el límite de 10 MB.";
  }
  return null;
}

export function EvaluationDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { can } = useAuth();
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [observations, setObservations] = useState<Observation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState<string | null>(null);

  const [gradcamUrl, setGradcamUrl] = useState<string | null>(null);
  const [rolloutUrl, setRolloutUrl] = useState<string | null>(null);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [selectedBoxIndex, setSelectedBoxIndex] = useState<number | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [replaceMode, setReplaceMode] = useState(false);
  const [dragging, setDragging] = useState(false);

  const [validatingQuality, setValidatingQuality] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);

  const canUpload = can("radiographs.upload");
  const canReadPredictions = can("predictions.read");
  const canReadObservations = can("observations.read");
  const canWriteObservations = can("observations.write");
  const canAnalyze = canUpload || canWriteObservations;

  useEffect(() => {
    if (!id) {
      return;
    }
    let active = true;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const loaded = await api.getEvaluation(id);
        if (!active) {
          return;
        }
        setEvaluation(loaded);

        if (loaded.has_prediction && canReadPredictions) {
          const pred = await api.getPrediction(id);
          if (active) {
            setPrediction(pred);
          }
        } else if (active) {
          setPrediction(null);
        }

        if (canReadObservations) {
          const obs = await api.listObservations(id);
          if (active) {
            setObservations(obs);
          }
        }
      } catch (loadError) {
        if (active) {
          setError(
            loadError instanceof ApiError
              ? loadError.message
              : "No se pudo cargar la evaluación.",
          );
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };
    void load();
    return () => {
      active = false;
    };
  }, [id, canReadPredictions, canReadObservations]);

  useEffect(() => {
    const contentUrl = evaluation?.radiograph?.content_url;
    const cacheKey = evaluation?.radiograph?.id;
    if (!contentUrl || !cacheKey) {
      setImageUrl(null);
      setImageError(null);
      return;
    }
    let active = true;
    let objectUrl: string | null = null;
    setImageLoading(true);
    setImageError(null);
    api
      .fetchRadiograph(`${contentUrl}?v=${encodeURIComponent(cacheKey)}`)
      .then((blob) => {
        const created = URL.createObjectURL(blob);
        if (!active) {
          URL.revokeObjectURL(created);
          return;
        }
        objectUrl = created;
        setImageUrl(created);
      })
      .catch((fetchError: unknown) => {
        if (active) {
          setImageError(
            fetchError instanceof Error
              ? fetchError.message
              : "No se pudo cargar la vista previa.",
          );
        }
      })
      .finally(() => {
        if (active) {
          setImageLoading(false);
        }
      });
    return () => {
      active = false;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [evaluation?.radiograph?.content_url, evaluation?.radiograph?.id]);

  useEffect(() => {
    if (!prediction || !canReadPredictions) {
      setGradcamUrl(null);
      setRolloutUrl(null);
      return;
    }
    let active = true;
    let createdGrad: string | null = null;
    let createdRoll: string | null = null;
    setHeatmapLoading(true);
    Promise.all([
      api.fetchExplainabilityMap(prediction.gradcam_url),
      api.fetchExplainabilityMap(prediction.rollout_url),
    ])
      .then(([gradBlob, rollBlob]) => {
        createdGrad = URL.createObjectURL(gradBlob);
        createdRoll = URL.createObjectURL(rollBlob);
        if (!active) {
          URL.revokeObjectURL(createdGrad);
          URL.revokeObjectURL(createdRoll);
          return;
        }
        setGradcamUrl(createdGrad);
        setRolloutUrl(createdRoll);
      })
      .catch(() => {
        if (active) {
          setGradcamUrl(null);
          setRolloutUrl(null);
        }
      })
      .finally(() => {
        if (active) {
          setHeatmapLoading(false);
        }
      });
    return () => {
      active = false;
      if (createdGrad) {
        URL.revokeObjectURL(createdGrad);
      }
      if (createdRoll) {
        URL.revokeObjectURL(createdRoll);
      }
    };
  }, [prediction, canReadPredictions]);

  const chooseFile = (file: File | null) => {
    setNotice(null);
    setFileError(null);
    if (!file) {
      setSelectedFile(null);
      return;
    }
    const validationError = validateSelectedFile(file);
    if (validationError) {
      setSelectedFile(null);
      setFileError(validationError);
      return;
    }
    setSelectedFile(file);
  };

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files?.[0] ?? null);
  };

  const handleUpload = async () => {
    if (!evaluation || !selectedFile) {
      return;
    }
    setUploading(true);
    setFileError(null);
    setNotice(null);
    try {
      if (evaluation.radiograph && replaceMode) {
        await api.replaceRadiograph(evaluation.id, selectedFile);
        setNotice("Radiografía reemplazada. Verifique ahora la calidad de imagen.");
      } else {
        await api.uploadRadiograph(evaluation.id, selectedFile);
        setNotice("Radiografía cargada. Verifique la calidad para habilitar el análisis.");
      }
      const refreshed = await api.getEvaluation(evaluation.id);
      setEvaluation(refreshed);
      setSelectedFile(null);
      setReplaceMode(false);
    } catch (uploadError) {
      setFileError(
        uploadError instanceof ApiError
          ? uploadError.message
          : "No se pudo cargar la radiografía.",
      );
    } finally {
      setUploading(false);
    }
  };

  const handleValidateQuality = async () => {
    if (!evaluation) {
      return;
    }
    setValidatingQuality(true);
    setError(null);
    setNotice(null);
    try {
      const qc = await api.validateRadiographQuality(evaluation.id);
      const refreshed = await api.getEvaluation(evaluation.id);
      setEvaluation(refreshed);
      if (qc.passed) {
        setNotice("Imagen verificada y lista para análisis.");
      } else {
        setError("La radiografía no superó los criterios mínimos de calidad.");
      }
    } catch (qcError) {
      setError(
        qcError instanceof ApiError
          ? qcError.message
          : "No se pudo validar la calidad de la imagen.",
      );
    } finally {
      setValidatingQuality(false);
    }
  };

  const handleAnalyze = async () => {
    if (!evaluation) {
      return;
    }
    setAnalyzing(true);
    setError(null);
    setNotice(null);
    try {
      const updated = await api.analyzeEvaluation(evaluation.id);
      setEvaluation(updated);
      if (canReadPredictions) {
        const pred = await api.getPrediction(evaluation.id);
        setPrediction(pred);
      }
      setNotice(
        canReadPredictions
          ? "Análisis completado."
          : "Análisis completado. El estudio quedó disponible para lectura médica.",
      );
    } catch (analyzeError) {
      setError(
        analyzeError instanceof ApiError
          ? analyzeError.message
          : "No se pudo completar el análisis.",
      );
    } finally {
      setAnalyzing(false);
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <PageHeader eyebrow="Evaluación" title="Cargando estudio…" />
        <LoadingPanel />
      </div>
    );
  }

  if (!evaluation) {
    return (
      <div className="space-y-6">
        <PageHeader
          eyebrow="Evaluación"
          title="Evaluación no disponible"
          actions={
            <Link to="/evaluaciones" className={buttonSecondary}>
              <ArrowLeft className="h-4 w-4" aria-hidden="true" />
              Volver al listado
            </Link>
          }
        />
        <Notice variant="error">{error ?? "La evaluación solicitada no existe."}</Notice>
      </div>
    );
  }

  const radiograph = evaluation.radiograph;
  const qualityCheck = evaluation.quality_check;
  const currentStepIndex = WORKFLOW_STEPS.findIndex((s) => s.status === evaluation.status);

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Estudio radiográfico"
        title={evaluation.internal_code}
        description={evaluation.notes ?? "Evaluación clínica anonimizada."}
        actions={
          <Link to="/evaluaciones" className={buttonSecondary}>
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
            Volver al listado
          </Link>
        }
      />

      {error ? (
        <Notice variant="error" onClose={() => setError(null)}>
          {error}
        </Notice>
      ) : null}
      {notice ? (
        <Notice variant="success" onClose={() => setNotice(null)}>
          {notice}
        </Notice>
      ) : null}

      <nav aria-label="Progreso del estudio" className={`${cardClass} px-5 py-3.5`}>
        <ol className="grid gap-2.5 sm:grid-cols-5">
          {WORKFLOW_STEPS.map((step, index) => {
            const done = index < currentStepIndex || evaluation.status === "revisada";
            const current = index === currentStepIndex && evaluation.status !== "revisada";
            return (
              <li
                key={step.status}
                aria-current={current ? "step" : undefined}
                className={`flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-semibold ${
                  current
                    ? "border-teal-300 bg-teal-50 text-teal-900"
                    : done
                      ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                      : "border-slate-200 bg-slate-50 text-slate-500"
                }`}
              >
                <span
                  className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] ${
                    current
                      ? "bg-teal-700 text-white"
                      : done
                        ? "bg-emerald-600 text-white"
                        : "bg-slate-200 text-slate-600"
                  }`}
                >
                  {done ? "✓" : index + 1}
                </span>
                <span>{step.label}</span>
              </li>
            );
          })}
        </ol>
      </nav>

      <div className="grid gap-6 xl:grid-cols-[1.35fr_0.65fr]">
        <div className="space-y-6">
          <section className={`${cardClass} overflow-hidden`}>
            <div className="flex flex-col gap-3 border-b border-slate-100 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-base font-semibold text-slate-900">Radiografía de tórax</h2>
                <p className="mt-0.5 text-sm text-slate-600">
                  {prediction
                    ? "Use los controles superiores para ampliar, alternar mapas visuales y resaltar zonas."
                    : radiograph
                      ? `${radiograph.width} × ${radiograph.height} px · ${radiograph.content_type === "image/png" ? "PNG" : "JPG"}`
                      : "Cargue una radiografía en formato PNG o JPG."}
                </p>
              </div>
              <StatusBadge status={evaluation.status} />
            </div>

            <div className="p-5">
              {radiograph && !replaceMode ? (
                <RadiographViewer
                  radiograph={radiograph}
                  imageUrl={imageUrl}
                  imageLoading={imageLoading}
                  imageError={imageError}
                  internalCode={evaluation.internal_code}
                  prediction={prediction}
                  gradcamUrl={gradcamUrl}
                  rolloutUrl={rolloutUrl}
                  heatmapLoading={heatmapLoading}
                  selectedBoxIndex={selectedBoxIndex}
                  onSelectBox={setSelectedBoxIndex}
                />
              ) : canUpload ? (
                <div className="space-y-4">
                  {fileError ? <Notice variant="error">{fileError}</Notice> : null}
                  {replaceMode ? (
                    <div className="flex items-center justify-between rounded-xl border border-amber-200 bg-amber-50 px-4 py-2.5 text-sm text-amber-900">
                      <span>Seleccione la nueva imagen PNG o JPG para reemplazar la anterior.</span>
                      <button
                        type="button"
                        className={buttonSecondary}
                        onClick={() => {
                          setReplaceMode(false);
                          chooseFile(null);
                        }}
                      >
                        Cancelar
                      </button>
                    </div>
                  ) : null}
                  <label
                    className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 text-center transition ${
                      dragging
                        ? "border-teal-500 bg-teal-50"
                        : "border-slate-300 bg-slate-50 hover:border-teal-400 hover:bg-teal-50/50"
                    }`}
                    onDragOver={(event) => {
                      event.preventDefault();
                      setDragging(true);
                    }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={handleDrop}
                  >
                    <input
                      type="file"
                      className="sr-only"
                      accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                      onChange={(event) => chooseFile(event.target.files?.[0] ?? null)}
                      disabled={uploading}
                    />
                    <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white text-teal-700 shadow-sm ring-1 ring-slate-200">
                      <Upload className="h-6 w-6" aria-hidden="true" />
                    </span>
                    <span className="mt-4 text-sm font-semibold text-slate-900">
                      Arrastre la radiografía o haga clic para seleccionar
                    </span>
                    <span className="mt-1 text-xs leading-5 text-slate-500">
                      PNG o JPG · máximo 10 MB · DICOM no admitido
                    </span>
                  </label>

                  {selectedFile ? (
                    <div className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
                      <div className="flex min-w-0 items-center gap-3">
                        <FileImage className="h-5 w-5 shrink-0 text-teal-700" aria-hidden="true" />
                        <div className="min-w-0">
                          <p className="truncate text-sm font-semibold text-slate-900">
                            {selectedFile.name}
                          </p>
                          <p className="text-xs text-slate-500">
                            {formatBytes(selectedFile.size)} · {selectedFile.type || "sin tipo MIME"}
                          </p>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          className={buttonSecondary}
                          onClick={() => chooseFile(null)}
                          disabled={uploading}
                        >
                          Quitar
                        </button>
                        <button
                          type="button"
                          className={buttonPrimary}
                          onClick={() => void handleUpload()}
                          disabled={uploading}
                        >
                          {uploading ? <Spinner className="h-4 w-4" /> : <Upload className="h-4 w-4" />}
                          {uploading ? "Cargando…" : replaceMode ? "Reemplazar imagen" : "Cargar imagen"}
                        </button>
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="rounded-2xl border border-slate-200 bg-slate-50 px-5 py-10 text-center">
                  <Lock className="mx-auto h-8 w-8 text-slate-400" aria-hidden="true" />
                  <h3 className="mt-3 text-sm font-semibold text-slate-900">
                    Esta evaluación aún no tiene radiografía
                  </h3>
                  <p className="mx-auto mt-1 max-w-md text-sm leading-6 text-slate-600">
                    La carga la realiza el tecnólogo médico en radiología.
                  </p>
                </div>
              )}
            </div>
          </section>

          {canReadObservations ? (
            <ObservationsPanel
              evaluationId={evaluation.id}
              status={evaluation.status}
              canWrite={canWriteObservations}
              observations={observations}
              onCreated={(created, markedReviewed) => {
                setObservations((current) => [created, ...current]);
                setEvaluation((current) =>
                  current
                    ? {
                        ...current,
                        status: markedReviewed ? "revisada" : current.status,
                        observation_count: current.observation_count + 1,
                      }
                    : current,
                );
                setNotice(
                  markedReviewed
                    ? "Observación registrada y evaluación marcada como revisada."
                    : "Observación clínica registrada.",
                );
              }}
            />
          ) : null}
        </div>

        <div className="space-y-5">
          {radiograph && (evaluation.status === "imagen_cargada" || evaluation.status === "lista_para_analisis" || !qualityCheck?.passed) ? (
            <section className={`${cardClass} p-5`}>
              <div className="flex items-center justify-between gap-2">
                <h2 className="text-base font-semibold text-slate-900">Verificación de calidad</h2>
                {qualityCheck ? (
                  <span
                    className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                      qualityCheck.passed
                        ? "bg-emerald-50 text-emerald-800"
                        : "bg-rose-50 text-rose-800"
                    }`}
                  >
                    {qualityCheck.passed ? (
                      <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
                    ) : (
                      <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
                    )}
                    {qualityCheck.passed ? "Aprobada" : "Rechazada"}
                  </span>
                ) : null}
              </div>

              {qualityCheck ? (
                <div className="mt-3 space-y-2.5 text-xs">
                  <div className="grid grid-cols-2 gap-2">
                    <div className="rounded-lg bg-slate-50 px-3 py-2">
                      <p className="text-slate-500">Contraste (p99-p01)</p>
                      <p className="mt-0.5 font-mono font-semibold text-slate-900">
                        {qualityCheck.metrics.dynamic_range_p99_p01.toFixed(1)}
                      </p>
                    </div>
                    <div className="rounded-lg bg-slate-50 px-3 py-2">
                      <p className="text-slate-500">Desv. estándar</p>
                      <p className="mt-0.5 font-mono font-semibold text-slate-900">
                        {qualityCheck.metrics.intensity_std.toFixed(1)}
                      </p>
                    </div>
                  </div>
                  {qualityCheck.failures.length > 0 ? (
                    <Notice variant="error">
                      <ul className="list-disc space-y-1 pl-4">
                        {qualityCheck.failures.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </Notice>
                  ) : null}
                  {qualityCheck.warnings.length > 0 ? (
                    <Notice variant="warning">
                      <ul className="list-disc space-y-1 pl-4">
                        {qualityCheck.warnings.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </Notice>
                  ) : null}
                </div>
              ) : (
                <p className="mt-2 text-sm text-slate-600">
                  Verifique resolución mínima (224 px) y legibilidad para habilitar el análisis.
                </p>
              )}

              {canUpload &&
              (evaluation.status === "imagen_cargada" ||
                evaluation.status === "lista_para_analisis") ? (
                <div className="mt-4 flex flex-wrap gap-2">
                  <button
                    type="button"
                    className={buttonPrimary}
                    onClick={() => void handleValidateQuality()}
                    disabled={validatingQuality || analyzing}
                  >
                    {validatingQuality ? (
                      <Spinner className="h-4 w-4" />
                    ) : (
                      <Sparkles className="h-4 w-4" />
                    )}
                    {validatingQuality
                      ? "Verificando…"
                      : qualityCheck?.passed
                        ? "Revalidar"
                        : "Verificar calidad"}
                  </button>
                  {evaluation.status === "imagen_cargada" ? (
                    <button
                      type="button"
                      className={buttonSecondary}
                      onClick={() => setReplaceMode(true)}
                      disabled={validatingQuality || analyzing}
                    >
                      <RefreshCw className="h-4 w-4" aria-hidden="true" />
                      Reemplazar imagen
                    </button>
                  ) : null}
                </div>
              ) : null}
            </section>
          ) : null}

          {evaluation.status === "lista_para_analisis" && canAnalyze ? (
            <section className={`${cardClass} border-teal-200 bg-teal-50/50 p-5`}>
              <div className="flex items-start gap-3">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-teal-700 text-white">
                  <Cpu className="h-5 w-5" aria-hidden="true" />
                </span>
                <div className="min-w-0 flex-1">
                  <h2 className="text-base font-semibold text-slate-900">Análisis asistido</h2>
                  <p className="mt-1 text-sm leading-5 text-slate-600">
                    Ejecuta clasificación global, detección de zonas sospechosas y mapas visuales.
                  </p>
                  <button
                    type="button"
                    className={`${buttonPrimary} mt-4 w-full`}
                    onClick={() => void handleAnalyze()}
                    disabled={analyzing}
                  >
                    {analyzing ? <Spinner className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    {analyzing ? "Procesando imagen…" : "Ejecutar análisis"}
                  </button>
                </div>
              </div>
            </section>
          ) : null}

          {evaluation.has_prediction && !canReadPredictions ? (
            <Notice variant="info" title="Análisis completado">
              La imagen fue procesada correctamente. Los resultados probabilísticos y mapas visuales
              quedan disponibles para los perfiles médicos autorizados.
            </Notice>
          ) : null}

          {prediction && canReadPredictions ? (
            <PredictionPanel
              prediction={prediction}
              selectedBoxIndex={selectedBoxIndex}
              onSelectBox={setSelectedBoxIndex}
            />
          ) : null}

          <details className={`${cardClass} group overflow-hidden`} open={!prediction}>
            <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-5 py-4 text-sm font-semibold text-slate-900 hover:bg-slate-50">
              <span>Datos del estudio</span>
              <ChevronDown
                className="h-4 w-4 text-slate-400 transition group-open:rotate-180"
                aria-hidden="true"
              />
            </summary>
            <div className="border-t border-slate-100 px-5 py-4">
              <dl className="space-y-2.5 text-sm">
                <div className="flex items-start justify-between gap-3">
                  <dt className="text-slate-500">Registrado</dt>
                  <dd className="text-right font-medium text-slate-900">
                    {formatDate(evaluation.created_at)}
                  </dd>
                </div>
                <div className="flex items-start justify-between gap-3">
                  <dt className="text-slate-500">Actualizado</dt>
                  <dd className="text-right font-medium text-slate-900">
                    {formatDate(evaluation.updated_at)}
                  </dd>
                </div>
                <div className="flex items-start justify-between gap-3">
                  <dt className="text-slate-500">Responsable de alta</dt>
                  <dd className="text-right font-medium text-slate-900">
                    {evaluation.created_by_display_name ?? "Sistema"}
                  </dd>
                </div>
                {radiograph ? (
                  <>
                    <div className="flex items-start justify-between gap-3 border-t border-slate-100 pt-2.5">
                      <dt className="text-slate-500">Archivo</dt>
                      <dd className="text-right font-medium text-slate-900">
                        {radiograph.width} × {radiograph.height} px ·{" "}
                        {formatBytes(radiograph.size_bytes)}
                      </dd>
                    </div>
                    <div className="flex items-start justify-between gap-3">
                      <dt className="text-slate-500">SHA-256</dt>
                      <dd className="font-mono text-xs text-slate-600">
                        {shortHash(radiograph.sha256)}
                      </dd>
                    </div>
                  </>
                ) : null}
              </dl>
            </div>
          </details>
        </div>
      </div>
    </div>
  );
}
