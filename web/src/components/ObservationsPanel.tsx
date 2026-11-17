import { CheckCircle2, FileText, Send } from "lucide-react";
import { useState, type FormEvent } from "react";

import { ApiError, api } from "../api/client";
import { formatDate } from "../lib/format";
import {
  AGREEMENT_LABELS,
  type EvaluationStatus,
  type Observation,
  type ObservationAgreement,
} from "../types";
import { buttonPrimary, cardClass, inputClass, labelClass } from "./styles";
import { Notice, Spinner } from "./ui";

const EMAIL_PATTERN = /\b[\w.+-]+@[\w-]+\.[\w.-]+\b/;
const LONG_DIGITS_PATTERN = /\b\d{8,}\b/;

export function ObservationsPanel({
  evaluationId,
  status,
  canWrite,
  observations,
  onCreated,
}: {
  evaluationId: string;
  status: EvaluationStatus;
  canWrite: boolean;
  observations: Observation[];
  onCreated: (created: Observation, markedReviewed: boolean) => void;
}) {
  const [impression, setImpression] = useState("");
  const [agreement, setAgreement] = useState<ObservationAgreement>("concordante");
  const [markReviewed, setMarkReviewed] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canRecordNow = canWrite && (status === "analizada" || status === "revisada");

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    const normalized = impression.trim();
    if (normalized.length < 3) {
      setError("Ingrese una observación clínica de al menos 3 caracteres.");
      return;
    }
    if (EMAIL_PATTERN.test(normalized) || normalized.includes("@")) {
      setError("La observación no puede contener correos ni datos personales.");
      return;
    }
    if (LONG_DIGITS_PATTERN.test(normalized)) {
      setError("La observación no puede contener documentos ni datos personales.");
      return;
    }

    setSubmitting(true);
    try {
      const created = await api.createObservation(evaluationId, {
        clinical_impression: normalized,
        agreement_status: agreement,
        mark_reviewed: markReviewed && status === "analizada",
      });
      setImpression("");
      onCreated(created, markReviewed && status === "analizada");
    } catch (submitError) {
      setError(
        submitError instanceof ApiError
          ? submitError.message
          : "No se pudo guardar la observación clínica.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className={`${cardClass} overflow-hidden`}>
      <div className="border-b border-slate-100 px-5 py-4">
        <div className="flex items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Evaluación médica y observaciones
            </h2>
            <p className="mt-1 text-sm text-slate-600">
              Registro de impresión radiológica y cierre de revisión clínica.
            </p>
          </div>
          {status === "revisada" ? (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-800">
              <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
              Revisada
            </span>
          ) : null}
        </div>
      </div>

      <div className="space-y-5 p-5">
        {canRecordNow ? (
          <form className="space-y-4 rounded-xl border border-slate-200 bg-slate-50 p-4" onSubmit={handleSubmit} noValidate>
            {error ? (
              <div id="observation-error">
                <Notice variant="error">{error}</Notice>
              </div>
            ) : null}

            <div>
              <label className={labelClass} htmlFor="observation-agreement">
                Contraste con el análisis
              </label>
              <select
                id="observation-agreement"
                name="agreement_status"
                className={inputClass}
                value={agreement}
                onChange={(event) => setAgreement(event.target.value as ObservationAgreement)}
                disabled={submitting}
              >
                {(Object.entries(AGREEMENT_LABELS) as Array<[ObservationAgreement, string]>).map(
                  ([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ),
                )}
              </select>
            </div>

            <div>
              <label className={labelClass} htmlFor="observation-text">
                Impresión y conducta clínica
              </label>
              <textarea
                id="observation-text"
                name="clinical_impression"
                className={`${inputClass} min-h-24 resize-y`}
                value={impression}
                onChange={(event) => {
                  setImpression(event.target.value);
                  if (error) setError(null);
                }}
                aria-invalid={Boolean(error)}
                aria-describedby={error ? "observation-error observation-hint" : "observation-hint"}
                maxLength={1000}
                placeholder="Ej.: Infiltrado apical derecho compatible con sospecha; derivar a baciloscopía y GeneXpert."
                disabled={submitting}
              />
              <div
                id="observation-hint"
                className="mt-1 flex items-center justify-between text-xs text-slate-500"
              >
                <span>No incluya datos personales ni identificadores del paciente.</span>
                <span>{impression.length}/1000</span>
              </div>
            </div>

            {status === "analizada" ? (
              <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-700">
                <input
                  type="checkbox"
                  checked={markReviewed}
                  onChange={(event) => setMarkReviewed(event.target.checked)}
                  disabled={submitting}
                  className="h-4 w-4 rounded border-slate-300 text-teal-700 accent-teal-700 focus:ring-teal-600"
                />
                Marcar evaluación como <strong>Revisada</strong> al guardar
              </label>
            ) : null}

            <div className="flex justify-end">
              <button type="submit" className={buttonPrimary} disabled={submitting}>
                {submitting ? <Spinner className="h-4 w-4" /> : <Send className="h-4 w-4" />}
                {submitting ? "Guardando…" : "Registrar observación"}
              </button>
            </div>
          </form>
        ) : canWrite ? (
          <Notice variant="info">
            Las observaciones clínicas se habilitan después de completar el análisis de la imagen.
          </Notice>
        ) : null}

        {observations.length === 0 ? (
          <div className="flex items-center gap-2.5 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-500">
            <FileText className="h-4 w-4 shrink-0" aria-hidden="true" />
            Aún no hay observaciones clínicas registradas para este estudio.
          </div>
        ) : (
          <ul className="space-y-3">
            {observations.map((obs) => (
              <li key={obs.id} className="rounded-xl border border-slate-200 bg-white p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-sm font-semibold text-slate-900">
                    {obs.author_display_name ?? "Médico autorizado"}
                  </span>
                  <span className="rounded-full border border-teal-200 bg-teal-50 px-2.5 py-0.5 text-xs font-semibold text-teal-800">
                    {AGREEMENT_LABELS[obs.agreement_status]}
                  </span>
                </div>
                <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-700">
                  {obs.clinical_impression}
                </p>
                <p className="mt-2 text-xs text-slate-400">{formatDate(obs.created_at)}</p>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
