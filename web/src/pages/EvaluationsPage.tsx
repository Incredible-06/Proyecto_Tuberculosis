import {
  ChevronRight,
  ClipboardList,
  FileImage,
  Filter,
  Plus,
  Search,
  X,
} from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import { ApiError, api } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import {
  buttonPrimary,
  buttonSecondary,
  cardClass,
  inputClass,
  labelClass,
} from "../components/styles";
import { EmptyState, LoadingPanel, Notice, PageHeader, Spinner, StatusBadge } from "../components/ui";
import { formatDate } from "../lib/format";
import {
  CONCORDANCE_LABELS,
  STATUS_LABELS,
  type Evaluation,
  type EvaluationStatus,
} from "../types";

const EMAIL_PATTERN = /\b[\w.+-]+@[\w-]+\.[\w.-]+\b/;
const LONG_DIGITS_PATTERN = /\b\d{8,}\b/;
const STATUS_OPTIONS: Array<{ value: EvaluationStatus | "todos"; label: string }> = [
  { value: "todos", label: "Todos los estados" },
  ...(Object.entries(STATUS_LABELS) as Array<[EvaluationStatus, string]>).map(
    ([value, label]) => ({ value, label }),
  ),
];

export function EvaluationsPage() {
  const { can } = useAuth();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [evaluations, setEvaluations] = useState<Evaluation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<EvaluationStatus | "todos">("todos");
  const [priorityOnly, setPriorityOnly] = useState(false);
  const [formOpen, setFormOpen] = useState(searchParams.get("nuevo") === "1");
  const [notes, setNotes] = useState("");
  const [creating, setCreating] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  const canCreate = can("evaluations.create");
  const canReadPredictions = can("predictions.read");

  useEffect(() => {
    let active = true;
    const timer = window.setTimeout(() => {
      const load = async () => {
        setLoading(true);
        setError(null);
        try {
          const loaded = await api.listEvaluations({
            status: statusFilter === "todos" ? undefined : statusFilter,
            search: search.trim() || undefined,
            review_required: canReadPredictions && priorityOnly ? true : undefined,
          });
          if (active) {
            setEvaluations(loaded);
          }
        } catch (loadError) {
          if (active) {
            setError(
              loadError instanceof Error
                ? loadError.message
                : "No se pudieron cargar las evaluaciones.",
            );
          }
        } finally {
          if (active) {
            setLoading(false);
          }
        }
      };
      void load();
    }, 180);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [search, statusFilter, priorityOnly, canReadPredictions]);

  useEffect(() => {
    if (searchParams.get("nuevo") === "1") {
      setFormOpen(true);
    }
  }, [searchParams]);

  const closeForm = () => {
    setFormOpen(false);
    setFormError(null);
    if (searchParams.has("nuevo")) {
      setSearchParams(
        (previous) => {
          const next = new URLSearchParams(previous);
          next.delete("nuevo");
          return next;
        },
        { replace: true },
      );
    }
  };

  const handleCreate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setFormError(null);
    setNotice(null);

    const normalizedNotes = notes.trim();
    if (normalizedNotes.length > 200) {
      setFormError("La nota interna admite como máximo 200 caracteres.");
      return;
    }
    if (EMAIL_PATTERN.test(normalizedNotes) || normalizedNotes.includes("@")) {
      setFormError("La nota no puede contener correos ni datos personales.");
      return;
    }
    if (LONG_DIGITS_PATTERN.test(normalizedNotes)) {
      setFormError("La nota no puede contener documentos ni datos personales.");
      return;
    }

    setCreating(true);
    try {
      const created = await api.createEvaluation(normalizedNotes || null);
      setEvaluations((current) => [created, ...current]);
      setNotes("");
      closeForm();
      navigate(`/evaluaciones/${created.id}`);
    } catch (createError) {
      setFormError(
        createError instanceof ApiError
          ? createError.message
          : "No se pudo registrar la evaluación.",
      );
    } finally {
      setCreating(false);
    }
  };

  const filtered = evaluations;
  const hasFilters = Boolean(search.trim()) || statusFilter !== "todos" || priorityOnly;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Evaluaciones radiográficas"
        title="Estudios de tórax"
        description="Cada evaluación recibe un código interno anonimizado. No registre nombres ni documentos de identidad."
        actions={
          canCreate ? (
            <button type="button" className={buttonPrimary} onClick={() => setFormOpen(true)}>
              <Plus className="h-4 w-4" aria-hidden="true" />
              Nueva evaluación
            </button>
          ) : null
        }
      />

      {error ? <Notice variant="error" onClose={() => setError(null)}>{error}</Notice> : null}
      {notice ? (
        <Notice variant="success" onClose={() => setNotice(null)}>
          {notice}
        </Notice>
      ) : null}

      {formOpen ? (
        <section className={`${cardClass} overflow-hidden`}>
          <div className="flex items-start justify-between gap-4 border-b border-slate-100 px-5 py-4">
            <div>
              <h2 className="text-base font-semibold text-slate-900">Registrar evaluación</h2>
              <p className="mt-1 text-sm text-slate-600">
                Se asignará automáticamente un código interno con formato TB-AAAAMMDD-XXXXXX.
              </p>
            </div>
            <button
              type="button"
              onClick={closeForm}
              className="rounded-lg p-2 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
              aria-label="Cerrar formulario"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <form className="space-y-5 px-5 py-5" onSubmit={handleCreate} noValidate>
            {formError ? <Notice variant="error">{formError}</Notice> : null}

            <div>
              <label className={labelClass} htmlFor="evaluation-notes">
                Nota interna opcional
              </label>
              <textarea
                id="evaluation-notes"
                name="notes"
                className={`${inputClass} min-h-24 resize-y`}
                value={notes}
                onChange={(event) => setNotes(event.target.value)}
                aria-describedby="evaluation-notes-hint"
                maxLength={200}
                placeholder="Ej.: control de calidad del turno, proyección PA"
                disabled={creating}
              />
              <div
                id="evaluation-notes-hint"
                className="mt-1.5 flex items-center justify-between text-xs text-slate-500"
              >
                <span>No incluya correos, teléfonos ni números de documento.</span>
                <span>{notes.length}/200</span>
              </div>
            </div>

            <div className="flex flex-col-reverse gap-3 border-t border-slate-100 pt-4 sm:flex-row sm:justify-end">
              <button type="button" className={buttonSecondary} onClick={closeForm} disabled={creating}>
                Cancelar
              </button>
              <button type="submit" className={buttonPrimary} disabled={creating}>
                {creating ? <Spinner className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
                {creating ? "Registrando…" : "Registrar evaluación"}
              </button>
            </div>
          </form>
        </section>
      ) : null}

      <search className={`${cardClass} block p-4`}>
        <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_16rem]">
          <div className="relative">
            <label className="sr-only" htmlFor="evaluation-search">
              Buscar por código interno
            </label>
            <Search
              className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
              aria-hidden="true"
            />
            <input
              id="evaluation-search"
              type="search"
              autoCorrect="off"
              autoCapitalize="none"
              spellCheck={false}
              className={`${inputClass} pl-10`}
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Buscar por código (ej. TB-2026…) "
              maxLength={64}
            />
          </div>
          <div className="relative">
            <label className="sr-only" htmlFor="evaluation-status">
              Filtrar por estado
            </label>
            <Filter
              className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
              aria-hidden="true"
            />
            <select
              id="evaluation-status"
              className={`${inputClass} pl-10`}
              value={statusFilter}
              onChange={(event) =>
                setStatusFilter(event.target.value as EvaluationStatus | "todos")
              }
            >
              {STATUS_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 pt-3">
          <div className="flex flex-wrap items-center gap-3">
            <p className="text-xs text-slate-500">
              {filtered.length} estudio{filtered.length === 1 ? "" : "s"} visible
              {filtered.length === 1 ? "" : "s"}.
            </p>
            {hasFilters ? (
              <button
                type="button"
                onClick={() => {
                  setSearch("");
                  setStatusFilter("todos");
                  setPriorityOnly(false);
                }}
                className="text-xs font-semibold text-teal-700 hover:text-teal-800"
              >
                Limpiar filtros
              </button>
            ) : null}
          </div>
          {canReadPredictions ? (
            <label className="inline-flex cursor-pointer items-center gap-2 text-xs font-semibold text-slate-700">
              <input
                type="checkbox"
                checked={priorityOnly}
                onChange={(event) => setPriorityOnly(event.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-teal-700 accent-teal-700 focus:ring-teal-600"
              />
              Solo revisión prioritaria
            </label>
          ) : null}
        </div>
      </search>

      {loading ? <LoadingPanel label="Cargando evaluaciones…" /> : null}

      {!loading && !error && evaluations.length === 0 && !hasFilters ? (
        <EmptyState
          icon={ClipboardList}
          title="Todavía no hay evaluaciones"
          description={
            canCreate
              ? "Registre la primera evaluación para habilitar la carga de una radiografía."
              : "Cuando el tecnólogo registre una evaluación, aparecerá en este listado."
          }
          action={
            canCreate ? (
              <button type="button" className={buttonPrimary} onClick={() => setFormOpen(true)}>
                <Plus className="h-4 w-4" aria-hidden="true" />
                Nueva evaluación
              </button>
            ) : undefined
          }
        />
      ) : null}

      {!loading && evaluations.length === 0 && hasFilters ? (
        <EmptyState
          icon={Search}
          title="Sin coincidencias"
          description="Pruebe con otro código interno o cambie el filtro de estado."
        />
      ) : null}

      {!loading && filtered.length > 0 ? (
        <section className={`${cardClass} overflow-hidden`}>
          <ul className="divide-y divide-slate-100">
            {filtered.map((evaluation) => (
              <li key={evaluation.id}>
                <Link
                  to={`/evaluaciones/${evaluation.id}`}
                  className="grid gap-4 px-5 py-4 transition hover:bg-slate-50 md:grid-cols-[minmax(0,1.4fr)_auto_auto_auto] md:items-center"
                >
                  <div className="min-w-0">
                    <p className="font-mono text-sm font-semibold tracking-tight text-slate-900">
                      {evaluation.internal_code}
                    </p>
                    <p className="mt-1 truncate text-xs text-slate-500">
                      {evaluation.created_by_display_name ?? "Cuenta del sistema"} ·{" "}
                      {formatDate(evaluation.created_at)}
                    </p>
                  </div>

                  <StatusBadge status={evaluation.status} />

                  {canReadPredictions && evaluation.concordance_state ? (
                    <span
                      className={`inline-flex w-fit items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold ${
                        evaluation.review_required
                          ? "border-rose-200 bg-rose-50 text-rose-800"
                          : "border-emerald-200 bg-emerald-50 text-emerald-700"
                      }`}
                    >
                      {CONCORDANCE_LABELS[evaluation.concordance_state]}
                    </span>
                  ) : (
                    <span
                      className={`inline-flex w-fit items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold ${
                        evaluation.radiograph
                          ? "border-sky-200 bg-sky-50 text-sky-700"
                          : "border-slate-200 bg-slate-100 text-slate-500"
                      }`}
                    >
                      <FileImage className="h-3.5 w-3.5" aria-hidden="true" />
                      {evaluation.radiograph ? "Imagen cargada" : "Sin imagen"}
                    </span>
                  )}

                  <span className="flex items-center justify-between gap-3 md:justify-end">
                    <span className="text-xs font-medium text-slate-500">
                      {evaluation.radiograph
                        ? `${evaluation.radiograph.width} × ${evaluation.radiograph.height} px`
                        : "Pendiente"}
                    </span>
                    <ChevronRight className="h-5 w-5 text-slate-400" aria-hidden="true" />
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}
