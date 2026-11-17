import {
  Activity,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  ClipboardList,
  Clock,
  FileImage,
  Plus,
  ShieldAlert,
  UserCheck,
  Users,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { buttonPrimary, buttonSecondary, cardClass } from "../components/styles";
import { EmptyState, Notice, PageHeader, StatusBadge } from "../components/ui";
import { formatDate, shortHash } from "../lib/format";
import {
  CONCORDANCE_LABELS,
  ROLE_LABELS,
  type Evaluation,
  type ModelVersion,
  type User,
} from "../types";

interface DashboardData {
  evaluations: Evaluation[];
  users: User[];
  modelVersions: ModelVersion[];
}

export function DashboardPage() {
  const { user, can } = useAuth();
  const [data, setData] = useState<DashboardData>({
    evaluations: [],
    users: [],
    modelVersions: [],
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const canModelInfo = can("model_versions.manage") || can("predictions.read");
        const [evaluations, users, modelVersions] = await Promise.all([
          can("evaluations.read") ? api.listEvaluations() : Promise.resolve([]),
          can("users.read") ? api.listUsers() : Promise.resolve([]),
          canModelInfo ? api.listModelVersions() : Promise.resolve([]),
        ]);
        if (active) {
          setData({ evaluations, users, modelVersions });
        }
      } catch (loadError) {
        if (active) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : "No se pudo cargar el resumen del sistema.",
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
  }, [can]);

  if (!user) {
    return null;
  }

  const canReadEvaluations = can("evaluations.read");
  const canCreateEvaluation = can("evaluations.create");
  const canReadPredictions = can("predictions.read");
  const canManageUsers = can("users.manage");
  const canReadUsers = can("users.read");

  const pendingImages = data.evaluations.filter(
    (evaluation) =>
      evaluation.status === "pendiente_imagen" || evaluation.status === "imagen_cargada",
  ).length;
  const readyForAnalysis = data.evaluations.filter(
    (evaluation) => evaluation.status === "lista_para_analisis",
  ).length;
  const analyzedOrReviewed = data.evaluations.filter(
    (evaluation) => evaluation.status === "analizada" || evaluation.status === "revisada",
  ).length;
  const priorityCases = data.evaluations.filter(
    (evaluation) => evaluation.review_required && evaluation.status === "analizada",
  ).length;

  const activeUsers = data.users.filter((account) => account.is_active).length;
  const activeModel = data.modelVersions.find((item) => item.is_active) ?? data.modelVersions[0];

  // Cola accionable priorizada según el rol activo
  const queueItems = canReadPredictions
    ? [...data.evaluations]
        .filter((item) => item.status === "analizada" || item.status === "lista_para_analisis")
        .sort((a, b) => Number(Boolean(b.review_required)) - Number(Boolean(a.review_required)))
        .slice(0, 6)
    : data.evaluations
        .filter((item) => item.status !== "revisada")
        .slice(0, 6);

  const firstName = user.display_name.split(/\s+/)[0] ?? user.display_name;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow={ROLE_LABELS[user.role]}
        title={`Buen día, ${firstName}`}
        description={
          canReadEvaluations
            ? "Resumen operativo de evaluaciones radiográficas y casos pendientes."
            : "Administración de cuentas autorizadas y verificación del motor de análisis."
        }
        actions={
          <>
            {canCreateEvaluation ? (
              <Link to="/evaluaciones?nuevo=1" className={buttonPrimary}>
                <Plus className="h-4 w-4" aria-hidden="true" />
                Nueva evaluación
              </Link>
            ) : null}
            {canReadEvaluations ? (
              <Link to="/evaluaciones" className={buttonSecondary}>
                Ver todas
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </Link>
            ) : null}
            {canManageUsers ? (
              <Link to="/usuarios?nuevo=1" className={buttonPrimary}>
                <Users className="h-4 w-4" aria-hidden="true" />
                Nuevo usuario
              </Link>
            ) : null}
          </>
        }
      />

      {error ? <Notice variant="error">{error}</Notice> : null}

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {canReadEvaluations ? (
          <>
            <SummaryCard
              label="Total evaluaciones"
              value={loading ? "—" : String(data.evaluations.length)}
              icon={ClipboardList}
              tone="teal"
            />
            <SummaryCard
              label="Por cargar o validar"
              value={loading ? "—" : String(pendingImages)}
              icon={Clock}
              tone="amber"
            />
            <SummaryCard
              label="Listas para análisis"
              value={loading ? "—" : String(readyForAnalysis)}
              icon={FileImage}
              tone="sky"
            />
            <SummaryCard
              label={canReadPredictions ? "Prioritarias sin cierre" : "Procesadas"}
              value={loading ? "—" : String(canReadPredictions ? priorityCases : analyzedOrReviewed)}
              icon={canReadPredictions ? ShieldAlert : CheckCircle2}
              tone={canReadPredictions && priorityCases > 0 ? "rose" : "emerald"}
            />
          </>
        ) : null}
        {canReadUsers ? (
          <>
            <SummaryCard
              label="Cuentas activas"
              value={loading ? "—" : `${activeUsers} de ${data.users.length}`}
              icon={UserCheck}
              tone="emerald"
            />
            <SummaryCard
              label="Motor de análisis"
              value={activeModel ? "Operativo" : "—"}
              icon={Activity}
              tone="teal"
            />
          </>
        ) : null}
      </section>

      <div className="grid gap-6 xl:grid-cols-[1.4fr_0.6fr]">
        {canReadEvaluations ? (
          <section className={`${cardClass} overflow-hidden`}>
            <div className="flex items-center justify-between gap-4 border-b border-slate-100 px-5 py-4">
              <div>
                <h2 className="text-base font-semibold text-slate-900">
                  {canReadPredictions ? "Casos pendientes de revisión" : "Cola de trabajo"}
                </h2>
                <p className="mt-0.5 text-sm text-slate-600">
                  {canReadPredictions
                    ? "Estudios analizados ordenados por prioridad de revisión."
                    : "Evaluaciones en curso que requieren imagen, validación o análisis."}
                </p>
              </div>
              <Link
                to="/evaluaciones"
                className="text-xs font-semibold text-teal-700 hover:text-teal-800"
              >
                Ir al listado
              </Link>
            </div>

            {queueItems.length === 0 ? (
              <div className="p-6">
                <EmptyState
                  icon={CheckCircle2}
                  title="Sin pendientes en cola"
                  description={
                    canCreateEvaluation
                      ? "Registre una nueva evaluación para iniciar el flujo con una radiografía."
                      : "No hay evaluaciones analizadas pendientes de revisión en este momento."
                  }
                />
              </div>
            ) : (
              <ul className="divide-y divide-slate-100">
                {queueItems.map((item) => (
                  <li key={item.id}>
                    <Link
                      to={`/evaluaciones/${item.id}`}
                      className="flex items-center justify-between gap-4 px-5 py-3.5 transition hover:bg-slate-50"
                    >
                      <div className="min-w-0">
                        <p className="font-mono text-sm font-semibold text-slate-900">
                          {item.internal_code}
                        </p>
                        <p className="mt-0.5 truncate text-xs text-slate-500">
                          {formatDate(item.updated_at)}
                        </p>
                      </div>
                      <div className="flex items-center gap-2.5">
                        {canReadPredictions && item.concordance_state ? (
                          <span
                            className={`hidden rounded-full border px-2.5 py-0.5 text-xs font-semibold sm:inline-flex ${
                              item.review_required
                                ? "border-rose-200 bg-rose-50 text-rose-800"
                                : "border-emerald-200 bg-emerald-50 text-emerald-700"
                            }`}
                          >
                            {CONCORDANCE_LABELS[item.concordance_state]}
                          </span>
                        ) : null}
                        <StatusBadge status={item.status} />
                        <ChevronRight className="h-4 w-4 text-slate-400" aria-hidden="true" />
                      </div>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </section>
        ) : (
          <section className={`${cardClass} p-6`}>
            <h2 className="text-base font-semibold text-slate-900">Gestión de cuentas</h2>
            <p className="mt-1 text-sm leading-6 text-slate-600">
              Administre los accesos de tecnólogos médicos, médicos del programa TB, radiólogos y
              coordinación. Los permisos clínicos se restringen automáticamente por rol.
            </p>
            <div className="mt-4 flex flex-wrap gap-3">
              <Link to="/usuarios" className={buttonPrimary}>
                <Users className="h-4 w-4" aria-hidden="true" />
                Administrar usuarios
              </Link>
            </div>
          </section>
        )}

        <div className="space-y-6">
          {activeModel ? (
            <section className={`${cardClass} p-5`}>
              <div className="flex items-center gap-3">
                <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-50 text-teal-700">
                  <Activity className="h-5 w-5" aria-hidden="true" />
                </span>
                <div>
                  <h2 className="text-sm font-semibold text-slate-900">Motor de apoyo diagnóstico</h2>
                  <p className="text-xs text-teal-700">Configuración verificada y activa</p>
                </div>
              </div>
              <dl className="mt-4 space-y-2 text-xs">
                <div className="flex justify-between gap-2">
                  <dt className="text-slate-500">Clasificador</dt>
                  <dd className="font-mono font-semibold text-slate-900">
                    {activeModel.classifier_display_name} (corte TB {activeModel.classifier_threshold})
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-slate-500">Detector focal</dt>
                  <dd className="font-mono font-semibold text-slate-900">
                    {activeModel.detector_arch} (conf. &gt;= {activeModel.detector_threshold})
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-slate-500">Integridad</dt>
                  <dd className="font-mono text-slate-700">
                    {shortHash(activeModel.contract_sha256)}
                  </dd>
                </div>
              </dl>
            </section>
          ) : null}

          <section className="rounded-2xl border border-amber-200 bg-amber-50 p-5 text-xs leading-5 text-amber-950">
            <p className="font-semibold">Uso clínico responsable</p>
            <p className="mt-1.5">
              Los resultados del modelo son orientativos para priorización y lectura visual; no
              constituyen diagnóstico definitivo. Registre únicamente códigos internos anonimizados.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}

function SummaryCard({
  label,
  value,
  icon: Icon,
  tone,
}: {
  label: string;
  value: string;
  icon: typeof ClipboardList;
  tone: "teal" | "amber" | "sky" | "emerald" | "rose";
}) {
  const tones = {
    teal: "bg-teal-50 text-teal-700",
    amber: "bg-amber-50 text-amber-700",
    sky: "bg-sky-50 text-sky-700",
    emerald: "bg-emerald-50 text-emerald-700",
    rose: "bg-rose-50 text-rose-700",
  };
  return (
    <div className={`${cardClass} p-5`}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-slate-500">{label}</p>
          <p className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">{value}</p>
        </div>
        <span className={`flex h-10 w-10 items-center justify-center rounded-xl ${tones[tone]}`}>
          <Icon className="h-5 w-5" aria-hidden="true" />
        </span>
      </div>
    </div>
  );
}
