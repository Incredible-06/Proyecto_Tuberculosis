import {
  AlertTriangle,
  CheckCircle2,
  Info,
  Loader2,
  X,
  type LucideIcon,
} from "lucide-react";
import type { ReactNode } from "react";

import { STATUS_LABELS, type EvaluationStatus } from "../types";
import { cardClass } from "./styles";

type NoticeVariant = "error" | "success" | "info" | "warning";

const noticeStyles: Record<NoticeVariant, string> = {
  error: "border-rose-200 bg-rose-50 text-rose-900",
  success: "border-emerald-200 bg-emerald-50 text-emerald-900",
  info: "border-sky-200 bg-sky-50 text-sky-900",
  warning: "border-amber-200 bg-amber-50 text-amber-900",
};

const noticeIcons: Record<NoticeVariant, LucideIcon> = {
  error: AlertTriangle,
  success: CheckCircle2,
  info: Info,
  warning: AlertTriangle,
};

export function Notice({
  variant = "error",
  title,
  children,
  onClose,
}: {
  variant?: NoticeVariant;
  title?: string;
  children: ReactNode;
  onClose?: () => void;
}) {
  const Icon = noticeIcons[variant];
  return (
    <div
      role={variant === "error" || variant === "warning" ? "alert" : "status"}
      className={`flex items-start gap-3 rounded-xl border px-4 py-3 text-sm ${noticeStyles[variant]}`}
    >
      <Icon className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
      <div className="min-w-0 flex-1">
        {title ? <p className="font-semibold">{title}</p> : null}
        <div className={title ? "mt-0.5" : ""}>{children}</div>
      </div>
      {onClose ? (
        <button
          type="button"
          onClick={onClose}
          className="rounded-md p-1 transition hover:bg-black/5"
          aria-label="Cerrar aviso"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      ) : null}
    </div>
  );
}

export function Spinner({ className = "h-5 w-5" }: { className?: string }) {
  return <Loader2 className={`animate-spin ${className}`} aria-hidden="true" />;
}

export function LoadingPanel({ label = "Cargando información…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-3 rounded-2xl border border-slate-200 bg-white px-6 py-12 text-sm text-slate-500 shadow-sm">
      <Spinner className="h-5 w-5 text-teal-600" />
      {label}
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div>
        {eyebrow ? (
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-teal-700">
            {eyebrow}
          </p>
        ) : null}
        <h1 className="mt-1 text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl">
          {title}
        </h1>
        {description ? (
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex flex-wrap gap-2">{actions}</div> : null}
    </div>
  );
}

const statusStyles: Record<EvaluationStatus, string> = {
  pendiente_imagen: "border-amber-200 bg-amber-50 text-amber-800",
  imagen_cargada: "border-sky-200 bg-sky-50 text-sky-800",
  lista_para_analisis: "border-indigo-200 bg-indigo-50 text-indigo-800",
  analizada: "border-emerald-200 bg-emerald-50 text-emerald-800",
  revisada: "border-slate-200 bg-slate-100 text-slate-700",
};

export function StatusBadge({ status }: { status: EvaluationStatus }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${statusStyles[status]}`}
    >
      {STATUS_LABELS[status]}
    </span>
  );
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className={`${cardClass} flex flex-col items-center px-6 py-14 text-center`}>
      <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-100 text-slate-500">
        <Icon className="h-6 w-6" aria-hidden="true" />
      </div>
      <h2 className="mt-4 text-base font-semibold text-slate-900">{title}</h2>
      <p className="mt-1 max-w-md text-sm leading-6 text-slate-600">{description}</p>
      {action ? <div className="mt-5">{action}</div> : null}
    </div>
  );
}

export function SectionCard({
  title,
  description,
  children,
  footer,
}: {
  title: string;
  description?: string;
  children: ReactNode;
  footer?: ReactNode;
}) {
  return (
    <section className={`${cardClass} overflow-hidden`}>
      <div className="border-b border-slate-100 px-5 py-4">
        <h2 className="text-base font-semibold text-slate-900">{title}</h2>
        {description ? <p className="mt-1 text-sm text-slate-600">{description}</p> : null}
      </div>
      <div className="px-5 py-5">{children}</div>
      {footer ? <div className="border-t border-slate-100 bg-slate-50 px-5 py-3">{footer}</div> : null}
    </section>
  );
}
