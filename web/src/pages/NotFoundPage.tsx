import { ArrowLeft, FileQuestion } from "lucide-react";
import { Link } from "react-router-dom";

import { buttonSecondary } from "../components/styles";

export function NotFoundPage() {
  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <div className="max-w-md text-center">
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-100 text-slate-500">
          <FileQuestion className="h-7 w-7" aria-hidden="true" />
        </div>
        <h1 className="mt-5 text-2xl font-semibold tracking-tight text-slate-900">
          Página no encontrada
        </h1>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          La ruta solicitada no existe o fue movida. Vuelva al panel principal para continuar.
        </p>
        <Link to="/" className={`${buttonSecondary} mt-6`}>
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          Volver al inicio
        </Link>
      </div>
    </div>
  );
}
