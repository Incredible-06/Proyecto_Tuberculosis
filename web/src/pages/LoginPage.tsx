import { Activity, Eye, EyeOff, Lock, ShieldCheck } from "lucide-react";
import { useRef, useState, type FormEvent } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";

import { ApiError } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { buttonPrimary, inputClass, labelClass } from "../components/styles";
import { Notice, Spinner } from "../components/ui";

export function LoginPage() {
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const usernameRef = useRef<HTMLInputElement>(null);
  const passwordRef = useRef<HTMLInputElement>(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (user) {
    return <Navigate to="/" replace />;
  }

  const from = (location.state as { from?: string } | null)?.from ?? "/";

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    if (!username.trim()) {
      setError("Ingrese su identificador de acceso.");
      usernameRef.current?.focus();
      return;
    }
    if (!password) {
      setError("Ingrese su contraseña.");
      passwordRef.current?.focus();
      return;
    }
    setSubmitting(true);
    try {
      await login(username.trim(), password);
      navigate(from, { replace: true });
    } catch (loginError) {
      if (loginError instanceof ApiError) {
        setError(loginError.message);
      } else {
        setError("No se pudo conectar con el servidor. Intente nuevamente.");
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[1.05fr_0.95fr]">
      <section className="relative hidden overflow-hidden bg-slate-950 px-12 py-14 text-white lg:flex lg:flex-col lg:justify-between">
        <div className="absolute inset-0 bg-clinical-grid opacity-60" aria-hidden="true" />
        <div className="absolute -left-24 top-24 h-72 w-72 rounded-full bg-teal-500/20 blur-3xl" />
        <div className="absolute -bottom-20 right-0 h-80 w-80 rounded-full bg-cyan-400/10 blur-3xl" />

        <div className="relative">
          <div className="flex items-center gap-3">
            <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-teal-500/20 ring-1 ring-teal-400/40">
              <ShieldCheck className="h-6 w-6 text-teal-200" aria-hidden="true" />
            </span>
            <div>
              <p className="text-sm font-semibold tracking-tight">Sistema TB</p>
              <p className="text-xs text-slate-400">Apoyo al cribado radiográfico</p>
            </div>
          </div>

          <div className="mt-20 max-w-lg">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-teal-300">
              Unidad de radiología y programa TB
            </p>
            <h1 className="mt-4 text-balance text-4xl font-semibold leading-tight tracking-tight">
              Priorización de radiografías de tórax con trazabilidad clínica.
            </h1>
            <p className="mt-5 text-base leading-7 text-slate-300">
              Flujo anonimizado para registro de estudios, control de calidad de imagen, análisis
              asistido y revisión médica por roles.
            </p>
          </div>

          <div className="mt-12 grid max-w-lg gap-4 sm:grid-cols-2">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
              <Lock className="h-5 w-5 text-teal-300" aria-hidden="true" />
              <p className="mt-3 text-sm font-semibold">Acceso por perfil</p>
              <p className="mt-1 text-xs leading-5 text-slate-400">
                Sesión protegida, registro sin datos del paciente y permisos separados por rol.
              </p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
              <Activity className="h-5 w-5 text-teal-300" aria-hidden="true" />
              <p className="mt-3 text-sm font-semibold">Soporte a la lectura</p>
              <p className="mt-1 text-xs leading-5 text-slate-400">
                Clasificación por probabilidad, localización focal y mapas visuales de respaldo.
              </p>
            </div>
          </div>
        </div>

        <p className="relative max-w-lg text-xs leading-5 text-slate-500">
          Herramienta de apoyo a la decisión clínica. No sustituye el criterio médico ni las
          pruebas microbiológicas confirmatorias.
        </p>
      </section>

      <section className="flex min-h-screen items-center justify-center bg-slate-50 px-5 py-10 sm:px-8">
        <div className="w-full max-w-md">
          <div className="mb-8 flex items-center gap-3 lg:hidden">
            <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-teal-700 text-white shadow-sm">
              <ShieldCheck className="h-6 w-6" aria-hidden="true" />
            </span>
            <div>
              <p className="text-sm font-semibold text-slate-900">Sistema TB</p>
              <p className="text-xs text-slate-500">Apoyo al cribado radiográfico</p>
            </div>
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-7 shadow-xl shadow-slate-200/60 sm:p-9">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-teal-700">
              Acceso autorizado
            </p>
            <h2 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">
              Iniciar sesión
            </h2>
            <p className="mt-2 text-sm leading-6 text-slate-600">
              Ingrese con las credenciales asignadas por su administrador.
            </p>

            <form className="mt-7 space-y-5" onSubmit={handleSubmit} noValidate>
              {error ? <Notice variant="error">{error}</Notice> : null}

              <div>
                <label className={labelClass} htmlFor="username">
                  Identificador de acceso
                </label>
                <input
                  ref={usernameRef}
                  id="username"
                  name="username"
                  type="text"
                  autoComplete="username"
                  autoCorrect="off"
                  autoCapitalize="none"
                  spellCheck={false}
                  enterKeyHint="next"
                  maxLength={64}
                  value={username}
                  onChange={(event) => {
                    setUsername(event.target.value);
                    if (error) setError(null);
                  }}
                  className={inputClass}
                  placeholder="nombre.apellido"
                  disabled={submitting}
                  required
                />
              </div>

              <div>
                <label className={labelClass} htmlFor="password">
                  Contraseña
                </label>
                <div className="relative">
                  <input
                    ref={passwordRef}
                    id="password"
                    name="password"
                    type={showPassword ? "text" : "password"}
                    autoComplete="current-password"
                    enterKeyHint="done"
                    maxLength={128}
                    value={password}
                    onChange={(event) => {
                      setPassword(event.target.value);
                      if (error) setError(null);
                    }}
                    className={`${inputClass} pr-12`}
                    placeholder="Ingrese su contraseña"
                    disabled={submitting}
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((current) => !current)}
                    className="absolute inset-y-0 right-0 flex w-11 items-center justify-center rounded-r-xl text-slate-400 transition hover:text-slate-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-teal-600"
                    aria-label={showPassword ? "Ocultar contraseña" : "Mostrar contraseña"}
                    aria-pressed={showPassword}
                  >
                    {showPassword ? (
                      <EyeOff className="h-4.5 w-4.5" aria-hidden="true" />
                    ) : (
                      <Eye className="h-4.5 w-4.5" aria-hidden="true" />
                    )}
                  </button>
                </div>
              </div>

              <button type="submit" className={`${buttonPrimary} w-full`} disabled={submitting}>
                {submitting ? <Spinner className="h-4 w-4" /> : null}
                {submitting ? "Verificando…" : "Ingresar al sistema"}
              </button>
            </form>
          </div>

          <p className="mt-6 text-center text-xs leading-5 text-slate-500">
            Sistema de apoyo clínico · No sustituye el diagnóstico médico
          </p>
        </div>
      </section>
    </div>
  );
}
