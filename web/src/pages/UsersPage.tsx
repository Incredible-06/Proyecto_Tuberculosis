import {
  Eye,
  EyeOff,
  Plus,
  ShieldCheck,
  UserPlus,
  Users,
  X,
} from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { useSearchParams } from "react-router-dom";

import { ApiError, api } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import {
  buttonDanger,
  buttonPrimary,
  buttonSecondary,
  cardClass,
  inputClass,
  labelClass,
} from "../components/styles";
import { EmptyState, LoadingPanel, Notice, PageHeader, Spinner } from "../components/ui";
import { formatDate, initials } from "../lib/format";
import { ROLE_LABELS, type Role, type RoleInfo, type User } from "../types";

const USERNAME_PATTERN = /^[A-Za-z0-9._-]{3,64}$/;

export function UsersPage() {
  const { user: currentUser, refresh } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();
  const confirmDialogRef = useRef<HTMLDialogElement>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [roles, setRoles] = useState<RoleInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [formOpen, setFormOpen] = useState(searchParams.get("nuevo") === "1");
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [role, setRole] = useState<Role>("tecnologo_radiologia");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [creating, setCreating] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [updatingUserId, setUpdatingUserId] = useState<string | null>(null);
  const [pendingDeactivate, setPendingDeactivate] = useState<User | null>(null);

  useEffect(() => {
    let active = true;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const [loadedUsers, loadedRoles] = await Promise.all([
          api.listUsers(),
          api.listRoles(),
        ]);
        if (active) {
          setUsers(loadedUsers);
          setRoles(loadedRoles);
        }
      } catch (loadError) {
        if (active) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : "No se pudo cargar la administración de usuarios.",
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
  }, []);

  useEffect(() => {
    if (searchParams.get("nuevo") === "1") {
      setFormOpen(true);
    }
  }, [searchParams]);

  const roleLabel = (value: Role) =>
    roles.find((item) => item.value === value)?.label ?? ROLE_LABELS[value];

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

  const resetForm = () => {
    setUsername("");
    setDisplayName("");
    setRole("tecnologo_radiologia");
    setPassword("");
    setShowPassword(false);
    setFormError(null);
  };

  const handleCreate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setFormError(null);
    setNotice(null);

    if (!USERNAME_PATTERN.test(username.trim())) {
      setFormError(
        "El identificador debe tener entre 3 y 64 caracteres: letras, números, punto, guion o guion bajo.",
      );
      return;
    }
    if (!displayName.trim()) {
      setFormError("Ingrese el nombre visible de la cuenta.");
      return;
    }
    if (password.length < 12 || password.length > 128) {
      setFormError("La contraseña debe tener entre 12 y 128 caracteres.");
      return;
    }
    if (!password.trim()) {
      setFormError("La contraseña no puede contener solo espacios.");
      return;
    }
    if (password.toLowerCase() === username.trim().toLowerCase()) {
      setFormError("La contraseña no puede coincidir con el identificador de acceso.");
      return;
    }

    setCreating(true);
    try {
      const created = await api.createUser({
        username: username.trim(),
        display_name: displayName.trim(),
        role,
        password,
      });
      setUsers((current) => [...current, created]);
      setNotice(`Cuenta creada para ${created.display_name}.`);
      resetForm();
      closeForm();
    } catch (createError) {
      setFormError(
        createError instanceof ApiError
          ? createError.message
          : "No se pudo crear la cuenta. Intente nuevamente.",
      );
    } finally {
      setCreating(false);
    }
  };

  const replaceUser = (updated: User) => {
    setUsers((current) => current.map((item) => (item.id === updated.id ? updated : item)));
  };

  const handleRoleChange = async (target: User, nextRole: Role) => {
    if (target.role === nextRole) {
      return;
    }
    setNotice(null);
    setError(null);
    setUpdatingUserId(target.id);
    try {
      const updated = await api.updateUser(target.id, { role: nextRole });
      replaceUser(updated);
      setNotice(`Rol actualizado para ${updated.display_name}.`);
      if (currentUser?.id === updated.id) {
        await refresh();
      }
    } catch (updateError) {
      setError(
        updateError instanceof ApiError
          ? updateError.message
          : "No se pudo actualizar el rol.",
      );
    } finally {
      setUpdatingUserId(null);
    }
  };

  const applyToggleActive = async (target: User) => {
    setNotice(null);
    setError(null);
    setUpdatingUserId(target.id);
    try {
      const updated = await api.updateUser(target.id, { is_active: !target.is_active });
      replaceUser(updated);
      setNotice(
        updated.is_active
          ? `Cuenta reactivada para ${updated.display_name}.`
          : `Cuenta desactivada para ${updated.display_name}.`,
      );
    } catch (updateError) {
      setError(
        updateError instanceof ApiError
          ? updateError.message
          : "No se pudo cambiar el estado de la cuenta.",
      );
    } finally {
      setUpdatingUserId(null);
    }
  };

  const requestToggleActive = (target: User) => {
    if (target.is_active) {
      setPendingDeactivate(target);
      confirmDialogRef.current?.showModal();
      return;
    }
    void applyToggleActive(target);
  };

  const handleConfirmDialogClose = () => {
    const dialog = confirmDialogRef.current;
    const action = dialog?.returnValue ?? "";
    if (dialog) {
      dialog.returnValue = "";
    }
    const target = pendingDeactivate;
    setPendingDeactivate(null);
    if (action === "confirm" && target) {
      void applyToggleActive(target);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Control de acceso"
        title="Usuarios autorizados"
        description="Gestione las cuentas del personal clínico y técnico. Los permisos se aplican por rol."
        actions={
          <button type="button" className={buttonPrimary} onClick={() => setFormOpen(true)}>
            <Plus className="h-4 w-4" aria-hidden="true" />
            Nuevo usuario
          </button>
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

      {formOpen ? (
        <section className={`${cardClass} overflow-hidden`}>
          <div className="flex items-start justify-between gap-4 border-b border-slate-100 px-5 py-4">
            <div>
              <h2 className="text-base font-semibold text-slate-900">Registrar usuario</h2>
              <p className="mt-1 text-sm text-slate-600">
                La cuenta queda activa con el rol inicial seleccionado.
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

          <form className="grid gap-5 px-5 py-5 sm:grid-cols-2" onSubmit={handleCreate} noValidate>
            {formError ? (
              <div className="sm:col-span-2">
                <Notice variant="error">{formError}</Notice>
              </div>
            ) : null}

            <div>
              <label className={labelClass} htmlFor="new-username">
                Identificador de acceso
              </label>
              <input
                id="new-username"
                name="username"
                type="text"
                autoComplete="off"
                autoCorrect="off"
                autoCapitalize="none"
                spellCheck={false}
                className={inputClass}
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                aria-describedby="new-username-hint"
                maxLength={64}
                placeholder="tecnologo.rayos"
                disabled={creating}
                required
              />
              <p id="new-username-hint" className="mt-1.5 text-xs text-slate-500">
                3–64 caracteres: letras, números, punto, guion o guion bajo.
              </p>
            </div>

            <div>
              <label className={labelClass} htmlFor="new-display-name">
                Nombre visible
              </label>
              <input
                id="new-display-name"
                name="display_name"
                type="text"
                className={inputClass}
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
                maxLength={100}
                placeholder="Nombre del integrante"
                disabled={creating}
                required
              />
            </div>

            <div>
              <label className={labelClass} htmlFor="new-role">
                Rol asignado
              </label>
              <select
                id="new-role"
                name="role"
                className={inputClass}
                value={role}
                onChange={(event) => setRole(event.target.value as Role)}
                disabled={creating}
              >
                {(roles.length > 0
                  ? roles
                  : (Object.keys(ROLE_LABELS) as Role[]).map((value) => ({
                      value,
                      label: ROLE_LABELS[value],
                      permissions: [],
                    }))
                ).map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className={labelClass} htmlFor="new-password">
                Contraseña temporal
              </label>
              <div className="relative">
                <input
                  id="new-password"
                  name="password"
                  type={showPassword ? "text" : "password"}
                  className={`${inputClass} pr-12`}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  maxLength={128}
                  autoComplete="new-password"
                  aria-describedby="new-password-hint"
                  placeholder="Mínimo 12 caracteres"
                  disabled={creating}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((current) => !current)}
                  className="absolute inset-y-0 right-0 flex w-11 items-center justify-center rounded-r-xl text-slate-400 transition hover:text-slate-700"
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
              <p id="new-password-hint" className="mt-1.5 text-xs text-slate-500">
                Entre 12 y 128 caracteres; distinta al identificador.
              </p>
            </div>

            <div className="flex flex-col-reverse gap-3 border-t border-slate-100 pt-4 sm:col-span-2 sm:flex-row sm:justify-end">
              <button type="button" className={buttonSecondary} onClick={closeForm} disabled={creating}>
                Cancelar
              </button>
              <button type="submit" className={buttonPrimary} disabled={creating}>
                {creating ? <Spinner className="h-4 w-4" /> : <UserPlus className="h-4 w-4" />}
                {creating ? "Creando cuenta…" : "Crear cuenta"}
              </button>
            </div>
          </form>
        </section>
      ) : null}

      {loading ? <LoadingPanel label="Cargando cuentas autorizadas…" /> : null}

      {!loading && !error && users.length === 0 ? (
        <EmptyState
          icon={Users}
          title="Aún no hay cuentas registradas"
          description="Cree la primera cuenta autorizada para habilitar el acceso del equipo."
          action={
            <button type="button" className={buttonPrimary} onClick={() => setFormOpen(true)}>
              <Plus className="h-4 w-4" aria-hidden="true" />
              Registrar usuario
            </button>
          }
        />
      ) : null}

      {!loading && users.length > 0 ? (
        <section className={`${cardClass} overflow-hidden`}>
          <div className="flex flex-col gap-2 border-b border-slate-100 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-base font-semibold text-slate-900">Cuentas del sistema</h2>
              <p className="mt-1 text-sm text-slate-600">
                {users.length} cuenta{users.length === 1 ? "" : "s"} registrada
                {users.length === 1 ? "" : "s"}.
              </p>
            </div>
            <span className="inline-flex items-center gap-2 rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-xs font-semibold text-teal-800">
              <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
              Permisos aplicados en servidor
            </span>
          </div>

          <ul className="divide-y divide-slate-100">
            {users.map((account) => {
              const isSelf = account.id === currentUser?.id;
              const isUpdating = updatingUserId === account.id;
              return (
                <li
                  key={account.id}
                  className="grid gap-4 px-5 py-4 md:grid-cols-[minmax(0,1.6fr)_minmax(0,1.2fr)_auto_auto] md:items-center"
                >
                  <div className="flex min-w-0 items-center gap-3">
                    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-slate-100 text-sm font-semibold text-slate-700">
                      {initials(account.display_name)}
                    </span>
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="truncate text-sm font-semibold text-slate-900">
                          {account.display_name}
                        </p>
                        {isSelf ? (
                          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-semibold text-slate-600">
                            Su cuenta
                          </span>
                        ) : null}
                      </div>
                      <p className="truncate text-xs text-slate-500">
                        @{account.username} · Alta: {formatDate(account.created_at)}
                      </p>
                    </div>
                  </div>

                  <div>
                    <label className="sr-only" htmlFor={`role-${account.id}`}>
                      Rol de {account.display_name}
                    </label>
                    <select
                      id={`role-${account.id}`}
                      className={`${inputClass} py-2 text-xs`}
                      value={account.role}
                      onChange={(event) => void handleRoleChange(account, event.target.value as Role)}
                      disabled={isUpdating}
                    >
                      {(roles.length > 0
                        ? roles
                        : (Object.keys(ROLE_LABELS) as Role[]).map((value) => ({
                            value,
                            label: ROLE_LABELS[value],
                            permissions: [],
                          }))
                      ).map((item) => (
                        <option key={item.value} value={item.value}>
                          {roleLabel(item.value)}
                        </option>
                      ))}
                    </select>
                  </div>

                  <span
                    className={`inline-flex w-fit items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold ${
                      account.is_active
                        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                        : "border-slate-200 bg-slate-100 text-slate-500"
                    }`}
                  >
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        account.is_active ? "bg-emerald-500" : "bg-slate-400"
                      }`}
                    />
                    {account.is_active ? "Activa" : "Inactiva"}
                  </span>

                  <div className="flex justify-start md:justify-end">
                    <button
                      type="button"
                      className={account.is_active ? buttonDanger : buttonSecondary}
                      onClick={() => requestToggleActive(account)}
                      disabled={isUpdating || (isSelf && account.is_active)}
                      title={
                        isSelf && account.is_active
                          ? "No puede desactivar su propia cuenta"
                          : undefined
                      }
                    >
                      {isUpdating ? <Spinner className="h-4 w-4" /> : null}
                      {account.is_active ? "Desactivar" : "Reactivar"}
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}

      <dialog
        ref={confirmDialogRef}
        aria-labelledby="deactivate-dialog-title"
        onClose={handleConfirmDialogClose}
        className="m-auto w-full max-w-md rounded-2xl border border-slate-200 bg-white p-6 text-left shadow-2xl"
      >
        <h2 id="deactivate-dialog-title" className="text-base font-semibold text-slate-900">
          ¿Desactivar cuenta de usuario?
        </h2>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          La cuenta de{" "}
          <strong className="text-slate-900">{pendingDeactivate?.display_name}</strong> perderá el
          acceso de inmediato y sus sesiones activas dejarán de ser válidas.
        </p>
        <form method="dialog" className="mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
          <button type="submit" value="cancel" className={buttonSecondary} autoFocus>
            Cancelar
          </button>
          <button type="submit" value="confirm" className={buttonDanger}>
            Desactivar cuenta
          </button>
        </form>
      </dialog>
    </div>
  );
}
