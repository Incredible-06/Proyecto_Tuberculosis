import {
  ClipboardList,
  LayoutDashboard,
  LogOut,
  Menu,
  ShieldCheck,
  Users,
  X,
} from "lucide-react";
import { useEffect, useState, type ComponentType } from "react";
import { Link, NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";
import { ROLE_LABELS } from "../types";
import { buttonGhost } from "./styles";
import { Spinner } from "./ui";

interface NavigationItem {
  to: string;
  label: string;
  icon: ComponentType<{ className?: string }>;
  end?: boolean;
}

function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <Link to="/" className="flex items-center gap-3">
      <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-700 text-white shadow-sm">
        <ShieldCheck className="h-5 w-5" aria-hidden="true" />
      </span>
      <span>
        <span className="block text-sm font-semibold tracking-tight text-slate-900">
          Sistema TB
        </span>
        <span className="block text-xs text-slate-500">
          {compact ? "Apoyo a la decisión" : "Apoyo a la decisión clínica"}
        </span>
      </span>
    </Link>
  );
}

function Navigation({ onNavigate }: { onNavigate?: () => void }) {
  const { can } = useAuth();
  const items: NavigationItem[] = [{ to: "/", label: "Inicio", icon: LayoutDashboard, end: true }];
  if (can("evaluations.read")) {
    items.push({ to: "/evaluaciones", label: "Evaluaciones", icon: ClipboardList });
  }
  if (can("users.read")) {
    items.push({ to: "/usuarios", label: "Usuarios", icon: Users });
  }

  return (
    <nav className="flex flex-1 flex-col gap-1 px-3" aria-label="Navegación principal">
      {items.map((item) => {
        const Icon = item.icon;
        return (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            onClick={onNavigate}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${
                isActive
                  ? "bg-teal-50 text-teal-800"
                  : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
              }`
            }
          >
            <Icon className="h-4.5 w-4.5" />
            {item.label}
          </NavLink>
        );
      })}
    </nav>
  );
}

function UserPanel({ onNavigate }: { onNavigate?: () => void }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [loggingOut, setLoggingOut] = useState(false);

  if (!user) {
    return null;
  }

  const handleLogout = async () => {
    setLoggingOut(true);
    try {
      await logout();
      onNavigate?.();
      navigate("/login", { replace: true });
    } finally {
      setLoggingOut(false);
    }
  };

  return (
    <div className="border-t border-slate-200 p-3">
      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
        <div className="flex items-center gap-3">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-teal-700 text-sm font-semibold text-white">
            {user.display_name
              .split(/\s+/)
              .filter(Boolean)
              .slice(0, 2)
              .map((part) => part[0]?.toUpperCase() ?? "")
              .join("")}
          </span>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-slate-900">{user.display_name}</p>
            <p className="truncate text-xs text-slate-500">{ROLE_LABELS[user.role]}</p>
          </div>
        </div>
        <button
          type="button"
          onClick={handleLogout}
          disabled={loggingOut}
          className={`${buttonGhost} mt-2 w-full justify-start`}
        >
          {loggingOut ? <Spinner className="h-4 w-4" /> : <LogOut className="h-4 w-4" />}
          {loggingOut ? "Cerrando sesión…" : "Cerrar sesión"}
        </button>
      </div>
    </div>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex h-full flex-col bg-white">
      <div className="px-5 py-5">
        <Brand />
      </div>
      <Navigation onNavigate={onNavigate} />
      <UserPanel onNavigate={onNavigate} />
    </div>
  );
}

export function AppShell() {
  const { user } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  return (
    <div className="min-h-screen bg-slate-50">
      <a
        href="#main-content"
        className="visually-hidden fixed left-3 top-3 z-50 rounded-xl bg-teal-700 px-4 py-2 text-sm font-semibold text-white shadow-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-400"
      >
        Ir al contenido principal
      </a>
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-72 border-r border-slate-200 lg:block">
        <SidebarContent />
      </aside>

      <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-slate-200 bg-white/95 px-4 backdrop-blur lg:hidden">
        <button
          type="button"
          onClick={() => setMobileOpen(true)}
          className="rounded-lg p-2 text-slate-600 transition hover:bg-slate-100"
          aria-label="Abrir menú de navegación"
        >
          <Menu className="h-5 w-5" />
        </button>
        <Brand compact />
        {user ? (
          <span className="flex h-9 w-9 items-center justify-center rounded-full bg-teal-700 text-xs font-semibold text-white">
            {user.display_name
              .split(/\s+/)
              .filter(Boolean)
              .slice(0, 2)
              .map((part) => part[0]?.toUpperCase() ?? "")
              .join("")}
          </span>
        ) : null}
      </header>

      {mobileOpen ? (
        <div className="fixed inset-0 z-40 lg:hidden">
          <button
            type="button"
            className="absolute inset-0 bg-slate-900/40"
            onClick={() => setMobileOpen(false)}
            aria-label="Cerrar menú"
          />
          <div className="absolute inset-y-0 left-0 w-80 max-w-[85vw] shadow-2xl">
            <button
              type="button"
              onClick={() => setMobileOpen(false)}
              className="absolute right-3 top-3 z-10 rounded-lg p-2 text-slate-500 transition hover:bg-slate-100"
              aria-label="Cerrar menú"
            >
              <X className="h-5 w-5" />
            </button>
            <SidebarContent onNavigate={() => setMobileOpen(false)} />
          </div>
        </div>
      ) : null}

      <div className="flex min-h-screen flex-col lg:pl-72">
        <main
          id="main-content"
          tabIndex={-1}
          className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 outline-none sm:px-6 lg:px-8 lg:py-8"
        >
          <Outlet />
        </main>
        <footer className="border-t border-slate-200 bg-white px-4 py-3.5 text-center text-xs text-slate-500 sm:px-6 lg:px-8">
          Herramienta de apoyo a la decisión clínica · No sustituye la valoración médica ni las
          pruebas confirmatorias
        </footer>
      </div>
    </div>
  );
}


