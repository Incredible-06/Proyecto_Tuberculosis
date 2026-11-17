import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { Navigate, useLocation } from "react-router-dom";

import { ApiError, api } from "../api/client";
import type { User } from "../types";

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
  can: (permission: string) => boolean;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const currentUser = await api.me();
      setUser(currentUser);
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        setUser(null);
        return;
      }
      throw error;
    }
  }, []);

  useEffect(() => {
    let active = true;
    api
      .me()
      .then((currentUser) => {
        if (active) {
          setUser(currentUser);
        }
      })
      .catch((error: unknown) => {
        if (active && !(error instanceof ApiError && error.status === 401)) {
          console.error(error);
        }
        if (active) {
          setUser(null);
        }
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const result = await api.login(username, password);
    setUser(result.user);
  }, []);

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } finally {
      setUser(null);
    }
  }, []);

  const can = useCallback(
    (permission: string) => Boolean(user?.permissions.includes(permission)),
    [user],
  );

  const value = useMemo<AuthContextValue>(
    () => ({ user, loading, login, logout, refresh, can }),
    [user, loading, login, logout, refresh, can],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth debe usarse dentro de AuthProvider.");
  }
  return context;
}

export function RequireAuth({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-50">
        <div className="flex flex-col items-center gap-3 text-slate-500">
          <span className="h-9 w-9 animate-spin rounded-full border-2 border-slate-200 border-t-teal-600" />
          <p className="text-sm font-medium">Verificando sesión…</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }

  return <>{children}</>;
}

export function RequirePermission({
  permission,
  children,
}: {
  permission: string;
  children: ReactNode;
}) {
  const { can } = useAuth();
  if (!can(permission)) {
    return (
      <div className="mx-auto max-w-xl rounded-2xl border border-amber-200 bg-amber-50 p-8 text-center">
        <h1 className="text-lg font-semibold text-amber-900">Acceso restringido</h1>
        <p className="mt-2 text-sm text-amber-800">
          Su rol no tiene permiso para consultar esta sección. Si necesita acceso, contacte al
          administrador del sistema.
        </p>
      </div>
    );
  }
  return <>{children}</>;
}
