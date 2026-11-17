import { BrowserRouter, Route, Routes } from "react-router-dom";

import { AuthProvider, RequireAuth, RequirePermission } from "./auth/AuthContext";
import { AppShell } from "./components/AppShell";
import { DashboardPage } from "./pages/DashboardPage";
import { EvaluationDetailPage } from "./pages/EvaluationDetailPage";
import { EvaluationsPage } from "./pages/EvaluationsPage";
import { LoginPage } from "./pages/LoginPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { UsersPage } from "./pages/UsersPage";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route
            element={
              <RequireAuth>
                <AppShell />
              </RequireAuth>
            }
          >
            <Route index element={<DashboardPage />} />
            <Route
              path="evaluaciones"
              element={
                <RequirePermission permission="evaluations.read">
                  <EvaluationsPage />
                </RequirePermission>
              }
            />
            <Route
              path="evaluaciones/:id"
              element={
                <RequirePermission permission="evaluations.read">
                  <EvaluationDetailPage />
                </RequirePermission>
              }
            />
            <Route
              path="usuarios"
              element={
                <RequirePermission permission="users.read">
                  <UsersPage />
                </RequirePermission>
              }
            />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
