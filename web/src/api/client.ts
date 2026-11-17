import type {
  ConcordanceState,
  Evaluation,
  EvaluationStatus,
  LoginResponse,
  ModelVersion,
  Observation,
  ObservationAgreement,
  Prediction,
  QualityCheck,
  Radiograph,
  Role,
  RoleInfo,
  User,
} from "../types";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

interface ValidationDetail {
  msg?: unknown;
  loc?: unknown;
}

function messageFromPayload(payload: unknown): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item: ValidationDetail) => (typeof item?.msg === "string" ? item.msg : null))
        .filter((message): message is string => Boolean(message));
      if (messages.length > 0) {
        return messages.join(" ");
      }
    }
  }
  return "No se pudo completar la solicitud.";
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new ApiError(response.status, messageFromPayload(payload));
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export interface CreateUserInput {
  username: string;
  display_name: string;
  role: Role;
  password: string;
}

export interface UpdateUserInput {
  role?: Role;
  is_active?: boolean;
}

export interface EvaluationFilters {
  status?: EvaluationStatus;
  search?: string;
  review_required?: boolean;
  concordance_state?: ConcordanceState;
}

export interface CreateObservationInput {
  clinical_impression: string;
  agreement_status: ObservationAgreement;
  mark_reviewed: boolean;
}

async function fetchProtectedBlob(url: string): Promise<Blob> {
  const response = await fetch(url, { credentials: "same-origin" });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new ApiError(response.status, messageFromPayload(payload));
  }
  return response.blob();
}

export const api = {
  login(username: string, password: string): Promise<LoginResponse> {
    return request<LoginResponse>("/api/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
  },

  logout(): Promise<{ message: string }> {
    return request<{ message: string }>("/api/v1/auth/logout", { method: "POST" });
  },

  me(): Promise<User> {
    return request<User>("/api/v1/auth/me");
  },

  listUsers(): Promise<User[]> {
    return request<User[]>("/api/v1/users");
  },

  createUser(input: CreateUserInput): Promise<User> {
    return request<User>("/api/v1/users", {
      method: "POST",
      body: JSON.stringify(input),
    });
  },

  updateUser(userId: string, input: UpdateUserInput): Promise<User> {
    return request<User>(`/api/v1/users/${encodeURIComponent(userId)}`, {
      method: "PATCH",
      body: JSON.stringify(input),
    });
  },

  listRoles(): Promise<RoleInfo[]> {
    return request<RoleInfo[]>("/api/v1/roles");
  },

  listEvaluations(filters: EvaluationFilters = {}): Promise<Evaluation[]> {
    const parameters = new URLSearchParams();
    if (filters.status) {
      parameters.set("status", filters.status);
    }
    if (filters.search) {
      parameters.set("search", filters.search);
    }
    if (filters.review_required !== undefined) {
      parameters.set("review_required", String(filters.review_required));
    }
    if (filters.concordance_state) {
      parameters.set("concordance_state", filters.concordance_state);
    }
    const query = parameters.toString();
    return request<Evaluation[]>(`/api/v1/evaluations${query ? `?${query}` : ""}`);
  },

  createEvaluation(notes: string | null): Promise<Evaluation> {
    return request<Evaluation>("/api/v1/evaluations", {
      method: "POST",
      body: JSON.stringify({ notes }),
    });
  },

  getEvaluation(evaluationId: string): Promise<Evaluation> {
    return request<Evaluation>(`/api/v1/evaluations/${encodeURIComponent(evaluationId)}`);
  },

  uploadRadiograph(evaluationId: string, file: File): Promise<Radiograph> {
    const formData = new FormData();
    formData.append("file", file);
    return request<Radiograph>(
      `/api/v1/evaluations/${encodeURIComponent(evaluationId)}/radiograph`,
      { method: "POST", body: formData },
    );
  },

  replaceRadiograph(evaluationId: string, file: File): Promise<Radiograph> {
    const formData = new FormData();
    formData.append("file", file);
    return request<Radiograph>(
      `/api/v1/evaluations/${encodeURIComponent(evaluationId)}/radiograph`,
      { method: "PUT", body: formData },
    );
  },

  validateRadiographQuality(evaluationId: string): Promise<QualityCheck> {
    return request<QualityCheck>(
      `/api/v1/evaluations/${encodeURIComponent(evaluationId)}/radiograph/validate`,
      { method: "POST" },
    );
  },

  analyzeEvaluation(evaluationId: string): Promise<Evaluation> {
    return request<Evaluation>(
      `/api/v1/evaluations/${encodeURIComponent(evaluationId)}/analyze`,
      { method: "POST" },
    );
  },

  getPrediction(evaluationId: string): Promise<Prediction> {
    return request<Prediction>(
      `/api/v1/evaluations/${encodeURIComponent(evaluationId)}/prediction`,
    );
  },

  fetchRadiograph(contentUrl: string): Promise<Blob> {
    return fetchProtectedBlob(contentUrl);
  },

  fetchExplainabilityMap(mapUrl: string): Promise<Blob> {
    return fetchProtectedBlob(mapUrl);
  },

  listObservations(evaluationId: string): Promise<Observation[]> {
    return request<Observation[]>(
      `/api/v1/evaluations/${encodeURIComponent(evaluationId)}/observations`,
    );
  },

  createObservation(evaluationId: string, input: CreateObservationInput): Promise<Observation> {
    return request<Observation>(
      `/api/v1/evaluations/${encodeURIComponent(evaluationId)}/observations`,
      {
        method: "POST",
        body: JSON.stringify(input),
      },
    );
  },

  listModelVersions(): Promise<ModelVersion[]> {
    return request<ModelVersion[]>("/api/v1/model-versions");
  },
};
