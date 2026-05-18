import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ChatResponse {
  response: string;
  intent: string;
  language_style: string;
  slots: Record<string, string>;
  missing_slots: string[];
  tool_call: { name: string; arguments: Record<string, unknown> } | null;
  tool_result: Record<string, unknown> | null;
  rag_results: Array<Record<string, unknown>>;
  memory_hits: Array<Record<string, unknown>>;
  session_state: Record<string, unknown>;
  needs_human_review: boolean;
  latency_ms: number;
  session_id: string;
  model_variant: string;
  runtime_mode: string;
  response_source: string;
  response_script_target: string;
  response_script_detected: string;
  correction_applied: boolean;
  routing_reason: string;
}

export interface HealthResponse {
  status: string;
  runtime: {
    gpu_count: number;
    gpus: { name: string; vram_gb: number }[];
    cuda_available: boolean;
  };
  domain: {
    name: string;
    organization: string;
    assistant_name: string;
    languages: string[];
  };
  models: Record<string, string | null>;
  rag: {
    chunks: number;
    embedding_backend: string;
    faiss_enabled: boolean;
  } | null;
  learning: {
    pending_candidates: number;
    approved_sft: number;
    approved_dpo: number;
    feedback_events: number;
  };
}

export interface ModelsResponse {
  active_variants: Record<string, string | null>;
  default_variant: string;
}

export interface ToolDescriptor {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface RecommendationProfile {
  age?: number;
  frame_type?: 'nylor' | 'plastique' | 'metallique' | 'perce';
  correction_total?: number;
  add_power?: number;
  od_og_diff?: number;
  main_need?: 'transparence' | 'lumiere_bleue' | 'conduite_soir' | 'rayures' | 'nettoyage';
  ocular_health?: string;
  work_env?: 'interieur' | 'exterieur' | 'mixte';
  light_discomfort?: 'faible' | 'moyenne' | 'forte';
  wants_sun_pair?: boolean;
  glare_exposure?: boolean;
  sun_vision_difficulty?: boolean;
  head_eye_behavior?: 'head' | 'eyes' | 'mixed';
  innovation_sensitive?: boolean;
  adaptation_easy?: boolean;
  night_driving?: boolean;
  computer_usage?: 'high' | 'medium' | 'low';
}

export interface RecommendationData {
  lens_type: string;
  index: string;
  treatment: string;
  color: string;
  confidence: number;
  rationale: string[];
  applied_rules: string[];
}

export interface RecommendationResponse {
  session_id: string;
  response: string;
  profile: RecommendationProfile;
  extracted_updates: Record<string, unknown>;
  incoming_updates: Record<string, unknown>;
  missing_fields: string[];
  next_questions: string[];
  next_question_fields: string[];
  recommendation: RecommendationData | null;
  rag_results: Array<Record<string, unknown>>;
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly api: string;
  private sessionId = this.createSessionId('chat');

  constructor(private readonly http: HttpClient) {
    this.api = ChatService.resolveApi();
  }

  private static resolveApi(): string {
    const runtimeOverride = (window as Window & { __CALL_CENTER_API_URL__?: string }).__CALL_CENTER_API_URL__;
    if (runtimeOverride && runtimeOverride.trim()) {
      return runtimeOverride.trim().replace(/\/$/, '');
    }

    const queryApi = new URLSearchParams(window.location.search).get('api');
    if (queryApi && queryApi.trim()) {
      return queryApi.trim().replace(/\/$/, '');
    }

    const host = window.location.hostname;
    const isLocalHost = host === 'localhost' || host === '127.0.0.1';
    if (window.location.port === '8001') {
      return '/api';
    }

    if (host.includes('devtunnels.ms') || (!isLocalHost && window.location.protocol.startsWith('http'))) {
      return '/api';
    }

    if (isLocalHost) {
      return 'http://localhost:8000';
    }

    return `${window.location.protocol}//${host}:8000`;
  }

  private createSessionId(prefix: 'chat' | 'reco'): string {
    if (globalThis.crypto?.randomUUID) {
      return `${prefix}-${globalThis.crypto.randomUUID()}`;
    }
    return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }

  createEphemeralSessionId(): string {
    return this.createSessionId('reco');
  }

  getSessionId(): string {
    return this.sessionId;
  }

  private resolveUserContext(): Record<string, unknown> {
    const context: Record<string, unknown> = {};
    const append = (key: string, value: unknown): void => {
      if (value !== undefined && value !== null && String(value).trim() !== '') {
        context[key] = value;
      }
    };

    const runtimeSession = (window as Window & { __OPTIFLOW_USER_SESSION__?: Record<string, unknown> })
      .__OPTIFLOW_USER_SESSION__;
    if (runtimeSession && typeof runtimeSession === 'object') {
      Object.entries(runtimeSession).forEach(([key, value]) => append(key, value));
    }

    try {
      for (const storageKey of ['optiflow_user_session', 'optiflowUserSession', 'user_session', 'user']) {
        const raw = localStorage.getItem(storageKey);
        if (!raw) {
          continue;
        }
        const parsed = JSON.parse(raw) as Record<string, unknown>;
        if (parsed && typeof parsed === 'object') {
          Object.entries(parsed).forEach(([key, value]) => append(key, value));
        }
      }
      append('ACCESS_TOKEN', localStorage.getItem('access_token') ?? localStorage.getItem('accessToken'));
    } catch {
      // Ignore malformed localStorage values.
    }

    const query = new URLSearchParams(window.location.search);
    append('USER_ID', query.get('userId'));
    append('USER_PRENOM', query.get('prenom'));
    append('USER_NOM', query.get('nom'));
    append('USER_ROLE', query.get('role'));
    append('USER_CODE_CLIENT', query.get('codeClient'));
    append('USER_AGENCE', query.get('agence'));
    append('ACCESS_TOKEN', query.get('accessToken'));
    append('BACKEND_URL', query.get('backendUrl'));

    return context;
  }

  sendMessage(message: string, modelVariant: string, runtimeMode: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.api}/chat`, {
      message,
      session_id: this.sessionId,
      model_variant: modelVariant,
      runtime_mode: runtimeMode,
      user_context: this.resolveUserContext(),
    });
  }

  resetSession(): Observable<{ status: string; session_id: string | null }> {
    const currentSessionId = this.sessionId;
    this.sessionId = this.createSessionId('chat');
    const params = new HttpParams().set('session_id', currentSessionId);
    return this.http.post<{ status: string; session_id: string | null }>(`${this.api}/reset`, null, { params });
  }

  resetSessionById(sessionId: string): Observable<{ status: string; session_id: string | null }> {
    const params = new HttpParams().set('session_id', sessionId);
    return this.http.post<{ status: string; session_id: string | null }>(`${this.api}/reset`, null, { params });
  }

  sendRecommendation(
    sessionId: string,
    message: string,
    profile: RecommendationProfile,
    options?: { topK?: number; reset?: boolean }
  ): Observable<RecommendationResponse> {
    return this.http.post<RecommendationResponse>(`${this.api}/chat/recommendation`, {
      session_id: sessionId,
      message,
      profile,
      top_k: options?.topK ?? 4,
      reset: options?.reset ?? false,
    });
  }

  getHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.api}/health`);
  }

  getModels(): Observable<ModelsResponse> {
    return this.http.get<ModelsResponse>(`${this.api}/models`);
  }

  getTools(): Observable<{ tools: ToolDescriptor[] }> {
    return this.http.get<{ tools: ToolDescriptor[] }>(`${this.api}/tools`);
  }
}
