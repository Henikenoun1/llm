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
    if (host.includes('devtunnels.ms')) {
      return `https://${host.replace('-4200', '-8000')}`;
    }

    if (host === 'localhost' || host === '127.0.0.1') {
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

  sendMessage(message: string, modelVariant: string, runtimeMode: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.api}/chat`, {
      message,
      session_id: this.sessionId,
      model_variant: modelVariant,
      runtime_mode: runtimeMode,
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

  getTools(): Observable<{ tools: unknown[] }> {
    return this.http.get<{ tools: unknown[] }>(`${this.api}/tools`);
  }
}
