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

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly api: string;
  private sessionId = this.createSessionId();

  constructor(private readonly http: HttpClient) {
    this.api = ChatService.resolveApi();
  }

  private static resolveApi(): string {
    const host = window.location.hostname;
    if (host.includes('devtunnels.ms')) {
      return `https://${host.replace('-4200', '-8000')}`;
    }
    return 'http://localhost:8000';
  }

  private createSessionId(): string {
    if (globalThis.crypto?.randomUUID) {
      return globalThis.crypto.randomUUID();
    }
    return `session-${Date.now()}-${Math.random().toString(36).slice(2)}`;
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
    this.sessionId = this.createSessionId();
    const params = new HttpParams().set('session_id', currentSessionId);
    return this.http.post<{ status: string; session_id: string | null }>(`${this.api}/reset`, null, { params });
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
