import { AfterViewChecked, Component, ElementRef, ViewChild, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import {
  ChatService,
  ChatResponse,
  HealthResponse,
  ModelsResponse,
  RecommendationData,
  RecommendationProfile,
  RecommendationResponse,
} from './chat.service';

type Role = 'user' | 'assistant' | 'system';
type ModelVariant = 'prod' | 'dpo';
type RuntimeMode = 'collect_execute' | 'speak' | 'autonomous';
type WorkspaceView = 'assistant' | 'recommendation';

interface Message {
  role: Role;
  text: string;
  timestamp: Date;
  meta?: {
    intent?: string;
    slots?: Record<string, string>;
    missingSlots?: string[];
    toolCall?: { name: string; arguments: Record<string, unknown> } | null;
    toolResult?: Record<string, unknown> | null;
    ragResults?: Array<Record<string, unknown>>;
    memoryHits?: Array<Record<string, unknown>>;
    sessionState?: Record<string, unknown>;
    needsHumanReview?: boolean;
    latencyMs?: number;
    modelVariant?: string;
    runtimeMode?: string;
    correctionApplied?: boolean;
    recommendation?: RecommendationData | Record<string, unknown> | null;
    missingFields?: string[];
    nextQuestions?: string[];
  };
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App implements AfterViewChecked {
  @ViewChild('chatContainer') private chatContainer?: ElementRef<HTMLDivElement>;
  @ViewChild('recoContainer') private recoContainer?: ElementRef<HTMLDivElement>;

  messages = signal<Message[]>([]);
  recommendationMessages = signal<Message[]>([]);
  inputText = '';
  recommendationInput = '';
  isLoading = signal(false);
  recommendationLoading = signal(false);
  backendOnline = signal(false);
  health = signal<HealthResponse | null>(null);
  models = signal<ModelsResponse | null>(null);
  showDebug = signal(false);
  selectedVariant = signal<ModelVariant>('prod');
  selectedRuntimeMode = signal<RuntimeMode>('autonomous');
  activeView = signal<WorkspaceView>('assistant');
  selectedMessage = signal<Message | null>(null);
  recommendationState = signal<RecommendationResponse | null>(null);

  recommendationSessionId = '';
  recommendationProfile: RecommendationProfile = {};

  suggestions = [
    'قداش سوم البلارات البروقرسيف 1.67؟',
    'وقتاش يفتح المحل في صفاقس؟',
    'نحب نتبع الكوموند ORD-ABC12345',
    'نحب رنديفو في تونس نهار 2026-03-20 على 10:00 ورقمي +216 22 333 444',
  ];

  recommendationSuggestions = [
    "J'ai 46 ans, montage plastique, correction entre 225 et 400, ADD 2.25.",
    'Mon besoin principal: transparence. Je travaille en interieur.',
    'Gene a la lumiere faible. Je ne veux pas de paire solaire.',
    'Pouvez-vous me faire une recommandation finale complete ?',
  ];

  constructor(private readonly chatService: ChatService) {
    this.recommendationSessionId = this.chatService.createEphemeralSessionId();
    this.refreshRuntime();
  }

  ngAfterViewChecked(): void {
    this.scrollToBottom();
  }

  refreshRuntime(): void {
    this.chatService.getHealth().subscribe({
      next: (health) => {
        this.health.set(health);
        this.backendOnline.set(true);
      },
      error: () => this.backendOnline.set(false),
    });

    this.chatService.getModels().subscribe({
      next: (models) => this.models.set(models),
      error: () => this.models.set(null),
    });
  }

  setVariant(variant: ModelVariant): void {
    this.selectedVariant.set(variant);
  }

  setRuntimeMode(mode: RuntimeMode): void {
    this.selectedRuntimeMode.set(mode);
  }

  setView(view: WorkspaceView): void {
    this.activeView.set(view);
  }

  send(text?: string): void {
    const message = (text ?? this.inputText).trim();
    if (!message || this.isLoading()) {
      return;
    }

    this.messages.update((current) => [
      ...current,
      { role: 'user', text: message, timestamp: new Date() },
    ]);
    this.inputText = '';
    this.isLoading.set(true);

    this.chatService.sendMessage(message, this.selectedVariant(), this.selectedRuntimeMode()).subscribe({
      next: (response: ChatResponse) => {
        this.messages.update((current) => [
          ...current,
          {
            role: 'assistant',
            text: response.response,
            timestamp: new Date(),
            meta: {
              intent: response.intent,
              slots: response.slots,
              missingSlots: response.missing_slots,
              toolCall: response.tool_call,
              toolResult: response.tool_result,
              ragResults: response.rag_results,
              memoryHits: response.memory_hits,
              sessionState: response.session_state,
              needsHumanReview: response.needs_human_review,
              latencyMs: response.latency_ms,
              modelVariant: response.model_variant,
              runtimeMode: response.runtime_mode,
              correctionApplied: response.correction_applied,
            },
          },
        ]);
        this.isLoading.set(false);
      },
      error: (error) => {
        this.messages.update((current) => [
          ...current,
          {
            role: 'system',
            text: `Erreur backend: ${error?.message ?? 'service indisponible'}`,
            timestamp: new Date(),
          },
        ]);
        this.isLoading.set(false);
      },
    });
  }

  sendRecommendation(text?: string): void {
    const message = (text ?? this.recommendationInput).trim();
    if (!message || this.recommendationLoading()) {
      return;
    }

    this.recommendationMessages.update((current) => [
      ...current,
      { role: 'user', text: message, timestamp: new Date() },
    ]);
    this.recommendationInput = '';
    this.recommendationLoading.set(true);

    const profilePayload = this.sanitizeRecommendationProfile(this.recommendationProfile);
    this.chatService.sendRecommendation(this.recommendationSessionId, message, profilePayload).subscribe({
      next: (response) => {
        this.recommendationState.set(response);
        this.recommendationProfile = { ...this.recommendationProfile, ...response.profile };
        this.recommendationMessages.update((current) => [
          ...current,
          {
            role: 'assistant',
            text: response.response,
            timestamp: new Date(),
            meta: {
              recommendation: response.recommendation,
              ragResults: response.rag_results,
              missingFields: response.missing_fields,
              nextQuestions: response.next_questions,
            },
          },
        ]);
        this.recommendationLoading.set(false);
      },
      error: (error) => {
        this.recommendationMessages.update((current) => [
          ...current,
          {
            role: 'system',
            text: `Erreur endpoint recommandation: ${error?.message ?? 'service indisponible'}`,
            timestamp: new Date(),
          },
        ]);
        this.recommendationLoading.set(false);
      },
    });
  }

  clearChat(): void {
    this.chatService.resetSession().subscribe({ error: () => undefined });
    this.messages.set([]);
    this.selectedMessage.set(null);
  }

  clearRecommendation(): void {
    const currentSession = this.recommendationSessionId;
    this.chatService.resetSessionById(currentSession).subscribe({ error: () => undefined });
    this.recommendationSessionId = this.chatService.createEphemeralSessionId();
    this.recommendationMessages.set([]);
    this.recommendationState.set(null);
    this.recommendationProfile = {};
    this.selectedMessage.set(null);
  }

  toggleDebug(): void {
    this.showDebug.update((value) => !value);
  }

  selectMessage(message: Message): void {
    if (!message.meta) {
      return;
    }
    this.selectedMessage.set(this.selectedMessage() === message ? null : message);
  }

  onKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.send();
    }
  }

  onRecommendationKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendRecommendation();
    }
  }

  formatJson(value: unknown): string {
    return JSON.stringify(value, null, 2);
  }

  recommendationReady(): boolean {
    return Boolean(this.recommendationState()?.recommendation);
  }

  applyQuickSuggestionForCurrentView(suggestion: string): void {
    if (this.activeView() === 'assistant') {
      this.send(suggestion);
      return;
    }
    this.sendRecommendation(suggestion);
  }

  private sanitizeRecommendationProfile(profile: RecommendationProfile): RecommendationProfile {
    const clone: RecommendationProfile = {};
    const numericFields = new Set(['age', 'correction_total', 'add_power', 'od_og_diff']);
    const booleanFields = new Set([
      'wants_sun_pair',
      'glare_exposure',
      'sun_vision_difficulty',
      'innovation_sensitive',
      'adaptation_easy',
      'night_driving',
    ]);

    for (const [key, rawValue] of Object.entries(profile)) {
      if (rawValue === undefined || rawValue === null || rawValue === '') {
        continue;
      }

      if (numericFields.has(key)) {
        const numericValue = Number(rawValue);
        if (!Number.isNaN(numericValue)) {
          clone[key as keyof RecommendationProfile] = numericValue as never;
        }
        continue;
      }

      if (booleanFields.has(key)) {
        if (typeof rawValue === 'boolean') {
          clone[key as keyof RecommendationProfile] = rawValue as never;
          continue;
        }
        if (String(rawValue).toLowerCase() === 'true') {
          clone[key as keyof RecommendationProfile] = true as never;
        } else if (String(rawValue).toLowerCase() === 'false') {
          clone[key as keyof RecommendationProfile] = false as never;
        }
        continue;
      }

      clone[key as keyof RecommendationProfile] = rawValue as never;
    }

    return clone;
  }

  private scrollToBottom(): void {
    if (!this.chatContainer) {
      // continue to recommendation container
    } else {
      this.chatContainer.nativeElement.scrollTop = this.chatContainer.nativeElement.scrollHeight;
    }

    if (this.recoContainer) {
      this.recoContainer.nativeElement.scrollTop = this.recoContainer.nativeElement.scrollHeight;
    }
  }
}
