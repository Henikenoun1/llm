import { Component, ElementRef, HostListener, ViewChild, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import {
  ChatResponse,
  ChatService,
  HealthResponse,
  ModelsResponse,
  RecommendationData,
  RecommendationProfile,
  RecommendationResponse,
  ToolDescriptor,
} from './chat.service';

type Role = 'user' | 'assistant' | 'system';
type RuntimeMode = 'collect_execute' | 'speak' | 'autonomous';
type WorkspaceView = 'assistant' | 'recommendation';

interface MessageMeta {
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
  responseSource?: string;
  responseScriptTarget?: string;
  responseScriptDetected?: string;
  correctionApplied?: boolean;
  recommendation?: RecommendationData | Record<string, unknown> | null;
  missingFields?: string[];
  nextQuestions?: string[];
}

interface Message {
  role: Role;
  text: string;
  timestamp: Date;
  meta?: MessageMeta;
}

interface KeyValueRow {
  key: string;
  value: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
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
  tools = signal<ToolDescriptor[]>([]);
  showDebug = signal(false);
  inspectorOpen = signal(false);
  selectedVariant = signal('prod');
  selectedRuntimeMode = signal<RuntimeMode>('autonomous');
  activeView = signal<WorkspaceView>('assistant');
  selectedMessage = signal<Message | null>(null);
  recommendationState = signal<RecommendationResponse | null>(null);

  recommendationSessionId = '';
  recommendationProfile: RecommendationProfile = {};

  assistantSuggestions = [
    'عسلامة، num client 5007، وين وصلت الكوموند ORD-ABC12345؟',
    'aslema nheb dispo mta3 reference 25YXSU fi tunis',
    'salem nheb na3mel commande progressive 1.67 marron, num client 3310',
    'شنوة créneau livraison disponible pour Aouina centre ville؟',
    'قداش سوم progressive 1.67 avec Crizal à Tunis؟',
    'bonjour, confirmez-moi la reference 25YXSU avant commande',
  ];

  recommendationSuggestions = [
    "J'ai 46 ans, montage plastique, correction entre 225 et 400, ADD 2.25.",
    'Mon besoin principal: transparence. Je travaille en interieur.',
    'Gene a la lumiere faible. Je ne veux pas de paire solaire.',
    'Pouvez-vous me faire une recommandation finale complete ?',
  ];

  constructor(private readonly chatService: ChatService) {
    this.recommendationSessionId = this.chatService.createEphemeralSessionId();
    this.inspectorOpen.set(this.isWideViewport());
    this.refreshRuntime();
  }

  @HostListener('window:resize')
  onViewportResize(): void {
    if (this.isWideViewport()) {
      this.inspectorOpen.set(true);
    } else if (!this.selectedMessage() && !this.showDebug()) {
      this.inspectorOpen.set(false);
    }
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
      next: (models) => {
        this.models.set(models);
        const variants = this.variantOptionsFrom(models);
        if (!variants.includes(this.selectedVariant())) {
          this.selectedVariant.set(variants[0] ?? models.default_variant ?? 'prod');
        }
      },
      error: () => this.models.set(null),
    });

    this.chatService.getTools().subscribe({
      next: (payload) => this.tools.set(payload.tools ?? []),
      error: () => this.tools.set([]),
    });
  }

  setVariant(variant: string): void {
    this.selectedVariant.set(variant);
  }

  setRuntimeMode(mode: RuntimeMode): void {
    this.selectedRuntimeMode.set(mode);
  }

  setView(view: WorkspaceView): void {
    this.activeView.set(view);
    this.selectedMessage.set(null);
    if (!this.isWideViewport()) {
      this.inspectorOpen.set(false);
    }
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
    this.scheduleScroll('assistant');

    this.chatService.sendMessage(message, this.selectedVariant(), this.selectedRuntimeMode()).subscribe({
      next: (response: ChatResponse) => {
        const assistantMessage: Message = {
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
            responseSource: response.response_source,
            responseScriptTarget: response.response_script_target,
            responseScriptDetected: response.response_script_detected,
            correctionApplied: response.correction_applied,
          },
        };

        this.messages.update((current) => [...current, assistantMessage]);
        this.selectedMessage.set(assistantMessage);
        this.isLoading.set(false);
        this.scheduleScroll('assistant');
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
        this.scheduleScroll('assistant');
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
    this.scheduleScroll('recommendation');

    const profilePayload = this.sanitizeRecommendationProfile(this.recommendationProfile);
    this.chatService.sendRecommendation(this.recommendationSessionId, message, profilePayload).subscribe({
      next: (response) => {
        this.recommendationState.set(response);
        this.recommendationProfile = { ...this.recommendationProfile, ...response.profile };
        const assistantMessage: Message = {
          role: 'assistant',
          text: response.response,
          timestamp: new Date(),
          meta: {
            recommendation: response.recommendation,
            ragResults: response.rag_results,
            missingFields: response.missing_fields,
            nextQuestions: response.next_questions,
          },
        };
        this.recommendationMessages.update((current) => [...current, assistantMessage]);
        this.selectedMessage.set(assistantMessage);
        this.recommendationLoading.set(false);
        this.scheduleScroll('recommendation');
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
        this.scheduleScroll('recommendation');
      },
    });
  }

  clearChat(): void {
    this.chatService.resetSession().subscribe({ error: () => undefined });
    this.messages.set([]);
    this.selectedMessage.set(null);
    if (!this.isWideViewport()) {
      this.inspectorOpen.set(false);
    }
  }

  clearRecommendation(): void {
    const currentSession = this.recommendationSessionId;
    this.chatService.resetSessionById(currentSession).subscribe({ error: () => undefined });
    this.recommendationSessionId = this.chatService.createEphemeralSessionId();
    this.recommendationMessages.set([]);
    this.recommendationState.set(null);
    this.recommendationProfile = {};
    this.selectedMessage.set(null);
    if (!this.isWideViewport()) {
      this.inspectorOpen.set(false);
    }
  }

  toggleDebug(): void {
    this.showDebug.update((value) => !value);
    if (!this.isWideViewport()) {
      this.inspectorOpen.set(this.showDebug());
    }
  }

  toggleInspector(): void {
    this.inspectorOpen.update((value) => !value);
  }

  closeInspector(): void {
    this.inspectorOpen.set(false);
  }

  selectMessage(message: Message): void {
    if (!message.meta) {
      return;
    }
    const nextSelection = this.selectedMessage() === message ? null : message;
    this.selectedMessage.set(nextSelection);
    if (!this.isWideViewport()) {
      this.inspectorOpen.set(Boolean(nextSelection));
    }
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

  currentSessionId(): string {
    return this.chatService.getSessionId();
  }

  variantOptions(): string[] {
    return this.variantOptionsFrom(this.models());
  }

  averageLatencyMs(): number | null {
    const assistantMessages = this.messages().filter((message) => typeof message.meta?.latencyMs === 'number');
    if (assistantMessages.length === 0) {
      return null;
    }
    const total = assistantMessages.reduce((sum, message) => sum + (message.meta?.latencyMs ?? 0), 0);
    return Math.round(total / assistantMessages.length);
  }

  needsReviewCount(): number {
    return this.messages().filter((message) => message.meta?.needsHumanReview).length;
  }

  assistantTurnCount(): number {
    return this.messages().filter((message) => message.role === 'assistant').length;
  }

  latestAssistantMessage(): Message | null {
    const assistantMessages = this.messages().filter((message) => message.role === 'assistant');
    return assistantMessages.at(-1) ?? null;
  }

  formatLatency(latencyMs: number | undefined): string {
    return typeof latencyMs === 'number' ? `${Math.round(latencyMs)} ms` : 'N/A';
  }

  formatLabel(value: string): string {
    return value.replace(/_/g, ' ');
  }

  formatSourceLabel(value: string | undefined): string {
    if (!value) {
      return 'unknown';
    }
    return value.replace(/_/g, ' ');
  }

  formatScriptLabel(value: string | undefined): string {
    return value ? value.toUpperCase() : 'N/A';
  }

  toolName(message: Message): string {
    return message.meta?.toolCall?.name ?? 'none';
  }

  toolStatus(message: Message): string {
    const status = message.meta?.toolResult?.['status'];
    return typeof status === 'string' && status.trim() ? status : 'n/a';
  }

  selectedMeta(): MessageMeta | null {
    return this.selectedMessage()?.meta ?? null;
  }

  selectedSlots(): KeyValueRow[] {
    return this.objectRows(this.selectedMeta()?.slots);
  }

  selectedSessionState(): KeyValueRow[] {
    return this.objectRows(this.selectedMeta()?.sessionState);
  }

  selectedToolArguments(): KeyValueRow[] {
    return this.objectRows(this.selectedMeta()?.toolCall?.arguments);
  }

  selectedToolResultRows(): KeyValueRow[] {
    return this.objectRows(this.selectedMeta()?.toolResult);
  }

  selectedRagResults(): Array<Record<string, unknown>> {
    return Array.isArray(this.selectedMeta()?.ragResults) ? this.selectedMeta()?.ragResults ?? [] : [];
  }

  ragMetaRows(item: Record<string, unknown>): KeyValueRow[] {
    const metadata = item['metadata'];
    if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) {
      return [];
    }
    return this.objectRows(metadata as Record<string, unknown>);
  }

  ragTextPreview(item: Record<string, unknown>): string {
    const text = item['text'];
    if (typeof text !== 'string') {
      return '';
    }
    return text.length > 220 ? `${text.slice(0, 220)}...` : text;
  }

  private objectRows(value: Record<string, unknown> | null | undefined): KeyValueRow[] {
    if (!value) {
      return [];
    }
    return Object.entries(value)
      .filter(([, rawValue]) => rawValue !== null && rawValue !== undefined && rawValue !== '')
      .map(([key, rawValue]) => ({
        key,
        value: typeof rawValue === 'string' ? rawValue : JSON.stringify(rawValue),
      }));
  }

  private variantOptionsFrom(models: ModelsResponse | null): string[] {
    const variants = new Set<string>([this.selectedVariant()]);
    if (models?.default_variant) {
      variants.add(models.default_variant);
    }
    Object.entries(models?.active_variants ?? {}).forEach(([variant, path]) => {
      if (path !== null || variant === 'prod') {
        variants.add(variant);
      }
    });
    return Array.from(variants);
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

  private scheduleScroll(view: WorkspaceView): void {
    queueMicrotask(() => {
      requestAnimationFrame(() => {
        if (view === 'assistant') {
          this.scrollContainerToBottom(this.chatContainer);
          return;
        }
        this.scrollContainerToBottom(this.recoContainer);
      });
    });
  }

  private scrollContainerToBottom(container?: ElementRef<HTMLDivElement>): void {
    if (!container) {
      return;
    }
    container.nativeElement.scrollTop = container.nativeElement.scrollHeight;
  }

  private isWideViewport(): boolean {
    return typeof window === 'undefined' ? true : window.innerWidth > 1260;
  }
}
