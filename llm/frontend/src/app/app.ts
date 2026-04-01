import { AfterViewChecked, Component, ElementRef, ViewChild, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { ChatService, ChatResponse, HealthResponse, ModelsResponse } from './chat.service';

type Role = 'user' | 'assistant' | 'system';
type ModelVariant = 'prod' | 'dpo';
type RuntimeMode = 'collect_execute' | 'speak' | 'autonomous';

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

  messages = signal<Message[]>([]);
  inputText = '';
  isLoading = signal(false);
  backendOnline = signal(false);
  health = signal<HealthResponse | null>(null);
  models = signal<ModelsResponse | null>(null);
  showDebug = signal(false);
  selectedVariant = signal<ModelVariant>('prod');
  selectedRuntimeMode = signal<RuntimeMode>('autonomous');
  selectedMessage = signal<Message | null>(null);

  suggestions = [
    'قداش سوم البلارات البروقرسيف 1.67؟',
    'وقتاش يفتح المحل في صفاقس؟',
    'نحب نتبع الكوموند ORD-ABC12345',
    'نحب رنديفو في تونس نهار 2026-03-20 على 10:00 ورقمي +216 22 333 444',
  ];

  constructor(private readonly chatService: ChatService) {
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

  clearChat(): void {
    this.chatService.resetSession().subscribe({ error: () => undefined });
    this.messages.set([]);
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

  formatJson(value: unknown): string {
    return JSON.stringify(value, null, 2);
  }

  private scrollToBottom(): void {
    if (!this.chatContainer) {
      return;
    }
    this.chatContainer.nativeElement.scrollTop = this.chatContainer.nativeElement.scrollHeight;
  }
}
