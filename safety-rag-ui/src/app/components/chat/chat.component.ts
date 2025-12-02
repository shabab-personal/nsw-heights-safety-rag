import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RagService, AskResponse } from '../../services/rag.service';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent {

  question = '';
  response: AskResponse | null = null;
  loading = false;
  errorMessage = '';

  constructor(private rag: RagService) {}

  askQuestion() {
    this.errorMessage = '';

    const trimmed = this.question.trim();
    if (!trimmed) return;

    this.loading = true;

    this.rag.ask({ question: trimmed, top_k: 4 }).subscribe({
      next: (res) => {
        this.response = res;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error calling RAG API:', err);
        this.errorMessage = 'Something went wrong talking to the backend.';
        this.loading = false;
      }
    });
  }
}
