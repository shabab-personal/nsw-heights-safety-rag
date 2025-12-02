import { Component } from '@angular/core';
import { RagService, AskResponse } from '../../services/rag.service';

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent {

  question = '';
  response: AskResponse | null = null;
  loading = false;

  constructor(private rag: RagService) {}

  askQuestion() {
    if (!this.question.trim()) return;

    this.loading = true;

    this.rag.ask({ question: this.question, top_k: 4 })
      .subscribe({
        next: (res) => {
          this.response = res;
          this.loading = false;
        },
        error: (err) => {
          console.error(err);
          this.loading = false;
        }
      });
  }
}
