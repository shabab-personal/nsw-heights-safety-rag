import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface AskRequest {
  question: string;
  top_k?: number;
}

export interface RetrievedChunk {
  id: string;
  text_preview: string;
  metadata: {
    doc_id: string;
    page_num: number;
    chunk_idx: number;
  };
}

export interface AskResponse {
  answer: string;
  chunks: RetrievedChunk[];
}

@Injectable({
  providedIn: 'root'
})
export class RagService {

  private API_URL = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  ask(req: AskRequest): Observable<AskResponse> {
    return this.http.post<AskResponse>(`${this.API_URL}/ask`, req);
  }
}
