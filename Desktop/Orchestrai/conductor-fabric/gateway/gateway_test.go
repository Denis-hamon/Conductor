package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOpenAICompatibleEndpoint(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		wantStatus int
		wantShape  string
	}{
		{
			name:       "valid chat completion request",
			body:       `{"model": "conductor-fabric", "messages": [{"role": "user", "content": "Hello"}]}`,
			wantStatus: http.StatusOK,
			wantShape:  "chat.completion",
		},
		{
			name:       "streaming request",
			body:       `{"model": "conductor-fabric", "messages": [{"role": "user", "content": "Hi"}], "stream": true}`,
			wantStatus: http.StatusOK,
			wantShape:  "data: [DONE]",
		},
		{
			name:       "invalid model name",
			body:       `{"model": "invalid-model", "messages": [{"role": "user", "content": "Hello"}]}`,
			wantStatus: http.StatusBadRequest,
			wantShape:  "error",
		},
		{
			name:       "missing messages field",
			body:       `{"model": "conductor-fabric"}`,
			wantStatus: http.StatusBadRequest,
			wantShape:  "error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(tt.body))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			// handler.HandleChatCompletions(w, req)

			resp := w.Result()
			if resp.StatusCode != tt.wantStatus {
				t.Errorf("got status %d, want %d", resp.StatusCode, tt.wantStatus)
			}

			var body map[string]interface{}
			json.NewDecoder(resp.Body).Decode(&body)
			resp.Body.Close()

			if tt.wantShape == "chat.completion" {
				if body["object"] != "chat.completion" {
					t.Errorf("response object = %v, want chat.completion", body["object"])
				}
				if body["id"] == nil || !strings.HasPrefix(body["id"].(string), "chatcmpl-") {
					t.Errorf("response id = %v, want chatcmpl- prefix", body["id"])
				}
			}
		})
	}
}

func TestAPIKeyAuthentication(t *testing.T) {
	tests := []struct {
		name       string
		authHeader string
		wantStatus int
	}{
		{
			name:       "valid API key",
			authHeader: "Bearer ovh-test-key-valid",
			wantStatus: http.StatusOK,
		},
		{
			name:       "missing auth header",
			authHeader: "",
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "malformed API key",
			authHeader: "Bearer ",
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "revoked API key",
			authHeader: "Bearer ovh-test-key-revoked",
			wantStatus: http.StatusUnauthorized,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
				bytes.NewBufferString(`{"model": "conductor-fabric", "messages": [{"role": "user", "content": "test"}]}`))
			req.Header.Set("Content-Type", "application/json")
			if tt.authHeader != "" {
				req.Header.Set("Authorization", tt.authHeader)
			}

			w := httptest.NewRecorder()
			// handler.HandleChatCompletions(w, req)

			resp := w.Result()
			if resp.StatusCode != tt.wantStatus {
				t.Errorf("got status %d, want %d", resp.StatusCode, tt.wantStatus)
			}
			resp.Body.Close()
		})
	}
}

func TestRateLimiting(t *testing.T) {
	tests := []struct {
		name          string
		requests      int
		wantFinalCode int
	}{
		{
			name:          "within rate limit returns 200",
			requests:      5,
			wantFinalCode: http.StatusOK,
		},
		{
			name:          "exceeded rate limit returns 429",
			requests:      150,
			wantFinalCode: http.StatusTooManyRequests,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for i := 0; i < tt.requests; i++ {
				req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
					bytes.NewBufferString(`{"model": "conductor-fabric", "messages": [{"role": "user", "content": "test"}]}`))
				req.Header.Set("Content-Type", "application/json")
				req.Header.Set("Authorization", "Bearer ovh-test-key-valid")

				w := httptest.NewRecorder()
				// handler.HandleChatCompletions(w, req)

				resp := w.Result()
				if i == tt.requests-1 && resp.StatusCode != tt.wantFinalCode {
					t.Errorf("final request: got %d, want %d", resp.StatusCode, tt.wantFinalCode)
				}
				resp.Body.Close()
			}
		})
	}
}

func TestStreamingSSEFormat(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model": "conductor-fabric", "messages": [{"role": "user", "content": "Count to 3"}], "stream": true}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer ovh-test-key-valid")

	w := httptest.NewRecorder()
	// handler.HandleChatCompletions(w, req)

	resp := w.Result()
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("got status %d, want 200", resp.StatusCode)
	}

	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	body := w.Body.String()
	if !strings.Contains(body, "data: [DONE]") {
		t.Error("streaming response missing [DONE] terminator")
	}

	chunks := strings.Split(body, "\n\n")
	for _, chunk := range chunks {
		chunk = strings.TrimSpace(chunk)
		if chunk == "" || chunk == "data: [DONE]" {
			continue
		}
		if !strings.HasPrefix(chunk, "data: ") {
			t.Errorf("invalid SSE format: %q (missing 'data: ' prefix)", chunk)
		}
	}
}

func TestErrorResponseShape(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		wantStatus int
		wantType   string
	}{
		{
			name:       "invalid JSON body",
			body:       `not json`,
			wantStatus: http.StatusBadRequest,
			wantType:   "invalid_request_error",
		},
		{
			name:       "unauthorized",
			body:       `{}`,
			wantStatus: http.StatusUnauthorized,
			wantType:   "authentication_error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(tt.body))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			// handler.HandleChatCompletions(w, req)

			resp := w.Result()
			if resp.StatusCode != tt.wantStatus {
				t.Errorf("got status %d, want %d", resp.StatusCode, tt.wantStatus)
			}

			var errResp struct {
				Error struct {
					Message string `json:"message"`
					Type    string `json:"type"`
				} `json:"error"`
			}
			json.NewDecoder(resp.Body).Decode(&errResp)
			resp.Body.Close()

			if errResp.Error.Type != tt.wantType {
				t.Errorf("error type = %q, want %q", errResp.Error.Type, tt.wantType)
			}
			if errResp.Error.Message == "" {
				t.Error("error message is empty")
			}
		})
	}
}
