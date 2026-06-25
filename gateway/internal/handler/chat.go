package handler

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/ovhcloud/conductor-fabric/gateway/internal/auth"
	"github.com/ovhcloud/conductor-fabric/gateway/internal/types"
)

var startTime = time.Now()

func newRequestID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("req_%x", b)
}

func generateResponse(req types.ChatRequest, reqID string) types.ChatResponse {
	n := time.Now().Unix()
	content := fmt.Sprintf("Echo from Conductor Fabric: %s", req.Messages[len(req.Messages)-1].Content)
	return types.ChatResponse{
		ID:      reqID,
		Object:  "chat.completion",
		Created: n,
		Model:   req.Model,
		Choices: []types.Choice{
			{
				Index:        0,
				Message:      types.Message{Role: "assistant", Content: content},
				FinishReason: "stop",
			},
		},
		Usage: &types.Usage{
			PromptTokens:     len(req.Messages[len(req.Messages)-1].Content) / 4,
			CompletionTokens: len(content) / 4,
			TotalTokens:      (len(req.Messages[len(req.Messages)-1].Content) + len(content)) / 4,
		},
	}
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg, errType string) {
	writeJSON(w, status, types.ErrorResponse{
		Error: types.ErrorDetail{Message: msg, Type: errType},
	})
}

type ChatHandler struct {
	QuotaMgr *auth.QuotaManager
}

func (h *ChatHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "Method not allowed", "invalid_request_error")
		return
	}

	var req types.ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "Invalid JSON body: "+err.Error(), "invalid_request_error")
		return
	}

	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model field is required", "invalid_request_error")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages field is required", "invalid_request_error")
		return
	}

	reqID := newRequestID()

	if req.Stream {
		handleStream(w, r, req, reqID)
		return
	}

	resp := generateResponse(req, reqID)
	totalTokens := 0
	if resp.Usage != nil {
		totalTokens = resp.Usage.TotalTokens
	}
	if h.QuotaMgr != nil {
		tenantID, _ := r.Context().Value(tenantCtxKey).(string)
		if tenantID == "" {
			tenantID = "default-tenant"
		}
		h.QuotaMgr.UseTokens(tenantID, totalTokens)
	}
	if tw, ok := w.(interface {
		SetTokensTotal(int)
		SetModelName(string)
	}); ok {
		tw.SetTokensTotal(totalTokens)
		tw.SetModelName(req.Model)
	}
	writeJSON(w, http.StatusOK, resp)
}

func handleStream(w http.ResponseWriter, r *http.Request, req types.ChatRequest, reqID string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported", "server_error")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	n := time.Now().Unix()
	content := fmt.Sprintf("Echo from Conductor Fabric: %s", req.Messages[len(req.Messages)-1].Content)

	roleChunk := types.SSEChunk{
		ID:      reqID,
		Object:  "chat.completion.chunk",
		Created: n,
		Model:   req.Model,
		Choices: []types.DeltaChoice{
			{Index: 0, Delta: types.Delta{Role: "assistant"}, FinishReason: ""},
		},
	}
	data, _ := json.Marshal(roleChunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()

	words := strings.Fields(content)
	for i, word := range words {
		select {
		case <-r.Context().Done():
			return
		default:
		}
		chunk := types.SSEChunk{
			ID:      reqID,
			Object:  "chat.completion.chunk",
			Created: n,
			Model:   req.Model,
			Choices: []types.DeltaChoice{
				{Index: 0, Delta: types.Delta{Content: word + " "}, FinishReason: ""},
			},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		if i < len(words)-1 {
			select {
			case <-r.Context().Done():
				return
			case <-time.After(30 * time.Millisecond):
			}
		}
	}

	done := types.SSEChunk{
		ID:      reqID,
		Object:  "chat.completion.chunk",
		Created: n,
		Model:   req.Model,
		Choices: []types.DeltaChoice{
			{Index: 0, Delta: types.Delta{}, FinishReason: "stop"},
		},
	}
	data, _ = json.Marshal(done)
	fmt.Fprintf(w, "data: %s\n\n", data)
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}
