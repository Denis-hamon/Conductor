package types

import "time"

type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream,omitempty"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

type DeltaChoice struct {
	Index        int    `json:"index"`
	Delta        Delta  `json:"delta"`
	FinishReason string `json:"finish_reason,omitempty"`
}

type Delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type SSEChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []DeltaChoice `json:"choices"`
}

type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type APIKey struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	KeyPrefix string `json:"key_prefix"`
	CreatedAt int64  `json:"created_at"`
	RateLimit int    `json:"rate_limit"`
	Revoked   bool   `json:"revoked"`
}

type CreateKeyRequest struct {
	Name      string `json:"name"`
	RateLimit int    `json:"rate_limit"`
}

type CreateKeyResponse struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	FullKey   string `json:"full_key"`
	KeyPrefix string `json:"key_prefix"`
	RateLimit int    `json:"rate_limit"`
}

type LogEntry struct {
	RequestID   string   `json:"request_id"`
	Timestamp   int64    `json:"timestamp"`
	TenantID    string   `json:"tenant_id"`
	Model       string   `json:"model"`
	TokensIn    int      `json:"tokens_in"`
	TokensOut   int      `json:"tokens_out"`
	LatencyMs   int64    `json:"latency_ms"`
	Status      int      `json:"status"`
	Reward      *float64 `json:"reward,omitempty"`
	FullTrace   string   `json:"full_trace,omitempty"`
}

type LogsResponse struct {
	Entries    []LogEntry `json:"entries"`
	Total      int        `json:"total"`
	Page       int        `json:"page"`
	PageSize   int        `json:"page_size"`
	NextCursor string     `json:"next_cursor,omitempty"`
}

type Quota struct {
	TenantID      string `json:"tenant_id"`
	MonthlyTokens int    `json:"monthly_tokens"`
	UsedTokens    int    `json:"used_tokens"`
	AlertSent80   bool   `json:"alert_sent_80"`
	AlertSent90   bool   `json:"alert_sent_90"`
}

type QuotaUsage struct {
	Used     int     `json:"used"`
	Limit    int     `json:"limit"`
	Percent  float64 `json:"percent"`
	Remaining int    `json:"remaining"`
}

var now = time.Now
