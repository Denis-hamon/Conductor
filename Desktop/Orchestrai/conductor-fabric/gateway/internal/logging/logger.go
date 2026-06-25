package logging

import (
	"encoding/csv"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ovhcloud/conductor-fabric/gateway/internal/types"
)

type Logger struct {
	mu     sync.RWMutex
	entries []types.LogEntry
	index  map[string]int
}

func NewLogger() *Logger {
	return &Logger{
		entries: make([]types.LogEntry, 0, 1000),
		index:   make(map[string]int),
	}
}

func (l *Logger) Log(entry types.LogEntry) {
	l.mu.Lock()
	defer l.mu.Unlock()
	entry.Timestamp = time.Now().UnixMilli()
	l.index[entry.RequestID] = len(l.entries)
	l.entries = append(l.entries, entry)
}

func (l *Logger) Query(from, to int64, page, pageSize int, tenantID string) *types.LogsResponse {
	l.mu.RLock()
	defer l.mu.RUnlock()

	var filtered []types.LogEntry
	for _, e := range l.entries {
		if e.Timestamp >= from && e.Timestamp <= to && (tenantID == "" || e.TenantID == tenantID) {
			filtered = append(filtered, e)
		}
	}

	total := len(filtered)
	start := (page - 1) * pageSize
	if start >= total {
		return &types.LogsResponse{Entries: []types.LogEntry{}, Total: total, Page: page, PageSize: pageSize}
	}

	end := start + pageSize
	if end > total {
		end = total
	}

	return &types.LogsResponse{
		Entries:  reverseCopy(filtered[start:end]),
		Total:    total,
		Page:     page,
		PageSize: pageSize,
	}
}

func (l *Logger) ExportCSV(from, to int64, tenantID string) string {
	l.mu.RLock()
	defer l.mu.RUnlock()

	var b strings.Builder
	writer := csv.NewWriter(&b)
	writer.Write([]string{"timestamp", "model", "tokens_in", "tokens_out", "latency_ms", "reward", "status"})

	for _, e := range l.entries {
		if e.Timestamp >= from && e.Timestamp <= to && (tenantID == "" || e.TenantID == tenantID) {
			reward := ""
			if e.Reward != nil {
				reward = fmt.Sprintf("%.2f", *e.Reward)
			}
			writer.Write([]string{
				strconv.FormatInt(e.Timestamp, 10),
				e.Model,
				strconv.Itoa(e.TokensIn),
				strconv.Itoa(e.TokensOut),
				strconv.FormatInt(e.LatencyMs, 10),
				reward,
				strconv.Itoa(e.Status),
			})
		}
	}

	writer.Flush()
	return b.String()
}

func (l *Logger) GetByID(requestID string) *types.LogEntry {
	l.mu.RLock()
	defer l.mu.RUnlock()
	idx, ok := l.index[requestID]
	if !ok || idx >= len(l.entries) {
		return nil
	}
	return &l.entries[idx]
}

func reverseCopy(src []types.LogEntry) []types.LogEntry {
	dst := make([]types.LogEntry, len(src))
	for i, v := range src {
		dst[len(src)-1-i] = v
	}
	return dst
}
