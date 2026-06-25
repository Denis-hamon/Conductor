package handler

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/ovhcloud/conductor-fabric/gateway/internal/auth"
	"github.com/ovhcloud/conductor-fabric/gateway/internal/logging"
	"github.com/ovhcloud/conductor-fabric/gateway/internal/types"
)

type contextKey string

const tenantCtxKey contextKey = "tenant_id"

func Middleware(keyStore *auth.Store, rateLimiter *auth.RateLimiter, quotaMgr *auth.QuotaManager, logger *logging.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			tenantID := "default-tenant"
			apiKey := extractBearer(r)
			if apiKey == "" {
				writeError(w, http.StatusUnauthorized, "missing API key", "authentication_error")
				return
			}

			stored := keyStore.Authenticate(apiKey)
			if stored == nil {
				writeError(w, http.StatusUnauthorized, "invalid API key", "authentication_error")
				return
			}

			limit := keyStore.GetRateLimit(stored)
			allowed, retryAfter := rateLimiter.Allow(apiKey, limit)
			if !allowed {
				w.Header().Set("Retry-After", fmt.Sprintf("%d", retryAfter))
				writeError(w, http.StatusTooManyRequests, "rate limit exceeded", "rate_limit_error")
				return
			}

			if quotaMgr.IsExceeded(tenantID) {
				writeError(w, http.StatusPaymentRequired, "Monthly quota exceeded", "quota_error")
				return
			}

			ctx := context.WithValue(r.Context(), tenantCtxKey, tenantID)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

func extractBearer(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if !strings.HasPrefix(auth, "Bearer ") {
		return ""
	}
	return strings.TrimPrefix(auth, "Bearer ")
}

type loggingResponseWriter struct {
	http.ResponseWriter
	statusCode  int
	tokensTotal int
	modelName   string
}

func (lrw *loggingResponseWriter) WriteHeader(code int) {
	lrw.statusCode = code
	lrw.ResponseWriter.WriteHeader(code)
}

func (lrw *loggingResponseWriter) SetTokensTotal(n int) {
	lrw.tokensTotal = n
}

func (lrw *loggingResponseWriter) SetModelName(name string) {
	lrw.modelName = name
}

func LoggingMiddleware(logger *logging.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			lrw := &loggingResponseWriter{ResponseWriter: w, statusCode: http.StatusOK}
			next.ServeHTTP(lrw, r)
			latencyMs := time.Since(start).Milliseconds()

			if strings.HasPrefix(r.URL.Path, "/v1/chat/completions") {
				tenantID, _ := r.Context().Value(tenantCtxKey).(string)
				logger.Log(types.LogEntry{
					RequestID: newRequestID(),
					TenantID:  tenantID,
					Model:     lrw.modelName,
					TokensIn:  lrw.tokensTotal,
					Status:    lrw.statusCode,
					LatencyMs: latencyMs,
				})
			}
		})
	}
}

func QuotaMiddleware(quotaMgr *auth.QuotaManager) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			tenantID, _ := r.Context().Value(tenantCtxKey).(string)
			if tenantID == "" {
				tenantID = "default-tenant"
			}
			usage := quotaMgr.GetUsage(tenantID)
			w.Header().Set("X-Quota-Remaining", fmt.Sprintf("%d", usage.Remaining))
			next.ServeHTTP(w, r)
		})
	}
}
