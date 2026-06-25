package handler

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"github.com/ovhcloud/conductor-fabric/gateway/internal/auth"
	"github.com/ovhcloud/conductor-fabric/gateway/internal/logging"
	"github.com/ovhcloud/conductor-fabric/gateway/internal/types"
)

type AdminHandler struct {
	keyStore  *auth.Store
	rateLimit *auth.RateLimiter
	quotaMgmt *auth.QuotaManager
	logger    *logging.Logger
}

func NewAdminHandler(ks *auth.Store, rl *auth.RateLimiter, qm *auth.QuotaManager, l *logging.Logger) *AdminHandler {
	return &AdminHandler{
		keyStore:  ks,
		rateLimit: rl,
		quotaMgmt: qm,
		logger:    l,
	}
}

func (h *AdminHandler) HandleAPIKeys(w http.ResponseWriter, r *http.Request) {
	tenantID, _ := r.Context().Value(tenantCtxKey).(string)

	switch r.Method {
	case http.MethodPost:
		h.createKey(w, r, tenantID)
	case http.MethodDelete:
		h.revokeKey(w, r)
	default:
		writeError(w, http.StatusMethodNotAllowed, "Method not allowed", "invalid_request_error")
	}
}

func (h *AdminHandler) createKey(w http.ResponseWriter, r *http.Request, tenantID string) {
	var req types.CreateKeyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "Invalid JSON body", "invalid_request_error")
		return
	}
	if req.Name == "" {
		writeError(w, http.StatusBadRequest, "name is required", "invalid_request_error")
		return
	}
	if req.RateLimit <= 0 {
		req.RateLimit = 100
	}

	key := h.keyStore.Create(req.Name, req.RateLimit)
	writeJSON(w, http.StatusCreated, key)
}

func (h *AdminHandler) revokeKey(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 5 {
		writeError(w, http.StatusBadRequest, "missing key_id", "invalid_request_error")
		return
	}
	keyID := parts[4]

	if !h.keyStore.Revoke(keyID) {
		writeError(w, http.StatusNotFound, "key not found or already revoked", "invalid_request_error")
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (h *AdminHandler) HandleLogs(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "Method not allowed", "invalid_request_error")
		return
	}

	tenantID, _ := r.Context().Value(tenantCtxKey).(string)
	fromStr := r.URL.Query().Get("from")
	toStr := r.URL.Query().Get("to")
	format := r.URL.Query().Get("format")

	var from, to int64
	if fromStr != "" {
		from, _ = strconv.ParseInt(fromStr, 10, 64)
		if from == 0 {
			writeError(w, http.StatusBadRequest, "invalid from parameter: expected unix timestamp", "invalid_request_error")
			return
		}
	}
	if toStr != "" {
		to, _ = strconv.ParseInt(toStr, 10, 64)
		if to == 0 {
			to = 1 << 62
		}
	} else {
		to = 1 << 62
	}

	if format == "csv" {
		csvData := h.logger.ExportCSV(from, to, tenantID)
		w.Header().Set("Content-Type", "text/csv")
		w.Header().Set("Content-Disposition", "attachment; filename=logs.csv")
		w.Write([]byte(csvData))
		return
	}

	page, _ := strconv.Atoi(r.URL.Query().Get("page"))
	if page < 1 {
		page = 1
	}
	pageSize, _ := strconv.Atoi(r.URL.Query().Get("page_size"))
	if pageSize < 1 || pageSize > 100 {
		pageSize = 50
	}

	resp := h.logger.Query(from, to, page, pageSize, tenantID)
	writeJSON(w, http.StatusOK, resp)
}

func (h *AdminHandler) HandleLogByID(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 5 {
		writeError(w, http.StatusBadRequest, "missing request_id", "invalid_request_error")
		return
	}
	reqID := parts[4]

	entry := h.logger.GetByID(reqID)
	if entry == nil {
		writeError(w, http.StatusNotFound, "log entry not found", "invalid_request_error")
		return
	}

	writeJSON(w, http.StatusOK, entry)
}

func (h *AdminHandler) HandleQuota(w http.ResponseWriter, r *http.Request) {
	tenantID, _ := r.Context().Value(tenantCtxKey).(string)
	if tenantID == "" {
		tenantID = "default-tenant"
	}

	usage := h.quotaMgmt.GetUsage(tenantID)
	writeJSON(w, http.StatusOK, usage)
}
