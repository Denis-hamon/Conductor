package auth

import (
	"sync"

	"github.com/ovhcloud/conductor-fabric/gateway/internal/types"
)

type QuotaManager struct {
	mu      sync.RWMutex
	quotas  map[string]*types.Quota
	webhook func(tenantID string, percent float64)
}

func NewQuotaManager(webhookFn func(tenantID string, percent float64)) *QuotaManager {
	return &QuotaManager{
		quotas:  make(map[string]*types.Quota),
		webhook: webhookFn,
	}
}

func (qm *QuotaManager) GetOrCreate(tenantID string, monthlyTokens int) *types.Quota {
	qm.mu.Lock()
	defer qm.mu.Unlock()
	q, ok := qm.quotas[tenantID]
	if !ok {
		q = &types.Quota{
			TenantID:      tenantID,
			MonthlyTokens: monthlyTokens,
		}
		qm.quotas[tenantID] = q
	}
	return q
}

func (qm *QuotaManager) UseTokens(tenantID string, tokens int) *types.QuotaUsage {
	qm.mu.Lock()
	defer qm.mu.Unlock()

	q, ok := qm.quotas[tenantID]
	if !ok {
		q = &types.Quota{
			TenantID:      tenantID,
			MonthlyTokens: 10_000_000,
		}
		qm.quotas[tenantID] = q
	}

	q.UsedTokens += tokens
	usage := &types.QuotaUsage{
		Used:     q.UsedTokens,
		Limit:    q.MonthlyTokens,
		Remaining: q.MonthlyTokens - q.UsedTokens,
	}
	if q.MonthlyTokens > 0 {
		usage.Percent = float64(q.UsedTokens) / float64(q.MonthlyTokens) * 100
	}

	percent := usage.Percent
	if percent >= 80 && !q.AlertSent80 {
		q.AlertSent80 = true
		if qm.webhook != nil {
			qm.webhook(tenantID, percent)
		}
	}
	if percent >= 90 && !q.AlertSent90 {
		q.AlertSent90 = true
		if qm.webhook != nil {
			qm.webhook(tenantID, percent)
		}
	}

	return usage
}

func (qm *QuotaManager) IsExceeded(tenantID string) bool {
	qm.mu.RLock()
	defer qm.mu.RUnlock()
	q, ok := qm.quotas[tenantID]
	if !ok {
		return false
	}
	return q.UsedTokens >= q.MonthlyTokens
}

func (qm *QuotaManager) GetUsage(tenantID string) *types.QuotaUsage {
	qm.mu.RLock()
	defer qm.mu.RUnlock()

	q, ok := qm.quotas[tenantID]
	if !ok {
		return &types.QuotaUsage{Used: 0, Limit: 10_000_000, Percent: 0, Remaining: 10_000_000}
	}

	usage := &types.QuotaUsage{
		Used:      q.UsedTokens,
		Limit:     q.MonthlyTokens,
		Remaining: q.MonthlyTokens - q.UsedTokens,
	}
	if q.MonthlyTokens > 0 {
		usage.Percent = float64(q.UsedTokens) / float64(q.MonthlyTokens) * 100
	}
	return usage
}
