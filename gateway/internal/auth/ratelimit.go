package auth

import (
	"sync"
	"time"
)

type RateLimiter struct {
	mu    sync.Mutex
	slots map[string]*window
}

type window struct {
	count    int
	limit    int
	resetAt  int64
	duration int64
}

func NewRateLimiter() *RateLimiter {
	return &RateLimiter{slots: make(map[string]*window)}
}

func (rl *RateLimiter) Allow(key string, limit int) (bool, int64) {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now().Unix()
	duration := int64(60)

	w, ok := rl.slots[key]
	if !ok || now >= w.resetAt {
		rl.slots[key] = &window{
			count:    1,
			limit:    limit,
			resetAt:  now + duration,
			duration: duration,
		}
		return true, 0
	}

	if w.count >= w.limit {
		retryAfter := w.resetAt - now
		return false, retryAfter
	}

	w.count++
	return true, 0
}
