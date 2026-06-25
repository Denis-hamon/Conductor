package auth

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"

	"github.com/ovhcloud/conductor-fabric/gateway/internal/types"
)

type Store struct {
	mu    sync.RWMutex
	keys  map[string]*storedKey
}

type storedKey struct {
	ID        string
	Name      string
	Hash      string
	KeyPrefix string
	RateLimit int
	Revoked   bool
	CreatedAt int64
}

func NewStore() *Store {
	return &Store{keys: make(map[string]*storedKey)}
}

func generateKey() (string, string, string) {
	b := make([]byte, 32)
	rand.Read(b)
	fullKey := hex.EncodeToString(b)
	prefix := fullKey[:8]
	hash := sha256Hex(fullKey)
	return fullKey, prefix, hash
}

func sha256Hex(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

func (s *Store) Create(name string, rateLimit int) *types.CreateKeyResponse {
	fullKey, prefix, hash := generateKey()
	id := fmt.Sprintf("key_%s", prefix)

	entry := &storedKey{
		ID:        id,
		Name:      name,
		Hash:      hash,
		KeyPrefix: prefix,
		RateLimit: rateLimit,
		Revoked:   false,
		CreatedAt: time.Now().Unix(),
	}

	s.mu.Lock()
	s.keys[id] = entry
	s.mu.Unlock()

	return &types.CreateKeyResponse{
		ID:        id,
		Name:      name,
		FullKey:   fullKey,
		KeyPrefix: prefix,
		RateLimit: rateLimit,
	}
}

func (s *Store) Revoke(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	key, ok := s.keys[id]
	if !ok || key.Revoked {
		return false
	}
	key.Revoked = true
	return true
}

func (s *Store) Authenticate(fullKey string) *storedKey {
	hash := sha256Hex(fullKey)
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, key := range s.keys {
		if key.Hash == hash && !key.Revoked {
			return key
		}
	}
	return nil
}

func (s *Store) Get(id string) *types.APIKey {
	s.mu.RLock()
	defer s.mu.RUnlock()
	k, ok := s.keys[id]
	if !ok {
		return nil
	}
	return &types.APIKey{
		ID:        k.ID,
		Name:      k.Name,
		KeyPrefix: k.KeyPrefix,
		CreatedAt: k.CreatedAt,
		RateLimit: k.RateLimit,
		Revoked:   k.Revoked,
	}
}

func (s *Store) GetRateLimit(apiKey *storedKey) int {
	return apiKey.RateLimit
}
