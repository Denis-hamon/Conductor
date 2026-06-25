package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ovhcloud/conductor-fabric/gateway/internal/auth"
	"github.com/ovhcloud/conductor-fabric/gateway/internal/handler"
	"github.com/ovhcloud/conductor-fabric/gateway/internal/logging"
)

func main() {
	keyStore := auth.NewStore()
	rateLimiter := auth.NewRateLimiter()
	quotaMgr := auth.NewQuotaManager(func(tenantID string, percent float64) {
		log.Printf("ALERT: tenant %s reached %.0f%% of monthly quota", tenantID, percent)
	})
	logger := logging.NewLogger()

	chatHandler := &handler.ChatHandler{QuotaMgr: quotaMgr}
	adminHandler := handler.NewAdminHandler(keyStore, rateLimiter, quotaMgr, logger)

	mux := http.NewServeMux()

	mux.HandleFunc("/healthz", handler.HandleHealthz)
	mux.HandleFunc("/readyz", handler.HandleReadyz)

	protected := handler.Middleware(keyStore, rateLimiter, quotaMgr, logger)
	logged := handler.LoggingMiddleware(logger)
	quota := handler.QuotaMiddleware(quotaMgr)

	mux.Handle("/v1/chat/completions", protected(logged(quota(chatHandler))))

	mux.Handle("/v1/admin/api-keys", protected(http.HandlerFunc(adminHandler.HandleAPIKeys)))
	mux.Handle("/v1/admin/api-keys/", protected(http.HandlerFunc(adminHandler.HandleAPIKeys)))
	mux.Handle("/v1/admin/logs", protected(http.HandlerFunc(adminHandler.HandleLogs)))
	mux.Handle("/v1/admin/logs/", protected(http.HandlerFunc(adminHandler.HandleLogByID)))
	mux.Handle("/v1/admin/quota", protected(http.HandlerFunc(adminHandler.HandleQuota)))

	addr := ":8080"
	srv := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)

	go func() {
		fmt.Printf("Conductor Fabric Gateway listening on %s\n", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	<-done
	fmt.Println("\nshutting down...")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("shutdown error: %v", err)
	}
	fmt.Println("shutdown complete")
}
