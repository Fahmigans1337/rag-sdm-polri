#!/usr/bin/env bash
# ============================================================
#  RAG SDM Polri - jalankan dengan satu perintah:  ./start.sh
#  Meminta API key (tidak ditampilkan), memvalidasi, menulis .env,
#  lalu menjalankan Docker. Key TIDAK disimpan di Git.
#  Opsional non-interaktif:  API_KEY=sk-or-v1-xxx ./start.sh
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\033[1;33m%s\033[0m\n' "$*"; }
err() { printf '\033[1;31m%s\033[0m\n' "$*" >&2; }

command -v docker >/dev/null 2>&1 || { err "Docker belum terpasang / belum jalan. Pasang Docker lalu ulangi."; exit 1; }
docker compose version >/dev/null 2>&1 || { err "Perintah 'docker compose' tidak tersedia. Perbarui Docker."; exit 1; }

KEY="${API_KEY:-}"
if [ -z "$KEY" ] && [ -f .env ] && grep -Eq '^OPENAI_API_KEY=.{12,}' .env; then
  read -rp "API key sudah tersimpan di .env. Pakai yang ada? [Y/n] " ans || ans=""
  if [[ ! "$ans" =~ ^[Nn]$ ]]; then KEY="$(grep -E '^OPENAI_API_KEY=' .env | head -1 | cut -d= -f2-)"; fi
fi
if [ -z "$KEY" ]; then
  say "Tempel API key Anda (OpenRouter 'sk-or-v1-...', OpenAI 'sk-...', atau Groq 'gsk_...')."
  say "Teks tidak akan muncul di layar. Tekan Enter jika sudah. (Kosongkan = mode tanpa AI)"
  read -rsp "API key: " KEY || KEY=""; echo
fi
KEY="$(printf '%s' "$KEY" | tr -d '[:space:]')"

BASE=""; MODEL="gpt-4o-mini"; FALLBACK=""; CHECK="https://api.openai.com/v1/models"
case "$KEY" in
  sk-or-*) BASE="https://openrouter.ai/api/v1"; MODEL="openai/gpt-4o-mini"; FALLBACK="google/gemini-2.5-flash-lite"; CHECK="https://openrouter.ai/api/v1/key" ;;
  gsk_*)   BASE="https://api.groq.com/openai/v1"; MODEL="llama-3.3-70b-versatile"; CHECK="https://api.groq.com/openai/v1/models" ;;
esac

if [ -n "$KEY" ] && command -v curl >/dev/null 2>&1; then
  code="$(curl -s -o /dev/null -m 15 -w '%{http_code}' -H "Authorization: Bearer $KEY" "$CHECK" || echo 000)"
  if [ "$code" = "200" ]; then say "API key valid."
  elif [ "$code" = "401" ] || [ "$code" = "403" ]; then err "API key DITOLAK penyedia (HTTP $code). Periksa kembali key Anda."; read -rp "Tetap lanjut tanpa AI aktif? [y/N] " go || go=""; [[ "$go" =~ ^[Yy]$ ]] || exit 1
  else say "Tidak dapat memvalidasi key (HTTP $code, mungkin koneksi). Lanjut saja."; fi
fi

cat > .env <<EOF
# Dibuat oleh start.sh - JANGAN di-commit ke GitHub (key akan dicabut otomatis oleh penyedia)
LLM_PROVIDER=openai
OPENAI_API_KEY=$KEY
OPENAI_MODEL=$MODEL
OPENAI_BASE_URL=$BASE
OPENAI_FALLBACK_MODELS=$FALLBACK
EOF
chmod 600 .env 2>/dev/null || true

say "Membangun dan menjalankan container (pertama kali bisa beberapa menit)..."
docker compose up -d --build --force-recreate

say "Menunggu aplikasi siap..."
for _ in $(seq 1 90); do
  if curl -fs -m 3 http://localhost:8080/api/health 2>/dev/null | grep -q '"ready":true'; then
    say "SIAP.  Buka:  http://localhost:8080"
    [ -z "$KEY" ] && say "(Mode tanpa AI: jawaban dokumen ekstraktif. Jalankan ulang ./start.sh untuk mengisi key.)"
    exit 0
  fi
  sleep 2
done
err "Aplikasi belum siap. Lihat log:  docker compose logs --tail 50"
exit 1
