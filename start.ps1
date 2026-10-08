# ============================================================
#  RAG SDM Polri - jalankan di Windows PowerShell:  .\start.ps1
#  Meminta API key (tidak ditampilkan), memvalidasi, menulis .env,
#  lalu menjalankan Docker. Key TIDAK disimpan di Git.
#  Opsional: $env:API_KEY="sk-or-v1-xxx"; .\start.ps1
# ============================================================
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Say($m) { Write-Host $m -ForegroundColor Yellow }
function Err($m) { Write-Host $m -ForegroundColor Red }

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Err "Docker belum terpasang / belum jalan. Pasang Docker Desktop lalu ulangi."; exit 1 }

$key = $env:API_KEY
if (-not $key -and (Test-Path .env)) {
    $line = Select-String -Path .env -Pattern '^OPENAI_API_KEY=(.{12,})$' | Select-Object -First 1
    if ($line) {
        $ans = Read-Host "API key sudah tersimpan di .env. Pakai yang ada? [Y/n]"
        if ($ans -notmatch '^[Nn]') { $key = $line.Matches[0].Groups[1].Value }
    }
}
if (-not $key) {
    Say "Tempel API key Anda (OpenRouter 'sk-or-v1-...', OpenAI 'sk-...', atau Groq 'gsk_...')."
    Say "Teks tidak akan muncul di layar. Tekan Enter jika sudah. (Kosongkan = mode tanpa AI)"
    $sec = Read-Host "API key" -AsSecureString
    $key = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
}
$key = ($key -replace '\s', '')

$base = ""; $model = "gpt-4o-mini"; $fallback = ""; $check = "https://api.openai.com/v1/models"
if ($key -like "sk-or-*") { $base = "https://openrouter.ai/api/v1"; $model = "openai/gpt-4o-mini"; $fallback = "google/gemini-2.5-flash-lite"; $check = "https://openrouter.ai/api/v1/key" }
elseif ($key -like "gsk_*") { $base = "https://api.groq.com/openai/v1"; $model = "llama-3.3-70b-versatile"; $check = "https://api.groq.com/openai/v1/models" }

if ($key) {
    try {
        Invoke-RestMethod -Uri $check -Headers @{ Authorization = "Bearer $key" } -TimeoutSec 15 | Out-Null
        Say "API key valid."
    } catch {
        $code = 0; if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
        if ($code -eq 401 -or $code -eq 403) {
            Err "API key DITOLAK penyedia (HTTP $code). Periksa kembali key Anda."
            $go = Read-Host "Tetap lanjut tanpa AI aktif? [y/N]"
            if ($go -notmatch '^[Yy]') { exit 1 }
        } else { Say "Tidak dapat memvalidasi key (mungkin koneksi). Lanjut saja." }
    }
}

$envText = @"
# Dibuat oleh start.ps1 - JANGAN di-commit ke GitHub (key akan dicabut otomatis oleh penyedia)
LLM_PROVIDER=openai
OPENAI_API_KEY=$key
OPENAI_MODEL=$model
OPENAI_BASE_URL=$base
OPENAI_FALLBACK_MODELS=$fallback
"@
[IO.File]::WriteAllText((Join-Path $PSScriptRoot ".env"), $envText, (New-Object Text.UTF8Encoding($false)))

Say "Membangun dan menjalankan container (pertama kali bisa beberapa menit)..."
docker compose up -d --build --force-recreate
if ($LASTEXITCODE -ne 0) { Err "docker compose gagal. Pastikan Docker sedang berjalan."; exit 1 }

Say "Menunggu aplikasi siap..."
for ($i = 0; $i -lt 90; $i++) {
    try {
        $h = Invoke-RestMethod -Uri http://localhost:8080/api/health -TimeoutSec 3
        if ($h.status.ready) {
            Say "SIAP.  Buka:  http://localhost:8080"
            if (-not $key) { Say "(Mode tanpa AI: jawaban dokumen ekstraktif. Jalankan ulang .\start.ps1 untuk mengisi key.)" }
            exit 0
        }
    } catch { }
    Start-Sleep -Seconds 2
}
Err "Aplikasi belum siap. Lihat log:  docker compose logs --tail 50"
exit 1
