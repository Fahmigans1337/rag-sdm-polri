#!/bin/bash
echo '=== Setup Docker Auto-Start ==='

# 1. Start docker sekarang
echo '[1/3] Starting docker daemon...'
sudo service docker start

# 2. Tambah boot command ke wsl.conf (cek dulu biar ga duplikat)
if ! grep -q '\[boot\]' /etc/wsl.conf 2>/dev/null; then
    echo '[2/3] Configuring docker auto-start on WSL boot...'
    printf '\n[boot]\ncommand = service docker start\n' | sudo tee -a /etc/wsl.conf > /dev/null
    echo 'Done: Docker will auto-start next time WSL opens.'
else
    echo '[2/3] wsl.conf [boot] already configured, skipping.'
fi

# 3. Setup sudoers agar docker bisa start tanpa password (optional tapi nyaman)
echo '[3/3] Allow docker start without sudo password...'
echo '%docker ALL=(ALL) NOPASSWD: /usr/sbin/service docker start, /usr/sbin/service docker stop' | sudo tee /etc/sudoers.d/docker-service > /dev/null
sudo chmod 440 /etc/sudoers.d/docker-service

echo ''
echo '=== Done! Docker is running now. ==='
echo 'Starting app container...'
cd ~/rag-sdm-polri
docker compose up -d
echo ''
echo 'App running at: http://localhost:8080'
