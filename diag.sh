#!/bin/bash
echo "--- buildx tree"
P=$(pgrep -f 'docker-buildx bake' | head -1)
[ -n "$P" ] && pstree -ap "$P" | head -8 || echo "tidak ada proses buildx"
echo "--- docker config"
cat ~/.docker/config.json 2>/dev/null || echo "(tidak ada)"
echo "--- buildx builders"
timeout 10 docker buildx ls 2>&1 | head -6
