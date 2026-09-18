#!/bin/bash
echo '--- /v1/models ---'
curl -s http://127.0.0.1:8088/v1/models | head -c 800; echo
echo '--- /health ---'
curl -s -o /dev/null -w 'HTTP=%{http_code}\n' http://127.0.0.1:8088/health