#!/bin/bash
# ==============================================================================
# run_workers_perf_test.sh — PROXY_WORKERS 性能影响验证测试
#
# 在 k8s 上部署 proxy（使用 mock backend），对不同 PROXY_WORKERS 值进行
# 串行延迟和并发吞吐的对比测试。
#
# 前提:
#   - 148 机器上 k8s (单节点) 可用
#   - wings-control:test_new 镜像已存在
#   - Python3 + httpx + fastapi 可用
#
# 用法:
#   bash run_workers_perf_test.sh          # 完整测试
#   bash run_workers_perf_test.sh quick    # 快速测试（减少迭代）
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NS="workers-perf-zhanghui"
WORKERS_LIST="${WORKERS_LIST:-1 4 16 64 128}"
MOCK_PORT=17000
PROXY_PORT=18000
NODE_PORT=30180

# 测试参数
if [ "${1:-}" = "quick" ]; then
    ITERATIONS=50
    WARMUP=10
    CONCURRENCIES="1,10,50"
    echo "[quick mode] Reduced iterations"
else
    ITERATIONS=300
    WARMUP=30
    CONCURRENCIES="1,10,50,100"
fi

RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo " PROXY_WORKERS Performance Test — $(date)"
echo " Workers to test: $WORKERS_LIST"
echo " Iterations: $ITERATIONS, Warmup: $WARMUP"
echo " Concurrencies: $CONCURRENCIES"
echo "============================================================"

# ── 0. 安装依赖 ──────────────────────────────────────────────────
echo "[0] Checking dependencies..."
pip3 install --quiet httpx fastapi uvicorn 2>/dev/null || true

# ── 1. 启动 mock backend（本地进程） ──────────────────────────────
echo "[1] Starting mock backend on port $MOCK_PORT..."
pkill -f "mock_backend.py" 2>/dev/null || true
sleep 1
cd "$SCRIPT_DIR"
python3 mock_backend.py &
MOCK_PID=$!
sleep 2

if ! curl -s http://localhost:$MOCK_PORT/health | grep -q ok; then
    echo "ERROR: mock backend failed to start"
    kill $MOCK_PID 2>/dev/null
    exit 1
fi
echo "  Mock backend OK (PID=$MOCK_PID)"

# ── 2. 创建 k8s 命名空间 ─────────────────────────────────────────
echo "[2] Setting up k8s namespace: $NS"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -

# ── 3. 逐个测试不同 worker 数 ────────────────────────────────────
RESULT_FILES=""
for W in $WORKERS_LIST; do
    echo ""
    echo "============================================================"
    echo " Testing PROXY_WORKERS=$W"
    echo "============================================================"

    DEPLOY_NAME="proxy-w${W}-zhanghui"
    LABEL="workers-${W}"
    OUTPUT="$RESULTS_DIR/results_w${W}.json"

    # 生成 k8s deployment
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $DEPLOY_NAME
  namespace: $NS
  labels:
    app: proxy-workers-test-zhanghui
    workers: "${W}"
spec:
  replicas: 1
  selector:
    matchLabels:
      app: $DEPLOY_NAME
  template:
    metadata:
      labels:
        app: $DEPLOY_NAME
    spec:
      hostNetwork: true
      containers:
      - name: proxy
        image: docker.io/library/wings-control:test_new
        imagePullPolicy: Never
        command: ["python3", "-m", "uvicorn", "proxy.gateway:app",
                  "--host", "0.0.0.0", "--port", "${PROXY_PORT}",
                  "--log-level", "error",
                  "--workers", "${W}",
                  "--backlog", "8192"]
        env:
        - name: BACKEND_URL
          value: "http://127.0.0.1:${MOCK_PORT}"
        - name: PORT
          value: "${PROXY_PORT}"
        - name: HOST
          value: "0.0.0.0"
        - name: PROXY_WORKERS
          value: "${W}"
        - name: HEALTH_MONITOR_ENABLED
          value: "false"
        - name: RAG_ACC_ENABLED
          value: "false"
        - name: PYTHONPATH
          value: "/opt"
        - name: BACKEND_PROBE_TIMEOUT
          value: "5"
        ports:
        - containerPort: $PROXY_PORT
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "8000m"
            memory: "16Gi"
EOF

    echo "  Waiting for pod to be ready..."
    kubectl rollout status deployment/$DEPLOY_NAME -n $NS --timeout=120s 2>/dev/null || true
    sleep 5

    # 验证 proxy 可达
    RETRY=0
    while ! curl -s --connect-timeout 3 http://localhost:$PROXY_PORT/v1/models >/dev/null 2>&1; do
        RETRY=$((RETRY + 1))
        if [ $RETRY -gt 30 ]; then
            echo "  ERROR: proxy (workers=$W) failed to become ready"
            kubectl logs deployment/$DEPLOY_NAME -n $NS --tail=30
            kubectl delete deployment $DEPLOY_NAME -n $NS
            continue 2
        fi
        echo "  Waiting for proxy... ($RETRY)"
        sleep 3
    done
    echo "  Proxy (workers=$W) is ready"

    # 采集 pod 内存 (启动后)
    POD_NAME=$(kubectl get pod -n $NS -l app=$DEPLOY_NAME -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "unknown")
    MEM_BEFORE=$(kubectl exec -n $NS $POD_NAME -- sh -c 'cat /proc/meminfo | grep MemAvailable | awk "{print \$2}"' 2>/dev/null || echo "N/A")
    RSS_BEFORE=$(kubectl exec -n $NS $POD_NAME -- sh -c 'ps aux --sort=-rss | head -5' 2>/dev/null || echo "N/A")
    echo "  Pod: $POD_NAME, MemAvailable: ${MEM_BEFORE}kB"

    # 运行 benchmark
    echo "  Running benchmark..."
    cd "$SCRIPT_DIR"
    python3 bench_workers.py \
        --url http://localhost:$PROXY_PORT \
        --label "$LABEL" \
        --output "$OUTPUT" \
        -n $ITERATIONS \
        --warmup $WARMUP \
        --concurrency "$CONCURRENCIES"

    # 采集测试后内存
    RSS_AFTER=$(kubectl exec -n $NS $POD_NAME -- sh -c 'ps aux --sort=-rss | head -5' 2>/dev/null || echo "N/A")
    echo "  RSS after test:"
    echo "$RSS_AFTER"

    RESULT_FILES="$RESULT_FILES $OUTPUT"

    # 清理当前 deployment
    echo "  Cleaning up deployment $DEPLOY_NAME..."
    kubectl delete deployment $DEPLOY_NAME -n $NS --wait=true
    sleep 3
done

# ── 4. 生成对比报告 ───────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Generating comparison report"
echo "============================================================"

cd "$SCRIPT_DIR"
python3 gen_workers_report.py $RESULT_FILES -o "$RESULTS_DIR/report_workers.md"

# ── 5. 清理 ──────────────────────────────────────────────────────
echo "Cleaning up mock backend..."
kill $MOCK_PID 2>/dev/null || true

echo ""
echo "============================================================"
echo " Test Complete!"
echo " Results dir: $RESULTS_DIR/"
echo " Report:      $RESULTS_DIR/report_workers.md"
echo "============================================================"
