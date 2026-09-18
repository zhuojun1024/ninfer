#!/usr/bin/env bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
B=/home/zhuojun/ninfer/build
cuobjdump --dump-sass $B/apps/ninfer-serve 2>/dev/null > /tmp/serve_sass.txt
cuobjdump --dump-sass $B/tests/ninfer_qwen3_5_tp2_forward_test 2>/dev/null > /tmp/test_sass.txt
echo "serve sass bytes: $(wc -c < /tmp/serve_sass.txt)"
echo "test sass bytes: $(wc -c < /tmp/test_sass.txt)"
echo "SERVE total SASS funcs: $(grep -cE 'Function :' /tmp/serve_sass.txt)"
echo "SERVE set_i32_scalar_kernel hits: $(grep -c 'set_i32_scalar_kernel' /tmp/serve_sass.txt)"
echo "SERVE set_i32 lines:"
grep 'set_i32_scalar_kernel' /tmp/serve_sass.txt | head -5
echo ""
echo "TEST total SASS funcs: $(grep -cE 'Function :' /tmp/test_sass.txt)"
echo "TEST set_i32_scalar_kernel hits: $(grep -c 'set_i32_scalar_kernel' /tmp/test_sass.txt)"
echo "TEST set_i32 lines:"
grep 'set_i32_scalar_kernel' /tmp/test_sass.txt | head -5
