#!/bin/bash
# Sample clocks/power while the sweep is driving decode, then report the maxima per card.
timeout 40 nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu --format=csv,noheader -lms 500 > /tmp/clk.csv 2>/dev/null
awk -F, '{gsub(/ /,"",$1); gsub(/ MHz/,"",$2); gsub(/ MHz/,"",$3); gsub(/ W/,"",$4); gsub(/ %/,"",$5); gsub(/ C/,"",$6);
  i=$1+0; if ($2+0>sm[i]) sm[i]=$2+0; if ($3+0>mem[i]) mem[i]=$3+0; if ($4+0>pw[i]) pw[i]=$4+0; if ($5+0>ut[i]) ut[i]=$5+0; if ($6+0>tp[i]) tp[i]=$6+0}
END { for (i=0;i<=2;i++) printf "gpu%d max sm=%d MHz mem=%d MHz power=%.1f W util=%d%% temp=%d C\n", i, sm[i], mem[i], pw[i], ut[i], tp[i] }' /tmp/clk.csv
echo "--- throttle reasons (gpu0/gpu2) ---"
nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep -A3 'Clocks Event Reasons' | head -20
