#!/bin/bash
nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
echo '--- topology ---'
nvidia-smi topo -m 2>/dev/null | head -12
echo '--- per-device PCIe staging bandwidth (if the bench supports it) ---'
cat /mnt/d/Documents/workbench/ninfer/tools/tp_bootstrap/run_bench_pcie_stage.sh
