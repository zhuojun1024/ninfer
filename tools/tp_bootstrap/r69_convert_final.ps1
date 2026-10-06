$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'
$out = 'D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_final.ninfer'
if (Test-Path $out) { throw "output already exists: $out" }
$sw = [System.Diagnostics.Stopwatch]::StartNew()
# No --override: the draft Q4 set (r62-r66) plus the fused-QKV lever (r67) live in the official
# recipe's _optional, which this recipe imports. The codebook lever (r68) was reverted from the
# official recipe, so reproducing that artifact now needs
# --override tools/tp_bootstrap/r68_draft_codebook_q4.py.
python -m tools.convert `
  --model 'D:/LLM/W4A16/NVFP4/W4A4+W8A8' `
  --recipe 'D:/LLM/w4a4_family_recipe.py' `
  --source 'dflash2=D:/LLM/W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8' `
  --source 'quantized=D:/LLM/W4A16/NVFP4/W4A4+W8A8' `
  --components text,vision,mtp,dflash2 `
  --proposal `
  --name 'qwen3.8-27b-w4a4-w8a8' `
  --device cpu `
  --out $out
$code = $LASTEXITCODE
"CONVERT_EXIT=$code elapsed_s=$([math]::Round($sw.Elapsed.TotalSeconds,1))"
exit $code
