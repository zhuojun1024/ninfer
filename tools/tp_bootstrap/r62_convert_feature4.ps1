$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'
$out = 'D:/LLM/qwen3_8_27b_w4a4_w8a8_dflash2_featproj4.ninfer'
if (Test-Path $out) { throw "output already exists: $out" }
$sw = [System.Diagnostics.Stopwatch]::StartNew()
python -m tools.convert `
  --model 'D:/LLM/W4A16/NVFP4/W4A4+W8A8' `
  --recipe 'D:/LLM/w4a4_family_recipe.py' `
  --source 'dflash2=D:/LLM/W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8' `
  --source 'quantized=D:/LLM/W4A16/NVFP4/W4A4+W8A8' `
  --components text,vision,mtp,dflash2 `
  --proposal `
  --override 'tools/tp_bootstrap/r62_draft_feature_q4.py' `
  --name 'qwen3.8-27b-w4a4-w8a8-featproj4' `
  --device cpu `
  --out $out
$code = $LASTEXITCODE
"CONVERT_EXIT=$code elapsed_s=$([math]::Round($sw.Elapsed.TotalSeconds,1))"
exit $code
