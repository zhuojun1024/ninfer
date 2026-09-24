# Convert the W4A4 family's final section 8 draft-q4 artifact.
#
# Same merged override as the W4A4+W8A8 piece (r66_draft_q4_all.py), but on the plain W4A4 base:
# the two families differ only in the text/vision/MTP sources, and both take the DFlash2 draft from
# the shared DFlash2-FP8 checkpoint (that is what qwen3_8_27b_w4a4_dflash2_draftall.ninfer used).
#
#   pwsh -File tools/tp_bootstrap/r66_convert_q4all_w4a4.ps1
$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'
$out = 'D:/LLM/qwen3_8_27b_w4a4_dflash2_q4all.ninfer'
if (Test-Path $out) { throw "output already exists: $out" }
$sw = [System.Diagnostics.Stopwatch]::StartNew()
python -m tools.convert `
  --model 'D:/LLM/W4A16/NVFP4/W4A4' `
  --recipe 'D:/LLM/w4a4_family_recipe.py' `
  --source 'dflash2=D:/LLM/W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8' `
  --source 'quantized=D:/LLM/W4A16/NVFP4/W4A4' `
  --components text,vision,mtp,dflash2 `
  --proposal `
  --override 'tools/tp_bootstrap/r66_draft_q4_all.py' `
  --name 'qwen3.8-27b-w4a4-q4all' `
  --device cpu `
  --out $out
$code = $LASTEXITCODE
"CONVERT_EXIT=$code elapsed_s=$([math]::Round($sw.Elapsed.TotalSeconds,1))"
exit $code
