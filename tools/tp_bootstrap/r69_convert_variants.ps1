# Regenerate the three remaining Qwen3.8-27B variants with the promoted official recipe (the
# r62-r68 DFlash2 draft Q4 set now lives in official_recipes.py::_optional).
#
#   pwsh -File tools/tp_bootstrap/r69_convert_variants.ps1
#
# The DFlash2 draft is not part of any of these exports; they all use the shared DFlash2-FP8 source
# that the w4a4+w8a8 family already validated. Sequential: each conversion is CPU and disk heavy.
$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'

$draft = 'D:/LLM/W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8'

$variants = @(
    @{ Name = 'qwen3.8-27b-w4a4';       Dir = 'D:/LLM/W4A16/NVFP4/W4A4';                   Out = 'D:/LLM/qwen3_8_27b_w4a4_dflash2_final.ninfer' },
    @{ Name = 'qwen3.8-27b-swift1.5';   Dir = 'D:/LLM/Swift-1.5-Qwen3.8-27b-NVFP4';        Out = 'D:/LLM/qwen3_8_27b_swift15_dflash2_final.ninfer' },
    @{ Name = 'qwen3.8-27b-thinkingcap'; Dir = 'D:/LLM/ThinkingCap-Qwen3.8-27B-NVFP4A4-AWQ'; Out = 'D:/LLM/qwen3_8_27b_thinkingcap_dflash2_final.ninfer' }
)

foreach ($v in $variants) {
    if (Test-Path $v.Out) { throw ("output already exists: " + $v.Out) }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    python -m tools.convert `
      --model $v.Dir `
      --recipe 'D:/LLM/w4a4_family_recipe.py' `
      --source ("dflash2=" + $draft) `
      --source ("quantized=" + $v.Dir) `
      --components text,vision,mtp,dflash2 `
      --proposal `
      --name $v.Name `
      --device cpu `
      --out $v.Out
    $code = $LASTEXITCODE
    Write-Host ("VARIANT {0} exit={1} elapsed_s={2}" -f $v.Name, $code, [math]::Round($sw.Elapsed.TotalSeconds, 1))
    if ($code -ne 0) { Write-Host ("stopping after failed variant " + $v.Name); exit $code }
}
Write-Host 'R69_VARIANTS_DONE'
