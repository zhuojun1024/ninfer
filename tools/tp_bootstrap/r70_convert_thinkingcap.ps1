$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'
# The ThinkingCap export ships a tokenizer serialized by a different tokenizers version and a
# tokenizer_config.json without the mandatory added_tokens_decoder field. vocab.json and merges.txt
# are byte-identical to the canonical Qwen3.8 export and its 33 added_tokens_decoder entries match
# this checkpoint's tokenizer.json added_tokens exactly, so the canonical tokenizer.json is
# substituted and only the missing decoder is merged into the checkpoint's own tokenizer_config.
$src = 'D:/LLM/ThinkingCap-Qwen3.8-27B-NVFP4A4-AWQ'
$canon = 'D:/LLM/W4A16/NVFP4/W4A4+W8A8'
$out = 'D:/LLM/qwen3_8_27b_thinkingcap_dflash2_final.ninfer'
$stage = 'build-win/r70'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
if (Test-Path ($out + '.conversion.json')) {
    Copy-Item ($out + '.conversion.json') "$stage/thinkingcap-before-tokenizer-fix.conversion.json" -Force
}
Remove-Item $out, ($out + '.conversion.json') -Force -ErrorAction SilentlyContinue

python -c @"
import json, pathlib, sys
src, canon = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
config = json.loads((src / 'tokenizer_config.json').read_text(encoding='utf-8'))
config['added_tokens_decoder'] = json.loads((canon / 'tokenizer_config.json').read_text(encoding='utf-8'))['added_tokens_decoder']
out = pathlib.Path(sys.argv[3])
out.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
print('merged added_tokens_decoder entries:', len(config['added_tokens_decoder']))
"@ $src $canon "$stage/tokenizer_config.json"

$sw = [System.Diagnostics.Stopwatch]::StartNew()
python -m tools.convert `
  --model $src `
  --recipe 'D:/LLM/w4a4_family_recipe.py' `
  --source 'dflash2=D:/LLM/W4A16/NVFP4/W4A4+W8A8/DFlash2-FP8' `
  --source ("quantized=" + $src) `
  --components text,vision,mtp,dflash2 `
  --resource ("tokenizer.json=" + $canon + '/tokenizer.json') `
  --resource ("tokenizer_config.json=" + $stage + '/tokenizer_config.json') `
  --proposal `
  --name 'qwen3.8-27b-thinkingcap' `
  --device cpu `
  --out $out
Write-Output ('CONVERT_EXIT=' + $LASTEXITCODE + ' elapsed_s=' + [math]::Round($sw.Elapsed.TotalSeconds, 1))
exit $LASTEXITCODE
