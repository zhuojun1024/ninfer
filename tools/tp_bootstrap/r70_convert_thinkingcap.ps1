$ErrorActionPreference = 'Stop'
Set-Location 'D:\Documents\workbench\ninfer'
# The ThinkingCap export ships a tokenizer serialized by a different tokenizers version. Three engine
# frontend requirements fail on it:
#   tokenizer.json        Split pre-tokenizer uses [\p{L}]+ with trim_offsets=true (unsupported)
#   tokenizer_config.json lacks added_tokens_decoder (required by merge_added_tokens_decoder)
#   tokenizer_config.json lacks add_bos_token, and frontend.cpp reads value("add_bos_token", true),
#                         so the absent field defaults to true and fails the prefix-semantics check
# vocab.json and merges.txt are byte-identical to the canonical Qwen3.8 export, its 33
# added_tokens_decoder entries match this checkpoint's tokenizer.json exactly, and the canonical
# additional_special_tokens are all present in it. The canonical tokenizer.json is substituted and
# the frontend-required fields are merged into the checkpoint's own tokenizer_config.json.
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
src, canon, stage = map(pathlib.Path, sys.argv[1:4])
config = json.loads((src / 'tokenizer_config.json').read_text(encoding='utf-8'))
canonical = json.loads((canon / 'tokenizer_config.json').read_text(encoding='utf-8'))
tokenizer = json.loads((canon / 'tokenizer.json').read_text(encoding='utf-8'))
config['added_tokens_decoder'] = canonical['added_tokens_decoder']
config['add_bos_token'] = canonical['add_bos_token']
config['additional_special_tokens'] = canonical['additional_special_tokens']
# Mirror every rule the engine frontend enforces, so a bad config fails here and not at serve time.
assert config['add_bos_token'] is False, config['add_bos_token']
assert config['add_prefix_space'] is False, config['add_prefix_space']
assert config['pad_token'] == '<|endoftext|>', config['pad_token']
assert len(config['added_tokens_decoder']) == len(tokenizer['added_tokens'])
assert {str(t['id']) for t in tokenizer['added_tokens']} == set(config['added_tokens_decoder'])
stage.mkdir(parents=True, exist_ok=True)
(stage / 'tokenizer_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
print('merged tokenizer_config: decoder', len(config['added_tokens_decoder']), 'add_bos_token', config['add_bos_token'])
"@ $src $canon $stage

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
