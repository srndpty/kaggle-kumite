# PR のベースとの全差分から検査範囲を決める。
param(
    [Parameter(Mandatory)] [ValidatePattern('^[a-f0-9]{40}$')] [string] $Base,
    [Parameter(Mandatory)] [ValidatePattern('^[a-f0-9]{40}$')] [string] $Head
)

$ErrorActionPreference = 'Stop'
# リネームは削除と追加に分け、移動元と移動先の両方を検査対象にする。
$changed = @(git -c core.quotepath=false diff --name-only --no-renames "$Base...$Head" --)
if ($LASTEXITCODE -ne 0) {
    throw 'PR の差分を取得できません'
}

$mode = 'docs'
$competitions = @()
foreach ($path in $changed) {
    if ($path -cmatch '^(README\.md|AGENTS\.md|knowledge/[^/]+\.md|competitions/\d{3}-[^/]+/(README|report)\.md)$') {
        continue
    }
    if ($path -cmatch '^(competitions/\d{3}-[A-Za-z0-9_-]+)/') {
        $mode = 'competitions'
        $competitions += $Matches[1]
        continue
    }
    $mode = 'full'
    $competitions = @()
    break
}

$targets = @($competitions | Sort-Object -Unique)
"mode=$mode"
"competitions=$(ConvertTo-Json -InputObject $targets -Compress)"
