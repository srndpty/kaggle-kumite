#Requires -Version 7
# 開発用統一コマンド。使い方: dev <command> [args]  (dev help で一覧)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location $PSScriptRoot

$command = if ($args.Count -gt 0) { $args[0] } else { "help" }
$rest = @($args | Select-Object -Skip 1)

# 外部コマンドを実行し、失敗したらその終了コードで即終了する。
function Invoke-Native {
    $exe = $args[0]
    $exeArgs = @($args | Select-Object -Skip 1)
    & $exe @exeArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Kumite { Invoke-Native uv run python -m kumite @args }
function Invoke-Lint { Invoke-Native uv run ruff check . }
function Invoke-Test { Invoke-Native uv run pytest @rest }

$commands = [ordered]@{
    help     = @{ Description = "コマンド一覧を表示"; Action = { Show-Help } }
    setup    = @{ Description = "依存関係の同期と pre-commit の導入"; Action = {
            Invoke-Native uv sync
            Invoke-Native uv run pre-commit install
        }
    }
    lint     = @{ Description = "ruff による lint"; Action = { Invoke-Lint } }
    format   = @{ Description = "ruff による整形（lint の自動修正を含む）"; Action = {
            Invoke-Native uv run ruff check --fix .
            Invoke-Native uv run ruff format .
        }
    }
    test     = @{ Description = "pytest を実行（追加引数はそのまま渡す）"; Action = { Invoke-Test } }
    check    = @{ Description = "lint + 整形確認 + 型検査 + テスト（CI と同一）"; Action = {
            Invoke-Lint
            Invoke-Native uv run ruff format --check .
            Invoke-Native uv run pyright
            Invoke-Test
        }
    }
    clean    = @{ Description = "キャッシュを削除（data/ outputs/ は消さない）"; Action = {
            Get-Item -Force ".ruff_cache", ".pytest_cache" -ErrorAction Ignore |
                Remove-Item -Recurse -Force
            Get-ChildItem -Recurse -Directory -Force -Filter "__pycache__" "src", "tests", "competitions" |
                Remove-Item -Recurse -Force
        }
    }
    list     = @{ Description = "カタログ一覧と着手状況"; Action = { Invoke-Kumite list } }
    new      = @{ Description = "コンペ用ディレクトリを生成: dev new <番号>"; Action = { Invoke-Kumite new @rest } }
    download = @{ Description = "コンペのデータを取得: dev download <番号>"; Action = { Invoke-Kumite download @rest } }
    submit   = @{ Description = "提出: dev submit <番号> <ファイル> -m <メッセージ>"; Action = { Invoke-Kumite submit @rest } }
    compare  = @{ Description = "同じ fold どうしで比較: dev compare <番号> <基準> <run_id...>"; Action = { Invoke-Kumite compare @rest } }
    submissions = @{ Description = "提出履歴とスコア: dev submissions <番号>"; Action = { Invoke-Kumite submissions @rest } }
}

function Show-Help {
    Write-Host "使い方: dev <command> [args]`n"
    foreach ($name in $commands.Keys) {
        Write-Host ("  {0,-12} {1}" -f $name, $commands[$name].Description)
    }
}

if (-not $commands.Contains($command)) {
    Write-Host "不明なコマンド: $command`n"
    Show-Help
    exit 1
}

& $commands[$command].Action
