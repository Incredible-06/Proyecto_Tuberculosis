[CmdletBinding()]
param(
    [int]$Port = 8000,
    [switch]$RebuildFrontend,
    [switch]$SkipDependencyInstall
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$runtimeDirectory = Join-Path $projectRoot '.local\runtime\api'
$venvDirectory = Join-Path $runtimeDirectory '.venv'
$venvPython = Join-Path $venvDirectory 'Scripts\python.exe'
$requirementsFile = Join-Path $projectRoot 'api\requirements.txt'
$mlRequirementsFile = Join-Path $projectRoot 'api\requirements-ml.txt'
$frozenClassifier = Join-Path $projectRoot 'ml\models\frozen_phase5\vit_b_16_best.pt'
$frozenDetector = Join-Path $projectRoot 'ml\models\frozen_phase5\yolo11n_best.pt'
$demoDataDirectory = Join-Path $projectRoot 'api\demo_data'
$webDirectory = Join-Path $projectRoot 'web'
$webDistIndex = Join-Path $webDirectory 'dist\index.html'

foreach ($weightPath in @($frozenClassifier, $frozenDetector)) {
    if (-not (Test-Path -LiteralPath $weightPath)) {
        throw "Falta el archivo de pesos congelados: $weightPath"
    }
    if ((Get-Item -LiteralPath $weightPath).Length -lt 1024) {
        throw "El archivo $weightPath parece un puntero de Git LFS sin descargar. Ejecute 'git lfs pull' e intente nuevamente."
    }
}

New-Item -ItemType Directory -Force -Path $runtimeDirectory | Out-Null

function Get-SystemPython {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        return @{ Command = $launcher.Source; Prefix = @('-3') }
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @{ Command = $python.Source; Prefix = @() }
    }
    throw 'No se encontró Python. Instale Python 3.12 o superior y vuelva a ejecutar run.ps1.'
}

$systemPython = Get-SystemPython

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host 'Creando entorno virtual en .local/runtime/api/.venv ...'
    & $systemPython.Command @($systemPython.Prefix + @('-m', 'venv', $venvDirectory))
    if ($LASTEXITCODE -ne 0) {
        throw 'No se pudo crear el entorno virtual.'
    }
}

if (-not $SkipDependencyInstall) {
    Write-Host 'Instalando dependencias de la API ...'
    & $systemPython.Command @(
        $systemPython.Prefix + @(
            '-m', 'pip', '--python', $venvPython, 'install', '-r', $requirementsFile
        )
    )
    if ($LASTEXITCODE -ne 0) {
        throw 'No se pudieron instalar las dependencias de la API.'
    }
    & $venvPython -c "import torch, torchvision, ultralytics, PIL, numpy" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Instalando dependencias ML de inferencia (PyTorch CPU + Ultralytics) ...'
        & $systemPython.Command @(
            $systemPython.Prefix + @(
                '-m', 'pip', '--python', $venvPython, 'install',
                '--index-url', 'https://download.pytorch.org/whl/cpu',
                'torch', 'torchvision'
            )
        )
        if ($LASTEXITCODE -ne 0) {
            throw 'No se pudo instalar PyTorch/torchvision.'
        }
        & $systemPython.Command @(
            $systemPython.Prefix + @(
                '-m', 'pip', '--python', $venvPython, 'install', '-r', $mlRequirementsFile
            )
        )
        if ($LASTEXITCODE -ne 0) {
            throw 'No se pudieron instalar las dependencias ML de la API.'
        }
    }
}

$secretFile = Join-Path $runtimeDirectory 'dev-session-secret'
if (-not (Test-Path -LiteralPath $secretFile)) {
    $secretBytes = [byte[]]::new(48)
    $random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $random.GetBytes($secretBytes)
    }
    finally {
        $random.Dispose()
    }
    $secret = [Convert]::ToBase64String($secretBytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    Set-Content -LiteralPath $secretFile -Value $secret -Encoding ascii
}

if (-not $env:APP_ENV) {
    $env:APP_ENV = 'development'
}
$env:APP_SESSION_SECRET = (Get-Content -LiteralPath $secretFile -Raw).Trim()
if (-not $env:APP_DATABASE_PATH) {
    $env:APP_DATABASE_PATH = Join-Path $runtimeDirectory 'tuberculosis.sqlite3'
}
if (-not $env:APP_STORAGE_ROOT) {
    $env:APP_STORAGE_ROOT = $runtimeDirectory
}
$env:PYTHONPATH = Join-Path $projectRoot 'api'

$npm = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npm) {
    throw 'No se encontró npm. Instale Node.js 20 o superior y vuelva a ejecutar run.ps1.'
}

if ($RebuildFrontend -or -not (Test-Path -LiteralPath $webDistIndex)) {
    Push-Location -LiteralPath $webDirectory
    try {
        if (-not (Test-Path -LiteralPath (Join-Path $webDirectory 'node_modules'))) {
            Write-Host 'Instalando dependencias del frontend ...'
            & npm.Source install
            if ($LASTEXITCODE -ne 0) {
                throw 'No se pudieron instalar las dependencias del frontend.'
            }
        }
        Write-Host 'Construyendo el frontend ...'
        & npm.Source run build
        if ($LASTEXITCODE -ne 0) {
            throw 'No se pudo construir el frontend.'
        }
    }
    finally {
        Pop-Location
    }
}

$seedDb = Join-Path $demoDataDirectory 'tuberculosis.sqlite3'
if ((-not (Test-Path -LiteralPath $env:APP_DATABASE_PATH)) -and (Test-Path -LiteralPath $seedDb)) {
    Write-Host 'Inicializando base de datos y estudios de demostración ...'
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $env:APP_DATABASE_PATH) | Out-Null
    New-Item -ItemType Directory -Force -Path $env:APP_STORAGE_ROOT | Out-Null
    Copy-Item -LiteralPath $seedDb -Destination $env:APP_DATABASE_PATH -Force
    foreach ($folder in @('radiographs', 'explainability')) {
        $seedFolder = Join-Path $demoDataDirectory $folder
        if (Test-Path -LiteralPath $seedFolder) {
            Copy-Item -LiteralPath $seedFolder -Destination $env:APP_STORAGE_ROOT -Recurse -Force
        }
    }
}

$userCount = & $venvPython -c "from app.config import Settings; from app.database import initialize_database, connect_database; s = Settings.from_environment(); initialize_database(s); c = connect_database(s.database_path); print(c.execute('SELECT COUNT(*) FROM users').fetchone()[0]); c.close()"
if ([int]$userCount -eq 0) {
    Write-Host ''
    Write-Host 'No existen cuentas. Cree la cuenta administradora inicial.'
    Write-Host 'También puede definir BOOTSTRAP_ADMIN_PASSWORD antes de ejecutar este script.'
    Push-Location -LiteralPath (Join-Path $projectRoot 'api')
    try {
        & $venvPython -m app.bootstrap_admin --username admin.local --display-name 'Administrador inicial'
        if ($LASTEXITCODE -ne 0) {
            throw 'No se pudo crear la cuenta administradora inicial.'
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host ''
Write-Host "Sistema disponible en http://127.0.0.1:$Port/"
Write-Host 'Presione Ctrl+C para detenerlo.'
Push-Location -LiteralPath (Join-Path $projectRoot 'api')
try {
    & $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port $Port
}
finally {
    Pop-Location
}
