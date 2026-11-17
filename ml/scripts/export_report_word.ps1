param(
  [string]$InputDocx = 'docs/Informe_Fases_1_y_2_Final.docx',
  [string]$OutputPdf = 'docs/_qa/Informe_Fases_1_y_2.pdf'
)
$ErrorActionPreference = 'Stop'
$inputFile = (Resolve-Path -LiteralPath $InputDocx).Path
$outputFile = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputPdf))
New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($outputFile)) -Force | Out-Null
$word = $null
$document = $null
try {
  $word = New-Object -ComObject Word.Application
  $word.Visible = $false
  $word.DisplayAlerts = 0
  $word.AutomationSecurity = 3
  $document = $word.Documents.Open($inputFile, $false, $true, $false)
  $document.ExportAsFixedFormat($outputFile, 17)
  $pages = $document.ComputeStatistics(2)
  Write-Output "Word export completed. Pages: $pages"
} finally {
  if ($null -ne $document) { $document.Close(0); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
  if ($null -ne $word) { $word.Quit(0); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
}
