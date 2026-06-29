$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    './scripts/cpu_util_sample.ps1',
    [ref]$null, [ref]$errors) | Out-Null
if ($errors.Count -eq 0) { 'NO PARSE ERRORS' }
else { $errors | ForEach-Object { '{0}:{1}: {2}' -f $_.Extent.StartLineNumber, $_.Extent.StartColumnNumber, $_.Message } }
