function Get-PythonCodeLines {
    param(
        [Parameter(Mandatory)]
        [string] $Section,

        [Parameter(Mandatory)]
        [string[]] $Path
    )

    $total_python = 0
    $total_cython = 0

    foreach ($p in $Path) {
        if (-not (Test-Path $p)) {
            Write-Warning "Path not found: $p"
            continue
        }

        $csv = cloc $p --quiet --csv 2>$null
        if (-not $csv) { continue }

        $table = $csv | ConvertFrom-Csv
        $python = $table | Where-Object { $_.language -eq "Python" }
        $cython = $table | Where-Object { $_.language -eq "Cython" }
        if ($python) {
            $total_python += [int]$python.Code
        }
        if ($cython) {
            $total_cython += [int]$cython.Code
        }
    }

    return [pscustomobject]@{
        Section = $Section;
        Python = $total_python;
        Cython = $total_cython
    }
}

$table = @(
    Get-PythonCodeLines "Core" @(
        "src/seemps/cython"
        "src/seemps/typing.py"
        "src/seemps/version.py"
        "src/seemps/tools.py"
    )
    Get-PythonCodeLines "BLAS" @(
        "src/seemps/state"
        "src/seemps/operators"
        "src/seemps/hdf5.py"
    )
    Get-PythonCodeLines "LAPACK" @(
        "src/seemps/solve"
        "src/seemps/optimization"
        "src/seemps/qft.py"
    )
    Get-PythonCodeLines "Loading" @(
        "src/seemps/analysis/cross"
        "src/seemps/analysis/expansion"
        "src/seemps/analysis/factories.py"
        "src/seemps/analysis/mesh.py"
        "src/seemps/analysis/polynomials.py"
        "src/seemps/analysis/space.py"
        "src/seemps/analysis/tree"
    )
    Get-PythonCodeLines "Differentiation" @(
        "src/seemps/analysis/derivatives/finite_differences.py"
        "src/seemps/analysis/derivatives/fourier_differentiation.py"
        "src/seemps/analysis/hdaf.py"
    )
    Get-PythonCodeLines "Integration" @("src/seemps/analysis/integration")
    Get-PythonCodeLines "Interpolation" @(
        "src/seemps/analysis/interpolation.py"
        "src/seemps/analysis/lagrange.py"
    )
    Get-PythonCodeLines "Evolution" @("src/seemps/evolution")
    Get-PythonCodeLines "Quantum" @(
        "src/seemps/expectation.py"
        "src/seemps/hamiltonians.py"
        "src/seemps/register"
    )
    Get-PythonCodeLines "Unit tests" @(
        "tests"
    )
)

function Convert-ToLatexTable {
    param(
        [Parameter(Mandatory)]
        [object[]] $Data,

        [string] $Caption = "Lines of code by module",
        [string] $Label   = "tab:code-decomposition"
    )

    $lines = @()
    $lines += "\begin{table}[h]"
    $lines += "\centering"
    $lines += "\begin{tabular}{lrr}"
    $lines += "\hline"
    $lines += "Module & Python (LOC) & Cython (LOC) \\"
    $lines += "\hline"
    $total_python = 0
    $total_cython = 0
    foreach ($row in $Data) {
        $lines += "$($row.Section) & $($row.Python) & $($row.Cython) \\"
        $total_python += $row.Python
        $total_cython += $row.Cython
    }

    $total = ($Data | Measure-Object Lines -Sum).Sum
    $lines += "\hline"
    $lines += "\textbf{Total} & $total_python & $total_cython \\"
    $lines += "\hline"

    $lines += "\end{tabular}"
    $lines += "\caption{$Caption}"
    $lines += "\label{$Label}"
    $lines += "\end{table}"

    return $lines -join "`n"
}

Convert-ToLatexTable $table