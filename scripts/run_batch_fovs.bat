@echo off
REM Batch run SQI pipeline for multiple FOVs.
REM Usage: run_batch_fovs.bat

set DATA_FLD=\\192.168.0.116\durian3\Lilian\022425_FTD_smFISH_MBP_NRGN\coverslip3_FTD_group2\MBP_NRGN_set1
set CACHE_ROOT=\\192.168.0.116\durian3\Lilian\merfish_sqi_cache
set OUT_ROOT=output\022425_FTD_smFISH_MBP_NRGN\coverslip3_FTD_group2\MBP_NRGN_set1
set FOV_LIST=20 40

REM Extract set number from DATA_FLD (e.g. MBP_NRGN_set5 → 5)
for %%P in (%DATA_FLD%) do set _LAST_DIR=%%~nxP
set ZSCAN_NUM=%_LAST_DIR:*set=%

for %%F in (%FOV_LIST%) do (
    echo ============================================================
    echo Processing FOV %%F ...
    echo ============================================================
    python scripts\run_sqi_from_fov_zarr.py ^
        --fov_zarr   "%DATA_FLD%\Conv_zscan%ZSCAN_NUM%_%%F.zarr" ^
        --data_fld   "%DATA_FLD%" ^
        --cache_root "%CACHE_ROOT%" ^
        --out_root   "%OUT_ROOT%"
    if errorlevel 1 (
        echo [ERROR] FOV %%F failed!
    ) else (
        echo [OK] FOV %%F done.
    )
    echo.
)

echo ============================================================
echo All FOVs complete.
echo ============================================================
pause
