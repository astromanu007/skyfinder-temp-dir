@echo off
echo ============================================================
echo  SkyFinder DIR - Running all 4 experiments
echo ============================================================

echo.
echo [1/4] Baseline (no DIR techniques)
python train.py --experiment baseline --epochs 20 --batch_size 32
if errorlevel 1 goto error

echo.
echo [2/4] + LDS only
python train.py --experiment lds --epochs 20 --batch_size 32 --lds --reweight sqrt_inv
if errorlevel 1 goto error

echo.
echo [3/4] + FDS only
python train.py --experiment fds --epochs 20 --batch_size 32 --fds
if errorlevel 1 goto error

echo.
echo [4/4] + LDS + FDS (full DIR)
python train.py --experiment dir --epochs 20 --batch_size 32 --lds --reweight sqrt_inv --fds
if errorlevel 1 goto error

echo.
echo ============================================================
echo  All experiments done! Running analysis...
echo ============================================================
python analyze.py
goto done

:error
echo.
echo [ERROR] An experiment failed. Check the output above.
exit /b 1

:done
echo.
echo [✓] All done! Check ..\results\ for plots and tables.
