@echo off
REM ContextNext — train all three models with a single command.
REM
REM Usage:
REM   train_all.bat                        # train all 3 models, context=3 by default
REM   train_all.bat --max-train-samples 5000  # quick smoke-test
REM
REM All models share the same data splits, vocabulary, and hyperparameters
REM so their results are directly comparable.

SET ARGS=%*

echo.
echo ================================================
echo  ContextNext: Training GRU
echo ================================================
python -m src.training.train_unified --model gru %ARGS%
if errorlevel 1 ( echo GRU training failed & exit /b 1 )

echo.
echo ================================================
echo  ContextNext: Training RNN
echo ================================================
python -m src.training.train_unified --model rnn %ARGS%
if errorlevel 1 ( echo RNN training failed & exit /b 1 )

echo.
echo ================================================
echo  ContextNext: Training LSTM
echo ================================================
python -m src.training.train_unified --model lstm %ARGS%
if errorlevel 1 ( echo LSTM training failed & exit /b 1 )

echo.
echo ================================================
echo  All models trained.  Results in results/
echo ================================================
