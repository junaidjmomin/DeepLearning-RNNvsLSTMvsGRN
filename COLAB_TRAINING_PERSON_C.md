# Train Person C (LSTM) in Google Colab

Use Colab to train on GPU/TPU instead of local hardware.

## Files
- Notebook: `notebooks/person_c_lstm_colab.ipynb`
- Bundle (optional): `colab/person_c_training_code.zip` contains the Person C training files

## Steps
1. Open `notebooks/person_c_lstm_colab.ipynb` in Google Colab.
2. Set runtime: Runtime → Change runtime type → GPU (T4 recommended) or TPU.
3. Run cells top-to-bottom. The notebook will:
   - Clone the repo (or let you upload the zip if missing)
   - Install TensorFlow/Numpy
   - Run training for context lengths 3, 5, 10
   - Compare results
   - Export `results/lstm_prediction_examples.json`
   - Optionally save `artifacts/` and `results/` to Google Drive
4. When done, download the generated `artifacts/` and `results/` folders (or copy from Drive) to your local repo root.
5. Locally, run API/UI/Docker using those artifacts.

## Notes
- Training code is identical to repo: `src/training/train_lstm.py`, `src/training/analyze_predictions.py`, `src/models/lstm.py`.
- Outputs follow `ARTIFACT_GUIDELINES.md` (ctx3 uses `lstm_best.keras`, ctx5/10 get `_ctx5/_ctx10` suffixes).
- Make sure processed data (`data/processed/*.json`) is present in the repo/Colab context.
