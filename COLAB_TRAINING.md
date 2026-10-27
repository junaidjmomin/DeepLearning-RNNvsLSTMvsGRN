# Full-data training in Google Colab

Train three architectures: vanilla RNN, GRU, and LSTM, each at context lengths 3, 5, and 10. That is nine runs. The three-word split has 678,869 train, 89,517 validation, and 85,945 test examples. Longer contexts generate their own complete splits from the same dialogue partition and use the shared vocabulary. The test split is used only for final evaluation.

| Context words | Train examples | Validation examples | Test examples |
|---:|---:|---:|---:|
| 3 | 678,869 | 89,517 | 85,945 |
| 5 | 537,088 | 71,241 | 68,148 |
| 10 | 290,569 | 39,346 | 37,065 |

1. Commit and push the repository changes so Colab clones the corrected code.
2. Open `notebooks/train_all_colab.ipynb` in Colab and choose a GPU runtime.
3. Run the cells in order. Training uses the full split files at each context length, up to 20 epochs, and selects the best checkpoint by validation Top-1 accuracy. It stops after three epochs without improvement. Nine full runs may exceed one Colab session. The trainer backs up progress after each epoch in Drive, so rerunning the cell resumes an interrupted model; completed runs are skipped when their full-data config and checkpoint exist.
4. Copy `MyDrive/ContextNext_FullData/artifacts/models/` and `results/` to the same local paths in this repository. For local dataset-row lookup at longer contexts, run `python -m src.data.preprocess --context-size 5` and `python -m src.data.preprocess --context-size 10` here. Restart the API and dashboard. Docker Compose generates those split files during its image build.

Full-data training does not guarantee the highest possible accuracy. Compare the held-out test metrics, then tune settings consistently if needed. Never choose settings using the test set.
