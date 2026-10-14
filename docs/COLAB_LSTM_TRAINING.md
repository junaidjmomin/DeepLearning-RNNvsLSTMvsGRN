# Train Person C's LSTM in Google Colab

Open [`notebooks/person_c_lstm_colab.ipynb`](../notebooks/person_c_lstm_colab.ipynb) in Google Colab or run it from VS Code using the official Google Colab extension. Both options can use a hosted Colab GPU runtime.

## Run from VS Code on Fedora

1. Install the **Google Colab** extension from the VS Code Extensions view.
2. Open `notebooks/person_c_lstm_colab.ipynb` in VS Code.
3. Select **Select Kernel → Colab → New Colab Server**, sign in to Google, and choose a GPU machine type when provisioning the server.
4. Upload `colab/person_c_training_code.zip` to the Colab server from the VS Code Explorer (right-click the ZIP and choose **Upload to Colab**).
5. Run the notebook cells. The code detects the uploaded ZIP, mounts Drive, and writes model artifacts and metrics to Drive.

Choose a hosted Colab server. A Colab **local runtime** executes on your Fedora machine instead of Google's GPU.

## Before running

The processed DailyDialog files are tracked in the GitHub repository and will be cloned by the notebook. The Person C training code is currently local, so download [`colab/person_c_training_code.zip`](../colab/person_c_training_code.zip) from this project. The notebook will prompt you to upload that ZIP after cloning, then copy its files into the Colab project directory. If you later push the Person C files to GitHub, the upload prompt is skipped.

In Colab, choose **Runtime → Change runtime type → T4 GPU** (or another available GPU), upload and open the notebook, then run the cells in order. Upload the code ZIP if prompted and connect Google Drive when prompted. Train contexts 3, 5, and 10 separately; the notebook writes each run's checkpoints and metrics to `MyDrive/ContextNext_PersonC/`.

After training, copy files under the Drive `artifacts/models/` and `results/` folders back into those matching folders in the local project. The 3-word `lstm_best.keras` checkpoint is the default LSTM checkpoint used by the prediction app.
