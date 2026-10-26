from src.data.build_vocab import vocab_data
import tensorflow as tf

def get_embedding_layer(embedding_dim=128):
    vocab_size = len(vocab_data)

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim
    )

    return embedding_layer

'''

Embedding vector — 128 dimensions
[0.21, -0.35, 0.78, 0.14, ... , -0.09]
128 learned numerical values for this token.


When you run a file using standard python path/to/file.py, Python automatically adds that file's folder (src/models) to its import search path (sys.path).

Because of that, Python starts looking for packages inside src/models. When it reads from src.data..., it searches for a folder named src inside src/models (which doesn't exist), causing the ModuleNotFoundError.

What -m Actually DoesRunning python -m src.models.embeddings flips how Python initializes:Sets Root Directory: Instead of adding src/models to sys.path, it adds your current working directory (C:\Users\fahad\OneDrive\Desktop\DLRL) to sys.path.Treats Code as a Module: Python looks into your current folder, finds the src folder, and traces down the package path src $\rightarrow$ data $\rightarrow$ build_vocab.

'''