import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.layers import TextVectorization
    return TextVectorization, layers, os, pd, tf


@app.cell
def _(os):
    DATA_DIR = "/home/sibel/Langue-wu/Data/Corpus_aligné"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    DEV_PATH = os.path.join(DATA_DIR, "dev.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")

    src = "mandarin"  # langue source (Mandarin)
    tgt = "wu" # langue cible (Wu)
    return DEV_PATH, TEST_PATH, TRAIN_PATH, src, tgt


@app.cell
def _(tf):
    BATCH_SIZE     = 64
    EPOCHS         = 20
    MAX_SRC_LEN    = 50
    MAX_TGT_LEN    = 50
    MAX_VOCAB_SIZE = 4000

    D_MODEL     = 128
    N_ENC       = 4
    N_DEC       = 4
    N_HEADS     = 8
    DFF         = 512
    DROP        = 0.1

    AUTOTUNE = tf.data.AUTOTUNE
    return (
        AUTOTUNE,
        BATCH_SIZE,
        DFF,
        D_MODEL,
        EPOCHS,
        MAX_SRC_LEN,
        MAX_TGT_LEN,
        MAX_VOCAB_SIZE,
        N_DEC,
        N_ENC,
        N_HEADS,
    )


@app.cell
def _(DEV_PATH, TEST_PATH, TRAIN_PATH, pd, src, tgt):
    def load_data(path):
        df = pd.read_csv(path)
        df = df[[src, tgt]].dropna()
        df[src] = df[src].astype(str)
        df[tgt] = df[tgt].astype(str)
        return df

    train_df = load_data(TRAIN_PATH)
    dev_df   = load_data(DEV_PATH)
    test_df  = load_data(TEST_PATH)

    train_df.head()
    return dev_df, test_df, train_df


@app.cell
def _(dev_df, test_df, tgt, train_df):
    START = "[START]"
    END   = "[END]"

    # Ajouter START et END à la langue cible (Wu)
    train_df[tgt] = START + train_df[tgt] + END
    dev_df[tgt]   = START + dev_df[tgt] + END
    test_df[tgt]  = START + test_df[tgt] + END
    return END, START


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""TextVectorization""")
    return


@app.cell
def _(
    MAX_SRC_LEN,
    MAX_TGT_LEN,
    MAX_VOCAB_SIZE,
    TextVectorization,
    src,
    tgt,
    train_df,
):
    src_vectorizer = TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_SRC_LEN,
        standardize=None,
        split="character" # par caractère
    )

    tgt_vectorizer = TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_TGT_LEN + 1,  # pour decoder shift
        standardize=None,
        split="character"
    )

    # adapter que le train
    src_vectorizer.adapt(train_df[src].values)
    tgt_vectorizer.adapt(train_df[tgt].values)

    SRC_VOCAB_SIZE = len(src_vectorizer.get_vocabulary())
    TGT_VOCAB_SIZE = len(tgt_vectorizer.get_vocabulary())

    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE
    return SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, src_vectorizer, tgt_vectorizer


@app.cell
def _(
    AUTOTUNE,
    BATCH_SIZE,
    MAX_TGT_LEN,
    dev_df,
    src,
    src_vectorizer,
    test_df,
    tf,
    tgt,
    tgt_vectorizer,
    train_df,
):
    def make_dataset(df):
        ds = tf.data.Dataset.from_tensor_slices((df[src], df[tgt]))

        def _prep(x_src, x_tgt):
            # vectorize
            x_src_vec = src_vectorizer(x_src)
            x_tgt_vec = tgt_vectorizer(x_tgt)

            # decoder inputs = 去掉最后一个 token
            dec_in  = x_tgt_vec[:-1][:MAX_TGT_LEN]

            # decoder outputs = 去掉第一个 token
            dec_out = x_tgt_vec[1:][:MAX_TGT_LEN]

            # 
            dec_in  = tf.pad(dec_in,  [[0, MAX_TGT_LEN - tf.shape(dec_in)[0]]])
            dec_out = tf.pad(dec_out, [[0, MAX_TGT_LEN - tf.shape(dec_out)[0]]])

            return {
                "encoder_inputs": x_src_vec,
                "decoder_inputs": dec_in
            }, dec_out

        return (
            ds.shuffle(len(df))
            .map(_prep, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )

    train_ds = make_dataset(train_df)
    dev_ds   = make_dataset(dev_df)
    test_ds  = make_dataset(test_df)

    train_ds
    return dev_ds, train_ds


@app.cell
def _(layers, tf):
    class PositionalEmbedding(layers.Layer):
        def __init__(self, vocab_size, d_model, max_len):
            super().__init__()
            self.token = layers.Embedding(vocab_size, d_model)
            self.pos   = layers.Embedding(max_len, d_model)

        def call(self, x):
            seq_len = tf.shape(x)[1]
            positions = tf.range(seq_len)
            return self.token(x) + self.pos(positions)
    return (PositionalEmbedding,)


@app.cell
def _(mo):
    mo.md(r"""https://keras.io/guides/functional_api/""")
    return


@app.cell
def _(
    DFF,
    D_MODEL,
    MAX_SRC_LEN,
    MAX_TGT_LEN,
    N_DEC,
    N_ENC,
    N_HEADS,
    PositionalEmbedding,
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE,
    layers,
    tf,
):
    # ----- 1. Définition des entrées -----
    encoder_inputs = tf.keras.Input(
        shape=(MAX_SRC_LEN,), dtype="int64", name="encoder_inputs"
    )
    decoder_inputs = tf.keras.Input(
        shape=(MAX_TGT_LEN,), dtype="int64", name="decoder_inputs"
    )

    # ----- 2. Embedding + Positionnel -----
    x_enc = PositionalEmbedding(SRC_VOCAB_SIZE, D_MODEL, MAX_SRC_LEN)(encoder_inputs)

    # Encoder
    for _ in range(N_ENC):
        # --- Self-Attention ---
        attn_out = layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=D_MODEL)(x_enc, x_enc)
        x_enc = layers.LayerNormalization()(x_enc + attn_out)

        # --- Feed-Forward ---
        ffn_out = layers.Dense(DFF, activation="relu")(x_enc)
        ffn_out = layers.Dense(D_MODEL)(ffn_out)
        x_enc = layers.LayerNormalization()(x_enc + ffn_out)

    encoder_outputs = x_enc   # Contexte pour le décodeur

    x_dec = PositionalEmbedding(TGT_VOCAB_SIZE, D_MODEL, MAX_TGT_LEN)(decoder_inputs)

    for _ in range(N_DEC):
        # --- Masked Self-Attention (auto-régressif) ---
        masked_att = layers.MultiHeadAttention(
            num_heads=N_HEADS, key_dim=D_MODEL
        )(x_dec, x_dec, use_causal_mask=True)
        x_dec = layers.LayerNormalization()(x_dec + masked_att)

        # --- Cross-Attention avec l’encodeur ---
        cross_att = layers.MultiHeadAttention(
            num_heads=N_HEADS, key_dim=D_MODEL
        )(x_dec, encoder_outputs, encoder_outputs)
        x_dec = layers.LayerNormalization()(x_dec + cross_att)

        # --- Feed-Forward ---
        ffn_out = layers.Dense(DFF, activation="relu")(x_dec)
        ffn_out = layers.Dense(D_MODEL)(ffn_out)
        x_dec = layers.LayerNormalization()(x_dec + ffn_out)

    decoder_outputs = layers.Dense(TGT_VOCAB_SIZE, name="final_output")(x_dec)
    return decoder_inputs, decoder_outputs, encoder_inputs


@app.cell
def _(decoder_inputs, decoder_outputs, encoder_inputs, tf):
    model = tf.keras.Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_outputs,
        name="Transformer"
    )

    model.summary()
    return (model,)


@app.cell
def _(model, tf):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def masked_loss(y_true, y_pred):
        mask = tf.cast(y_true != 0, tf.float32)
        loss = loss_fn(y_true, y_pred)
        return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

    optimizer = tf.keras.optimizers.Adam(1e-3)

    model.compile(
        optimizer=optimizer,
        loss=masked_loss
    )
    return


@app.cell
def _(EPOCHS, dev_ds, model, train_ds):
    history = model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=EPOCHS
    )
    return


@app.cell
def _(model):
    model.save("/home/sibel/Langue-wu/models/baseline_tr_m2w.keras")
    return


@app.cell
def _(END, MAX_TGT_LEN, START, model, src_vectorizer, tf, tgt_vectorizer):
    def translate(sentence):
        # Encodage de la phrase source
        src_seq = src_vectorizer([sentence])

        # ID du token START
        start_id = tgt_vectorizer([START])[0][0].numpy()

        # Décodage auto-régressif
        dec_seq = [start_id]

        for _ in range(MAX_TGT_LEN):
            # Pad à la bonne longueur
            dec_input = tf.constant(
                [dec_seq + [0] * (MAX_TGT_LEN - len(dec_seq))],
                dtype=tf.int32
            )

            # Passage dans le modèle
            logits = model({
                "encoder_inputs": src_seq,
                "decoder_inputs": dec_input
            })

            next_id = int(tf.argmax(logits[0, len(dec_seq)-1]))

            # Arrêter si END ou PAD
            if next_id == 0 or tgt_vectorizer.get_vocabulary()[next_id] == END:
                break

            dec_seq.append(next_id)

        # Convertir les IDs en caractères
        vocab = tgt_vectorizer.get_vocabulary()
        tokens = dec_seq[1:]   # enlever START

        return "".join(vocab[i] for i in tokens)
    return (translate,)


@app.cell
def _(translate):
    print( translate("你好") )
    print( translate("你今天吃饭了吗") )
    print( translate("我不会说上海话") )
    print( translate("今天天气真好") )
    print( translate("请你帮我一下") )
    print( translate("祝您开心") )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
