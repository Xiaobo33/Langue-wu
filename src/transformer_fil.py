import marimo

__generated_with = "0.19.0"
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
    return TextVectorization, layers, np, os, pd, tf


@app.cell
def _(os):
    WORKSPACE_ROOT = "/Users/elodie/TAL/M2/traduction auto/Traduction-automatique-Mandarin-Wu"
    DATA_DIR = os.path.join(WORKSPACE_ROOT, "Data/Corpus_aligné")
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    DEV_PATH = os.path.join(DATA_DIR, "dev.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")

    BT_PATH = os.path.join(WORKSPACE_ROOT, "Data/backtranslated_w2m.csv")

    src = "mandarin"  # langue source (Mandarin)
    tgt = "wu" # langue cible (Wu)
    return BT_PATH, DEV_PATH, TEST_PATH, TRAIN_PATH, WORKSPACE_ROOT, src, tgt


@app.cell
def _(tf):
    BATCH_SIZE     = 64
    EPOCHS_BASE = 20
    EPOCHS_BT = 10

    MAX_SRC_LEN    = 50
    MAX_TGT_LEN    = 50
    MAX_VOCAB_SIZE = 4000

    D_MODEL     = 128
    N_ENC       = 4
    N_DEC       = 4
    N_HEADS     = 8
    DFF         = 512
    DROP        = 0.1

    PAD_ID = 0
    UNK_ID = 1

    AUTOTUNE = tf.data.AUTOTUNE
    return (
        AUTOTUNE,
        BATCH_SIZE,
        DFF,
        DROP,
        D_MODEL,
        EPOCHS_BASE,
        EPOCHS_BT,
        MAX_SRC_LEN,
        MAX_TGT_LEN,
        MAX_VOCAB_SIZE,
        N_DEC,
        N_ENC,
        N_HEADS,
    )


@app.cell
def _(pd, src, tgt):
    def load_data(path):
        df = pd.read_csv(path)
        df = df[[src, tgt]].dropna()
        df[src] = df[src].astype(str).str.strip()
        df[tgt] = df[tgt].astype(str).str.strip()
        return df
    return (load_data,)


@app.cell
def _(pd, src, tgt):
    def load_bt_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df[["mandarin", "wu"]].dropna()
        df["mandarin"] = df["mandarin"].astype(str).str.strip()
        df["wu"] = df["wu"].astype(str).str.strip()
        df = df.rename(columns={"mandarin": src, "wu": tgt})
        return df.reset_index(drop=True)
    return (load_bt_csv,)


@app.cell
def _(BT_PATH, DEV_PATH, TEST_PATH, TRAIN_PATH, load_bt_csv, load_data, pd):
    train_real = load_data(TRAIN_PATH)
    dev_df = load_data(DEV_PATH)
    test_df = load_data(TEST_PATH)

    bt_df = load_bt_csv(BT_PATH)

    train_base = train_real.sample(frac=1, random_state=42).reset_index(drop=True)
    train_bt = pd.concat([train_real, bt_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return dev_df, test_df, train_base, train_bt, train_real


@app.cell
def _(WORKSPACE_ROOT, load_bt_csv, os, pd, train_real):
    BT_FILTERED_PATH = os.path.join(WORKSPACE_ROOT, "Data/backtranslated_w2m_filtered.csv")

    bt_df_filtered = load_bt_csv(BT_FILTERED_PATH)

    train_bt_filtered = pd.concat([train_real, bt_df_filtered], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return (train_bt_filtered,)


@app.cell
def _(
    MAX_SRC_LEN,
    MAX_TGT_LEN,
    MAX_VOCAB_SIZE,
    TextVectorization,
    src,
    tgt,
    train_base,
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
        output_sequence_length=MAX_TGT_LEN,  # pour decoder shift
        standardize=None,
        split="character",
    )

    # adapter que le train
    src_vectorizer.adapt(train_base[src].values)
    tgt_vectorizer.adapt(train_base[tgt].values)

    SRC_VOCAB_SIZE = len(src_vectorizer.get_vocabulary())
    TGT_VOCAB_SIZE = len(tgt_vectorizer.get_vocabulary()) + 2 # pour START et END

    START_ID = TGT_VOCAB_SIZE - 2
    END_ID = TGT_VOCAB_SIZE - 1

    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, START_ID, END_ID
    return (
        END_ID,
        SRC_VOCAB_SIZE,
        START_ID,
        TGT_VOCAB_SIZE,
        src_vectorizer,
        tgt_vectorizer,
    )


@app.cell
def _(
    AUTOTUNE,
    BATCH_SIZE,
    MAX_TGT_LEN,
    TGT_VOCAB_SIZE,
    dev_df,
    src,
    src_vectorizer,
    test_df,
    tf,
    tgt,
    tgt_vectorizer,
    train_base,
    train_bt,
    train_bt_filtered,
):
    def make_dataset(df, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((df[src].values, df[tgt].values))

        def _prep(x_src, x_tgt):
            # vectorize
            x_src_vec = src_vectorizer(x_src)
            x_tgt_vec = tgt_vectorizer(x_tgt)

            x_tgt_vec = x_tgt_vec[x_tgt_vec != 0]

            #start id = size -2 et end id = size - 1
            start_token = tf.constant([TGT_VOCAB_SIZE - 2], dtype=tf.int64)
            end_token = tf.constant([TGT_VOCAB_SIZE - 1], dtype=tf.int64)

            x_tgt_full = tf.concat([start_token, x_tgt_vec, end_token], axis=0)

            # decoder inputs
            dec_in  = x_tgt_full[:-1]
            # decoder outputs
            dec_out = x_tgt_full[1:]

            dec_in = dec_in[:MAX_TGT_LEN]
            dec_out = dec_out[:MAX_TGT_LEN]

            dec_in  = tf.pad(dec_in,  [[0, MAX_TGT_LEN - tf.shape(dec_in)[0]]])
            dec_out = tf.pad(dec_out, [[0, MAX_TGT_LEN - tf.shape(dec_out)[0]]])

            return {
                "encoder_inputs": x_src_vec,
                "decoder_inputs": dec_in
            }, dec_out

        if shuffle:
            ds = ds.shuffle(len(df), reshuffle_each_iteration=True)

        return (
            ds.map(_prep, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )

    train_ds_base = make_dataset(train_base, shuffle=True)
    train_ds_bt   = make_dataset(train_bt, shuffle=True)
    train_ds_bt_filtered = make_dataset(train_bt_filtered, shuffle=True)
    dev_ds = make_dataset(dev_df, shuffle=False)
    test_ds = make_dataset(test_df, shuffle=False)
    return dev_ds, train_ds_base, train_ds_bt, train_ds_bt_filtered


@app.cell
def _(layers, tf):
    class PositionalEmbedding(layers.Layer):
        def __init__(self, vocab_size, d_model, max_len, **kwargs):
            super().__init__(**kwargs)
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.max_len = max_len

            self.token_emb = layers.Embedding(vocab_size, d_model)
            self.pos_emb = layers.Embedding(max_len, d_model)

        def call(self, x):
            seq_len = tf.shape(x)[1]
            positions = tf.range(start=0, limit=seq_len, delta=1)
            pos_embeddings = self.pos_emb(positions)
            token_embeddings = self.token_emb(x)
            return token_embeddings + pos_embeddings

        def get_config(self):
            config = super().get_config()
            config.update({
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "max_len": self.max_len,
            })
            return config
    return (PositionalEmbedding,)


@app.cell
def _(mo):
    mo.md(r"""
    https://keras.io/guides/functional_api/
    """)
    return


@app.cell
def _(
    DFF,
    DROP,
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
    # Inputs
    encoder_inputs = tf.keras.Input(
        shape=(MAX_SRC_LEN,), dtype="int64", name="encoder_inputs"
    )
    decoder_inputs = tf.keras.Input(
        shape=(MAX_TGT_LEN,), dtype="int64", name="decoder_inputs"
    )

    # Encoder
    x_enc = PositionalEmbedding(SRC_VOCAB_SIZE, D_MODEL, MAX_SRC_LEN)(encoder_inputs)
    x_enc = layers.Dropout(DROP)(x_enc)

    for i in range(N_ENC):
        # Self-Attention
        attn_out = layers.MultiHeadAttention(
            num_heads=N_HEADS, 
            key_dim=D_MODEL // N_HEADS,
            dropout=DROP
        )(x_enc, x_enc)
        attn_out = layers.Dropout(DROP)(attn_out)
        x_enc = layers.LayerNormalization(epsilon=1e-6)(x_enc + attn_out)

        # Feed-Forward
        ffn_out = layers.Dense(DFF, activation="relu")(x_enc)
        ffn_out = layers.Dense(D_MODEL)(ffn_out)
        ffn_out = layers.Dropout(DROP)(ffn_out)
        x_enc = layers.LayerNormalization(epsilon=1e-6)(x_enc + ffn_out)

    encoder_outputs = x_enc

    # Decoder
    x_dec = PositionalEmbedding(TGT_VOCAB_SIZE, D_MODEL, MAX_TGT_LEN)(decoder_inputs)
    x_dec = layers.Dropout(DROP)(x_dec)

    for i in range(N_DEC):
        # Masked Self-Attention
        masked_attn = layers.MultiHeadAttention(
            num_heads=N_HEADS,
            key_dim=D_MODEL // N_HEADS,
            dropout=DROP
        )(x_dec, x_dec, use_causal_mask=True)
        masked_attn = layers.Dropout(DROP)(masked_attn)
        x_dec = layers.LayerNormalization(epsilon=1e-6)(x_dec + masked_attn)

        # Cross-Attention
        cross_attn = layers.MultiHeadAttention(
            num_heads=N_HEADS,
            key_dim=D_MODEL // N_HEADS,
            dropout=DROP
        )(x_dec, encoder_outputs, encoder_outputs)
        cross_attn = layers.Dropout(DROP)(cross_attn)
        x_dec = layers.LayerNormalization(epsilon=1e-6)(x_dec + cross_attn)

        # Feed-Forward
        ffn_out = layers.Dense(DFF, activation="relu")(x_dec)
        ffn_out = layers.Dense(D_MODEL)(ffn_out)
        ffn_out = layers.Dropout(DROP)(ffn_out)
        x_dec = layers.LayerNormalization(epsilon=1e-6)(x_dec + ffn_out)

    # Output layer
    decoder_outputs = layers.Dense(TGT_VOCAB_SIZE, name="output_layer")(x_dec)

    # Model
    model = tf.keras.Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_outputs,
        name="Transformer_Mandarin_to_Wu"
    )

    model.summary()
    return (model,)


@app.cell
def _(model, tf):
    # Masked loss
    def masked_loss(y_true, y_pred):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )
        loss = loss_fn(y_true, y_pred)

        # Mask padding tokens (ID = 0)
        mask = tf.cast(y_true != 0, tf.float32)
        loss = loss * mask

        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    # Accuracy metric
    def masked_accuracy(y_true, y_pred):
        pred_ids = tf.cast(tf.argmax(y_pred, axis=-1), y_true.dtype)
        mask = tf.cast(y_true != 0, tf.float32)

        correct = tf.cast(tf.equal(y_true, pred_ids), tf.float32)
        correct = correct * mask

        return tf.reduce_sum(correct) / tf.reduce_sum(mask)

    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=5e-4,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=1.0  # Gradient clipping
    )

    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_accuracy]
    )
    return masked_accuracy, masked_loss


@app.cell
def _(EPOCHS_BASE, WORKSPACE_ROOT, dev_ds, model, os, tf, train_ds_base):
    os.makedirs(os.path.join(WORKSPACE_ROOT, "models"), exist_ok=True)

    # Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(WORKSPACE_ROOT, "models", "best_base_m2w.keras"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Training
    history_base = model.fit(
        train_ds_base,
        validation_data=dev_ds,
        epochs=EPOCHS_BASE,
        callbacks=[checkpoint_cb, early_stop_cb]
    )
    return


@app.cell
def _(
    EPOCHS_BT,
    WORKSPACE_ROOT,
    dev_ds,
    masked_accuracy,
    masked_loss,
    model,
    os,
    tf,
    train_ds_bt,
):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=masked_loss,
        metrics=[masked_accuracy]
    )

    checkpoint_bt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(WORKSPACE_ROOT, "models", "best_bt_m2w.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    early_stop_bt = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    history_bt = model.fit(
        train_ds_bt,
        validation_data=dev_ds,
        epochs=EPOCHS_BT,
        callbacks=[checkpoint_bt, early_stop_bt],
    )
    return


@app.cell
def _(WORKSPACE_ROOT, model, os):
    os.makedirs(os.path.join(WORKSPACE_ROOT, "models"), exist_ok=True)
    model.save(os.path.join(WORKSPACE_ROOT, "models", "transformer_bt_m2w.keras"))
    return


@app.cell
def _(
    EPOCHS_BT,
    PositionalEmbedding,
    WORKSPACE_ROOT,
    dev_ds,
    masked_accuracy,
    masked_loss,
    os,
    tf,
    train_ds_bt_filtered,
):
    baseline_path = os.path.join(WORKSPACE_ROOT, "models", "best_base_m2w.keras")
    model_filtered = tf.keras.models.load_model(
        baseline_path,
        compile=False,
        custom_objects={"PositionalEmbedding": PositionalEmbedding}
    )

    model_filtered.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=masked_loss,
        metrics=[masked_accuracy]
    )

    checkpoint_bt_filtered = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(WORKSPACE_ROOT, "models", "best_bt_filtered_m2w.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    early_stop_bt_filtered = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    history_bt_filtered = model_filtered.fit(
        train_ds_bt_filtered,
        validation_data=dev_ds,
        epochs=EPOCHS_BT,
        callbacks=[checkpoint_bt_filtered, early_stop_bt_filtered],
    )
    return


@app.cell
def _(
    END_ID,
    MAX_TGT_LEN,
    START_ID,
    model,
    src_vectorizer,
    tf,
    tgt_vectorizer,
):
    def translate(sentence, max_len=MAX_TGT_LEN):
        # input
        src_seq = src_vectorizer([sentence])
        # initialiser decoder
        dec_seq = [START_ID]

        for _ in range(max_len):
            # decoder input
            dec_input = dec_seq + [0] * (MAX_TGT_LEN - len(dec_seq))
            dec_input = tf.constant([dec_input], dtype=tf.int64)

            # prediction
            predictions = model({
                "encoder_inputs": src_seq,
                "decoder_inputs": dec_input
            }, training=False)

            # next token
            next_token_logits = predictions[0, len(dec_seq) - 1, :]
            next_token = tf.argmax(next_token_logits).numpy()

            # si next token est PAD ou END
            if next_token == END_ID or next_token == 0:
                break

            dec_seq.append(int(next_token))

        # id vers caractere
        tgt_vocab_list = tgt_vectorizer.get_vocabulary()
        result = []

        for token_id in dec_seq[1:]:  # sauter START token
            if token_id < len(tgt_vocab_list):
                char = tgt_vocab_list[token_id]
                if char and char != '':
                    result.append(char)

        return "".join(result)
    return (translate,)


@app.cell
def _(translate):
    translate("你好")
    translate("你今天吃饭了吗")
    translate("我不会说上海话")
    translate("今天天气真好")
    translate("请你帮我一下")
    translate("祝您开心")
    return


@app.cell
def _():
    import sacrebleu
    return (sacrebleu,)


@app.cell
def _(PositionalEmbedding, WORKSPACE_ROOT, os, tf):
    MBASE = os.path.join(WORKSPACE_ROOT, "models", "best_base_m2w.keras")
    MBT   = os.path.join(WORKSPACE_ROOT, "models", "best_bt_m2w.keras")
    MBT_FILTERED = os.path.join(WORKSPACE_ROOT, "models", "best_bt_filtered_m2w.keras")

    base_model = tf.keras.models.load_model(
        MBASE,
        compile=False,
        custom_objects={"PositionalEmbedding": PositionalEmbedding}
    )

    bt_model = tf.keras.models.load_model(
        MBT,
        compile=False,
        custom_objects={"PositionalEmbedding": PositionalEmbedding}
    )

    bt_filtered_model = tf.keras.models.load_model(
        MBT_FILTERED,
        compile=False,
        custom_objects={"PositionalEmbedding": PositionalEmbedding}
    )
    return base_model, bt_filtered_model, bt_model


@app.cell
def _(END_ID, MAX_TGT_LEN, START_ID, src_vectorizer, tf, tgt_vectorizer):
    def translate_avec_model(model_obj, sentence, max_len=MAX_TGT_LEN):
        src_seq = src_vectorizer([sentence])
        dec_seq = [START_ID]

        for _ in range(max_len):
            dec_input = dec_seq + [0] * (MAX_TGT_LEN - len(dec_seq))
            dec_input = tf.constant([dec_input], dtype=tf.int64)

            logits = model_obj(
                {"encoder_inputs": src_seq, "decoder_inputs": dec_input},
                training=False
            )

            next_logits = logits[0, len(dec_seq) - 1, :]
            next_id = int(tf.argmax(next_logits).numpy())

            if next_id == 0 or next_id == END_ID:
                break

            dec_seq.append(next_id)

        vocab = tgt_vectorizer.get_vocabulary()
        out = []
        for tid in dec_seq[1:]:
            if tid < len(vocab):
                ch = vocab[tid]
                if ch:
                    out.append(ch)

        return "".join(out)
    return (translate_avec_model,)


@app.cell
def _(base_model, bt_filtered_model, bt_model, test_df, translate_avec_model):
    src_sents = test_df["mandarin"].tolist()
    refs = test_df["wu"].tolist()

    base_hyps = [translate_avec_model(base_model, s) for s in src_sents]
    bt_hyps   = [translate_avec_model(bt_model,   s) for s in src_sents]
    bt_filtered_hyps = [translate_avec_model(bt_filtered_model, s) for s in src_sents]
    return base_hyps, bt_filtered_hyps, bt_hyps, refs


@app.cell
def _(base_hyps, bt_filtered_hyps, bt_hyps, refs, sacrebleu):
    bleu_base = sacrebleu.corpus_bleu(
        base_hyps,
        [refs],
        tokenize="char"
    ).score

    bleu_bt = sacrebleu.corpus_bleu(
        bt_hyps,
        [refs],
        tokenize="char"
    ).score

    bleu_bt_filtered = sacrebleu.corpus_bleu(
        bt_filtered_hyps,
        [refs],
        tokenize="char"
    ).score
    return bleu_base, bleu_bt, bleu_bt_filtered


@app.cell
def _(base_hyps, bt_filtered_hyps, bt_hyps, refs, sacrebleu):
    chrf_base = sacrebleu.corpus_chrf(
        base_hyps,
        [refs]
    ).score

    chrf_bt = sacrebleu.corpus_chrf(
        bt_hyps,
        [refs]
    ).score

    chrf_bt_filtered = sacrebleu.corpus_chrf(
        bt_filtered_hyps,
        [refs]
    ).score
    return chrf_base, chrf_bt, chrf_bt_filtered


@app.cell
def _(
    bleu_base,
    bleu_bt,
    bleu_bt_filtered,
    chrf_base,
    chrf_bt,
    chrf_bt_filtered,
    pd,
):
    metrics_df = pd.DataFrame([
        {"model": "baseline", "BLEU": bleu_base, "chrF": chrf_base},
        {"model": "+pseudo",  "BLEU": bleu_bt,   "chrF": chrf_bt},
        {"model": "+pseudo filtered", "BLEU": bleu_bt_filtered, "chrF": chrf_bt_filtered},
    ])

    metrics_df
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _(
    bleu_base,
    bleu_bt,
    bleu_bt_filtered,
    chrf_base,
    chrf_bt,
    chrf_bt_filtered,
    np,
    plt,
):
    labels = ["BLEU", "chrF"]
    base_vals = [bleu_base, chrf_base]
    bt_vals   = [bleu_bt,   chrf_bt]
    bt_filtered_vals = [bleu_bt_filtered, chrf_bt_filtered]

    x = np.arange(len(labels))
    w = 0.25

    plt.figure(figsize=(8, 4))
    plt.bar(x - w, base_vals, width=w, label="baseline")
    plt.bar(x, bt_vals, width=w, label="+pseudo")
    plt.bar(x + w, bt_filtered_vals, width=w, label="+pseudo filtered")
    plt.xticks(x, labels)
    plt.ylabel("score")
    plt.title("Test set evaluation (sacrebleu) - Comparison")
    plt.legend()
    plt.show()
    return


@app.cell
def _(base_hyps, bt_filtered_hyps, bt_hyps, pd, refs, test_df):
    sample_idx = test_df.sample(n=20, random_state=42).index.tolist()

    eval_table = pd.DataFrame({
        "mandarin (src)": [test_df.loc[i, "mandarin"] for i in sample_idx],
        "wu_ref (gold)":  [refs[i] for i in sample_idx],
        "wu_baseline":    [base_hyps[i] for i in sample_idx],
        "wu_pseudo":      [bt_hyps[i] for i in sample_idx],
        "wu_pseudo_filtered": [bt_filtered_hyps[i] for i in sample_idx],
    })

    eval_table
    return (eval_table,)


@app.cell
def _(WORKSPACE_ROOT, eval_table, os):
    OUT_PATH = os.path.join(WORKSPACE_ROOT, "eval_manual_20.csv")
    eval_table.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
