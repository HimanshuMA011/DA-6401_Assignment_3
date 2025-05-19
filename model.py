from functions import *

# ——— Utility: choose the right RNN layer ———
def get_rnn_layer(name, units, dropout, return_sequences, return_state):
    name = name.lower()
    if name == "rnn":
        return layers.SimpleRNN(units,
                                dropout=dropout,
                                return_sequences=return_sequences,
                                return_state=return_state)
    elif name == "gru":
        return layers.GRU(units,
                          dropout=dropout,
                          return_sequences=return_sequences,
                          return_state=return_state)
    elif name == "lstm":
        return layers.LSTM(units,
                           dropout=dropout,
                           return_sequences=return_sequences,
                           return_state=return_state)
    else:
        raise ValueError(f"Unsupported RNN type: {name}")

# ——— Optional Bahdanau Attention ———
class BahdanauAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V  = layers.Dense(1)

    def call(self, hidden_state, enc_output):
        # hidden_state: [batch, units]  or for LSTM, we concat [h, c]
        if isinstance(hidden_state, (list, tuple)):
            hidden = tf.concat(hidden_state, axis=-1)
        else:
            hidden = hidden_state
        # expand to [batch, 1, units*?]
        hidden_with_time = tf.expand_dims(hidden, 1)
        # score: [batch, seq_len, 1]
        score = self.V(tf.nn.tanh(self.W1(hidden_with_time) + self.W2(enc_output)))
        # attention_weights: [batch, seq_len, 1]
        attention_weights = tf.nn.softmax(score, axis=1)
        # context: [batch, units]
        context = tf.reduce_sum(attention_weights * enc_output, axis=1)
        return context, attention_weights

# ——— Encoder ———
class Encoder(tf.keras.Model):
    def __init__(self, layer_type, n_layers, units,
                 vocab_size, embedding_dim, dropout):
        super().__init__()
        self.layer_type = layer_type.lower()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)

        # create n_layers RNN layers, all return_sequences & return_state
        self.rnns = [
            get_rnn_layer(layer_type, units, dropout,
                          return_sequences=True,
                          return_state=True)
            for _ in range(n_layers)
        ]

    def call(self, x, hidden):
        # x: [batch, seq_len]
        x = self.embedding(x)  # → [batch, seq_len, emb_dim]

        # First layer: pass initial_state
        first_rnn = self.rnns[0]
        if self.layer_type == "lstm":
            # returns (out, h, c)
            out, h, c = first_rnn(x, initial_state=hidden)
            state = [h, c]
        else:
            # GRU or SimpleRNN: returns (out, h)
            out, h = first_rnn(x, initial_state=hidden)
            state = h  # single tensor

        # Remaining layers: only pass the sequence, ignore their states
        for rnn in self.rnns[1:]:
            # each returns (out, state...), but we only need out
            outputs = rnn(out)
            out = outputs[0]

        # out: [batch, seq_len, units], state: either tensor or [h, c]
        return out, state

    def initialize_hidden_state(self, batch_size):
        if self.layer_type == "lstm":
            # return two tensors [h0, c0]
            return [tf.zeros((batch_size, self.rnns[0].units)),
                    tf.zeros((batch_size, self.rnns[0].units))]
        else:
            # return single tensor h0
            return tf.zeros((batch_size, self.rnns[0].units))

# ——— Decoder ———
class Decoder(tf.keras.Model):
    def __init__(self, layer_type, n_layers, units,
                 vocab_size, embedding_dim, dropout,
                 attention=False):
        super().__init__()
        self.layer_type = layer_type.lower()
        self.attention = attention

        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        if attention:
            self.attn = BahdanauAttention(units)

        # For decoder we want: all but final layer return_sequences=True
        self.rnns = []
        for i in range(n_layers):
            return_seq = (i < n_layers - 1)
            self.rnns.append(
                get_rnn_layer(layer_type, units, dropout,
                              return_sequences=return_seq,
                              return_state=True)
            )

        # final projection
        self.fc = layers.Dense(vocab_size, activation="softmax")

    def call(self, x, hidden, enc_output=None):
        # x: [batch] or [batch, 1], we expect [batch, 1] time-step
        x = self.embedding(x)  # → [batch, 1, emb_dim]

        # apply attention to first layer only
        if self.attention and enc_output is not None:
            # hidden may be list (LSTM) or tensor (GRU/RNN)
            context, attn_w = self.attn(hidden, enc_output)
            # [batch, 1, emb_dim + units*?]
            x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        else:
            attn_w = None

        # First RNN layer with initial state
        first_rnn = self.rnns[0]
        if self.layer_type == "lstm":
            out, h, c = first_rnn(x, initial_state=hidden)
            state = [h, c]
        else:
            out, h = first_rnn(x, initial_state=hidden)
            state = h  # single tensor

        # Remaining layers (no initial_state)
        for rnn in self.rnns[1:]:
            outputs = rnn(out)
            out = outputs[0]

        # out shape: [batch, units]  if last layer return_sequences=False
        logits = self.fc(out)  # → [batch, vocab_size]
        return logits, state, attn_w


class BeamSearch():
    def __init__(self, model, k):
        self.k = k
        self.model = model
        self.acc = tf.keras.metrics.Accuracy()

    def sample_beam_search(self, probs):

        m, n = probs.shape
        output_sequences = [[[], 0.0]]

        for row in probs:
            beams = []

            for tup in output_sequences:
                seq, score = tup
                for j in range(n):
                    new_beam = [seq + [j], score - tf.math.log(row[j])]
                    beams.append(new_beam)

            output_sequences = sorted(beams, key=lambda x: x[1])[:self.k]

        tensors, scores = list(zip(*output_sequences))
        tensors = list(map(lambda x: tf.expand_dims(tf.constant(x),0), tensors))

        return tf.concat(tensors, 0), scores

    def beam_accuracy(self, input, target):
        accs = []

        for i in range(self.k):
            self.acc.reset_state()
            self.acc.update_state(target, input[i, :])
            accs.append(self.acc.result())

        return max(accs)

    def step(self, input, target, enc_state):

        batch_acc = 0
        sequences = []

        enc_out, enc_state = self.model.encoder(input, enc_state)

        dec_state = enc_state
        dec_input = tf.expand_dims([self.model.targ_tokenizer.word_index["\t"]]*self.model.batch_size ,1)

        for t in range(1, target.shape[1]):

            preds, dec_state, _ = self.model.decoder(dec_input, dec_state, enc_out)

            sequences.append(preds)
            preds = tf.argmax(preds, 1)
            dec_input = tf.expand_dims(preds, 1)

        sequences = tf.concat(list(map(lambda x: tf.expand_dims(x, 1), sequences)), axis=1)

        for i in range(target.shape[0]):

            possibilities, scores = self.sample_beam_search(sequences[i, :, :])
            batch_acc += self.beam_accuracy(possibilities, target[i, 1:])

        batch_acc = batch_acc / target.shape[0]

        return 0, batch_acc

    def evaluate(self, test_dataset, batch_size=None, upto=5, use_wandb=False):

        if batch_size is not None:
            self.model.batch_size = batch_size
            test_dataset = test_dataset.batch(batch_size)
        else:
            self.model.batch_size = 1

        test_acc = 0
        enc_state = self.model.encoder.initialize_hidden_state(self.model.batch_size)

        for batch, (input, target) in enumerate(test_dataset.take(upto)):

           _, acc = self.step(input, target, enc_state)
           test_acc += acc

        if use_wandb:
            wandb.log({"test acc (beam search)": test_acc / upto})

        print(f"Test Accuracy on {upto*batch_size} samples: {test_acc / upto:.4f}\n")

    def translate(self, word):

        word = "\t" + word + "\n"
        sequences = []
        result = []

        inputs = self.model.input_tokenizer.texts_to_sequences([word])
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                               maxlen=self.model.max_input_len,
                                                               padding="post")


        enc_state = self.model.encoder.initialize_hidden_state(1)
        enc_out, enc_state = self.model.encoder(inputs, enc_state)

        dec_state = enc_state
        dec_input = tf.expand_dims([self.model.targ_tokenizer.word_index["\t"]]*1, 1)

        for t in range(1, self.model.max_target_len):

            preds, dec_state, _ = self.model.decoder(dec_input, dec_state, enc_out)

            sequences.append(preds)
            preds = tf.argmax(preds, 1)
            dec_input = tf.expand_dims(preds, 1)

        sequences = tf.concat(list(map(lambda x: tf.expand_dims(x, 1), sequences)), axis=1)

        possibilities, scores = self.sample_beam_search(tf.squeeze(sequences, 0))
        output_words = self.model.targ_tokenizer.sequences_to_texts(possibilities.numpy())

        def post_process(word):
            word = word.split(" ")[:-1]
            return "".join([x for x in word])

        output_words = list(map(post_process, output_words))

        return output_words, scores
    
class Seq2SeqModel():
    def __init__(self, embedding_dim, encoder_layers, decoder_layers, layer_type, units, dropout, attention=False):
        self.embedding_dim = embedding_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers


        self.layer_type = layer_type
        self.units = units
        self.dropout = dropout
        self.attention = attention
        self.stats = []
        self.batch_size = 128
        self.use_beam_search = False

    def build(self, loss, optimizer, metric):
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric

    def set_vocabulary(self, input_tokenizer, targ_tokenizer):
        self.input_tokenizer = input_tokenizer
        self.targ_tokenizer = targ_tokenizer
        self.create_model()

    def create_model(self):

        encoder_vocab_size = len(self.input_tokenizer.word_index) + 1
        decoder_vocab_size = len(self.targ_tokenizer.word_index) + 1

        self.encoder = Encoder(self.layer_type, self.encoder_layers, self.units, encoder_vocab_size,
                               self.embedding_dim, self.dropout)

        self.decoder = Decoder(self.layer_type, self.decoder_layers, self.units, decoder_vocab_size,
                               self.embedding_dim,  self.dropout, self.attention)

        # # when building:
        # self.encoder = Encoder(layer_type, encoder_layers, units,
        #                        encoder_vocab_size, embedding_dim, dropout)
        # self.decoder = Decoder(layer_type, decoder_layers, units,
        #                        decoder_vocab_size, embedding_dim, dropout,
        #                        attention=self.attention)

    @tf.function
    def train_step(self, input, target, enc_state):

        loss = 0

        with tf.GradientTape() as tape:

            enc_out, enc_state = self.encoder(input, enc_state)

            dec_state = enc_state
            dec_input = tf.expand_dims([self.targ_tokenizer.word_index["\t"]]*self.batch_size ,1)

            ## We use Teacher forcing to train the network
            ## Each target at timestep t is passed as input for timestep t + 1

            if random.random() < self.teacher_forcing_ratio:

                for t in range(1, target.shape[1]):

                    preds, dec_state, _ = self.decoder(dec_input, dec_state, enc_out)
                    loss += self.loss(target[:,t], preds)
                    self.metric.update_state(target[:,t], preds)

                    dec_input = tf.expand_dims(target[:,t], 1)

            else:

                for t in range(1, target.shape[1]):

                    preds, dec_state, _ = self.decoder(dec_input, dec_state, enc_out)
                    loss += self.loss(target[:,t], preds)
                    self.metric.update_state(target[:,t], preds)

                    preds = tf.argmax(preds, 1)
                    dec_input = tf.expand_dims(preds, 1)


            batch_loss = loss / target.shape[1]

            variables = self.encoder.variables + self.decoder.variables
            gradients = tape.gradient(loss, variables)

            self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss, self.metric.result()

    @tf.function
    def validation_step(self, input, target, enc_state):

        loss = 0

        enc_out, enc_state = self.encoder(input, enc_state)

        dec_state = enc_state
        dec_input = tf.expand_dims([self.targ_tokenizer.word_index["\t"]]*self.batch_size ,1)

        for t in range(1, target.shape[1]):

            preds, dec_state, _ = self.decoder(dec_input, dec_state, enc_out)
            loss += self.loss(target[:,t], preds)
            self.metric.update_state(target[:,t], preds)

            preds = tf.argmax(preds, 1)
            dec_input = tf.expand_dims(preds, 1)

        batch_loss = loss / target.shape[1]

        return batch_loss, self.metric.result()


    def fit(self, dataset, val_dataset, batch_size=128, epochs=15, use_wandb=False, teacher_forcing_ratio=1.0):

        self.batch_size = batch_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

        run_name = (f"ed_{wandb.config.embedding_dim}_layers_{wandb.config.enc_dec_layers}_type_{wandb.config.layer_type}_"
                    f"units_{wandb.config.units}_dropout_{wandb.config.dropout}_attention_{wandb.config.attention}_"
                    f"beam_{wandb.config.beam_width}_tf_{wandb.config.teacher_forcing_ratio}")
        wandb.run.name = run_name

        steps_per_epoch = len(dataset) // self.batch_size
        steps_per_epoch_val = len(val_dataset) // self.batch_size

        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        val_dataset = val_dataset.batch(self.batch_size, drop_remainder=True)

        # useful when we need to translate the sentence
        sample_inp, sample_targ = next(iter(dataset))
        self.max_target_len = sample_targ.shape[1]
        self.max_input_len = sample_inp.shape[1]

        template = "\nTrain Loss: {0:.4f} Train Accuracy: {1:.4f} Validation Loss: {2:.4f} Validation Accuracy: {3:.4f}"

        print("-"*100)
        for epoch in range(0, epochs):
            print(f"EPOCH {epoch}\n")

            ## Training loop ##
            total_loss = 0
            total_acc = 0
            self.metric.reset_state()

            starting_time = time.time()
            enc_state = self.encoder.initialize_hidden_state(self.batch_size)

            print("Training ...\n")
            for batch, (input, target) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss, acc = self.train_step(input, target, enc_state)
                total_loss += batch_loss
                total_acc += acc


                if batch==0 or ((batch + 1) % 100 == 0):
                    print(f"Batch {batch+1} Loss {batch_loss:.4f}")

            avg_acc = total_acc / steps_per_epoch
            avg_loss = total_loss / steps_per_epoch

            # Validation loop ##
            total_val_loss = 0
            total_val_acc = 0
            self.metric.reset_state()

            enc_state = self.encoder.initialize_hidden_state(self.batch_size)

            print("\nValidating ...")
            for batch, (input, target) in enumerate(val_dataset.take(steps_per_epoch_val)):
                batch_loss, acc = self.validation_step(input, target, enc_state)
                total_val_loss += batch_loss
                total_val_acc += acc

            avg_val_acc = total_val_acc / steps_per_epoch_val
            avg_val_loss = total_val_loss / steps_per_epoch_val

            print(template.format(avg_loss, avg_acc*100, avg_val_loss, avg_val_acc*100))

            time_taken = time.time() - starting_time
            self.stats.append({"epoch": epoch,
                            "train_loss": avg_loss,
                            "dev_loss": avg_val_loss,
                            "train_acc": avg_acc,
                            "dev_acc": avg_val_acc,
                            # "training time": time_taken
                                      })

            if use_wandb:

                wandb.log(self.stats[-1])

            print(f"\nTime taken for the epoch {time_taken:.4f}")
            print("-"*100)

        print("\nModel trained successfully !!")

    def evaluate(self, test_dataset, batch_size=None):

        if batch_size is not None:
            self.batch_size = batch_size

        steps_per_epoch_test = len(test_dataset) // batch_size
        test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

        total_test_loss = 0
        total_test_acc = 0
        self.metric.reset_state()

        enc_state = self.encoder.initialize_hidden_state(self.batch_size)

        print("\nRunning test dataset through the model...\n")
        for batch, (input, target) in enumerate(test_dataset.take(steps_per_epoch_test)):
            batch_loss, acc = self.validation_step(input, target, enc_state)
            total_test_loss += batch_loss
            total_test_acc += acc

        avg_test_acc = total_test_acc / steps_per_epoch_test
        avg_test_loss = total_test_loss / steps_per_epoch_test

        print(f"Test Loss: {avg_test_loss:.4f} Test Accuracy: {avg_test_acc:.4f}")

        return avg_test_loss, avg_test_acc


    def translate(self, word, get_heatmap=False):

        word = "\t" + word + "\n"

        inputs = self.input_tokenizer.texts_to_sequences([word])
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                               maxlen=self.max_input_len,
                                                               padding="post")

        result = ""
        att_wts = []

        enc_state = self.encoder.initialize_hidden_state(1)
        enc_out, enc_state = self.encoder(inputs, enc_state)

        dec_state = enc_state
        dec_input = tf.expand_dims([self.targ_tokenizer.word_index["\t"]]*1, 1)

        for t in range(1, self.max_target_len):

            preds, dec_state, attention_weights = self.decoder(dec_input, dec_state, enc_out)

            if get_heatmap:
                att_wts.append(attention_weights)

            preds = tf.argmax(preds, 1)
            next_char = self.targ_tokenizer.index_word[preds.numpy().item()]
            result += next_char

            dec_input = tf.expand_dims(preds, 1)

            if next_char == "\n":
                return result[:-1], att_wts[:-1]

        return result[:-1], att_wts[:-1]

    def plot_attention_heatmap(self, word, ax, font_path="/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"):

        translated_word, attn_wts = self.translate(word, get_heatmap=True)
        attn_heatmap = tf.squeeze(tf.concat(attn_wts, 0), -1).numpy()

        input_word_len = len(word)
        output_word_len = len(translated_word)

        ax.imshow(attn_heatmap[:, :input_word_len])

        font_prop = FontProperties(fname=font_path, size=18)

        ax.set_xticks(np.arange(input_word_len))
        ax.set_yticks(np.arange(output_word_len))

        ax.set_xticklabels(list(word))
        ax.set_yticklabels(list(translated_word), fontproperties=font_prop)
