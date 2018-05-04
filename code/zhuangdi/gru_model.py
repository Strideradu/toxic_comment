import os
import pandas as pd
import numpy as np

from keras.layers import Dense, Embedding, Input, MaxPooling1D,AveragePooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Bidirectional, Dropout, CuDNNGRU, Flatten
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import Concatenate

from keras.models import Model

from keras.optimizers import RMSprop

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers.merge import Concatenate

from keras.models import Model

from tools.my_embedding import loadEmbeddingMatrix as loadEmbedding

MODEL_DIR = '../model/'
DATA_DIR = '../data/'
RESULT_DIR = '../result/'
submission_file = "gru.2.result.csv"
MAX_FEATURES = 100000
MAX_SENTENCE_LEN = 500
OUTPUT_SIZE = 60

model_name = 'gru.1.hdf5'
model_name = 'gru.2.hdf5'
batch_size = 256
epochs = 4
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

class CallBackModel(Callback):
    #def __init__(self):
    #    super(Callback, self).__init__()

    def on_train_begin(self, logs={}):
        self.losses = []
        lr = K.get_value(self.model.optimizer.lr)
        #uncomment the following line to set initial learning rate
        #lr = 0.1
        K.set_value(self.model.optimizer.lr, lr)
        print('Original Learning Rate:' + str(lr))

    def on_epoch_end(self, batch, logs={}):
        ''' for dynamic learning rate changes '''
        loss = logs.get('val_loss')
        k = 0.01
        exp_loss = np.exp(k*loss)
        lr_0 = K.get_value(self.model.optimizer.lr)
        lr_1 = lr_0 * exp_loss
        K.set_value(self.model.optimizer.lr, lr_1)
        print("\nLearning Rate: " + str(lr_1))

def check_embedding_exist(loadType='word2vec'):
    file_names = os.listdir(DATA_DIR)
    for name in file_names:
        if "embedding" in name and loadType in name:
            return DATA_DIR + name
    return None

def get_GRU_GlobalMax_Model(embedding_matrix,train_labels, train_tokens, test_tokens, call_backs):
    dropout_rate = 0.3
    recurrent_units = 64
    dense_size = 256
    input_layer = Input(shape=(MAX_SENTENCE_LEN, ))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    x = Concatenate(axis=1)([x1, x2])
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(6, activation="sigmoid")(x)
    #output_layer = Dense(6, activation="rmsprop")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['accuracy'])
    try:
        model.load_weights(MODEL_DIR+model_name)
        print("Train from existing models")
    except:
        print("Train from stratch...")

    model.fit(
            train_tokens,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=call_backs,
            validation_split=0.1)

    # load the current best model
    model.load_weights(MODEL_DIR+model_name)
    return model
    pass


def get_GRU_model(embedding_matrix,train_labels, train_tokens, test_tokens, call_backs):
    dropout_rate = 0.3
    recurrent_units = 64
    dense_size = 256
    input_layer = Input(shape=(MAX_SENTENCE_LEN, ))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    x1 = MaxPooling1D(4)(x)
    x2 = AveragePooling1D(4)(x)
    x = Concatenate(axis=1)([x1, x2])
    x = Flatten()(x)
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(6, activation="sigmoid")(x)
    #output_layer = Dense(6, activation="rmsprop")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['accuracy'])

    try:
        model.load_weights(MODEL_DIR+model_name)
        print("Train from existing models")
    except:
        print("Train from stratch...")

    model.fit(
            train_tokens,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=call_backs,
            validation_split=0.1)

    # load the current best model
    model.load_weights(MODEL_DIR+model_name)
    return model

def train_lstm_model(train_tokens, train_labels, vocab_size, embedding_matrix, call_backs):
    # Input layer
    input_layer = Input(shape=(MAX_SENTENCE_LEN, ))
    # Embedding layer
    #x = Embedding(MAX_FEATURES, 128)(input_layer)
    x = Embedding(
            vocab_size,
            embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=False)(input_layer)
    # Bidirectional LSTM layer
    lstm_out_size = 100
    #x = LSTM(lstm_out_size, return_sequences=True)(x)
    x = Bidirectional(
            LSTM(lstm_out_size,
                return_sequences=True,
                name='lstm_layer',
                dropout=0.1,
                recurrent_dropout=0.1))(x)
    # Maxpooling layer
    x = GlobalMaxPool1D()(x)
    # Dropout layer, randomly dropout neurons with p probability
    p1 = 0.1
    x = Dropout(p1)(x)
    # Dense layer
    dense_out_size = 50
    x = Dense(dense_out_size, activation="relu")(x)
    # Another dropout layer
    p2 = 0.1
    x = Dropout(p2)(x)
    # Dense layer, the final output layer
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
    # train start from the last checkpoint

    try:
        model.load_weights(MODEL_DIR+model_name)
        print("Train from existing models")
    except:
        print("Train from stratch...")

    model.fit(
            train_tokens,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=call_backs,
            validation_split=0.1)

    # load the current best model
    model.load_weights(MODEL_DIR+model_name)

    return model

def train_embedding_matrix(loadType):
    loadType = 'word2vec'
    embedding_file = check_embedding_exist(loadType)
    embedding_matrix = None
    if embedding_file:
        print("load pretrained embedding matrix ...")
        print(embedding_file)
        embedding_matrix = np.loadtxt(embedding_file, dtype=float)
        #with open(embedding_file) as input_file:
         #   embedding_matrix = json.load(input_file)
    else:
        print("Training Embedding Matrix...")
        embedding_matrix = loadEmbedding(tokenizer,loadType)
    print("Embedding Matrix is ready for use.")
    return embedding_matrix

def generate_call_backs():
    # callback1
    # update learning rate
    callback_model = CallBackModel()

    # callback2
    # stop training when validation loss is not improving
    early_stop_model = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=0,
            verbose=0,
            mode='auto')

    # callback3
    # save the current best model to a hdf5 file
    checkpoint_model = ModelCheckpoint(
            MODEL_DIR+model_name,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min')
    call_backs = [callback_model, early_stop_model, checkpoint_model]
    return call_backs

def initiate_data_set():
    # read dataset
    train_f = pd.read_csv(DATA_DIR+'train.csv')
    test_f = pd.read_csv(DATA_DIR+'test.csv')

    # replace null values with empty string
    train_f.fillna(' ', inplace=True)
    test_f.fillna(' ', inplace=True)

    #print (len(train_f))
    #print (train_f.head())
    #print (train_f.isnull().any(),test_f.isnull().any())

    # extract labels and sentences(texts).
    train_labels = train_f[list_classes].values

    train_texts = train_f["comment_text"]
    test_texts = test_f["comment_text"]

    # tokenize each word to an index
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(train_texts))

    # tokenizer.word_counts: dictionary of <word, count>
    # tokenizer.word_index: dictionary of <word, index>

    # convert sentences into lists of indexes
    # train_tokens is a list of lists,
    # and each nested list corresponds to a sentence.
    train_tokens = tokenizer.texts_to_sequences(train_texts)
    test_tokens = tokenizer.texts_to_sequences(test_texts)

    # pad training & testing data to make each sentence in the same length
    train_tokens = pad_sequences(train_tokens,maxlen=MAX_SENTENCE_LEN)
    test_tokens = pad_sequences(test_tokens,maxlen=MAX_SENTENCE_LEN)

    vocab_size = len(tokenizer.word_index),
    return (train_tokens, test_tokens, train_labels, vocab_size)

def read_dataset():
    embed_path = DATA_DIR + 'embeddings.npz'
    data = np.load(embed_path)
    embedding_matrix = data['arr_0']

    train_path = DATA_DIR + 'train.npz'
    data = np.load(train_path)
    train_tokens = data['arr_0']

    test_path = DATA_DIR + 'test.npz'
    data = np.load(test_path)
    test_tokens = data['arr_0']

    label_path = DATA_DIR + 'label.npz'
    data = np.load(label_path)
    train_labels = data['arr_0']
    print("Data Ready for Use.")

    return (embedding_matrix, train_tokens, test_tokens, train_labels)


def main():
    try:
        os.system("mkdir -p " + MODEL_DIR)
        os.system("mkdir -p " + RESULT_DIR)
    except:
        print("Please manually create folder 'model' and 'result'")
        return

    #train_tokens, test_tokens, train_labels, vocab_size = initiate_data_set()
    embedding_matrix, train_tokens, test_tokens, train_labels = read_dataset()

    # callback functions are used to adjust learning rate, set early_stop strategy, etc.
    call_backs = generate_call_backs()

    model = get_GRU_GlobalMax_Model(embedding_matrix, train_labels, train_tokens, test_tokens, call_backs)

    #model = get_GRU_model(embedding_matrix, train_labels, train_tokens, test_tokens, call_backs)
    '''
    # train the lstm model
    model = train_lstm_model(
            train_tokens,
            train_labels,
            vocab_size,
            embedding_matrix,
            call_backs)
    '''

    test_labels = model.predict(test_tokens)

    submission = pd.read_csv(DATA_DIR + 'sample_submission.csv')
    submission[list_classes] = test_labels

    submission.to_csv(RESULT_DIR + submission_file, index=False)

    #model.summary()

if __name__ == "__main__":
    main()
