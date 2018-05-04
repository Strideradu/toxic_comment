import os
import pandas as pd
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K

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
        pass
        #loss = logs.get('val_loss')
        #k = 0.1
        #exp_loss = np.exp(k*loss)
        #lr_0 = K.get_value(self.model.optimizer.lr)
        #lr_1 = lr_0 * exp_loss
        #K.set_value(self.model.optimizer.lr, lr_1)
        #print("\nLearning Rate: " + str(lr_1))

MODEL_DIR = '../model/'
DATA_DIR = '../data/'
RESULT_DIR = '../result/'
EMBED_SIZE = 128
MAX_FEATURES = 2000
MAX_SENTENCE_LEN = 200
OUTPUT_SIZE = 60
try:
    os.system("mkdir -p " + MODEL_DIR)
    os.system("mkdir -p " + RESULT_DIR)
except:
    print("Please manually create folder 'model' and 'result'")
    pass

# callback1
# update learning rate (not used yet)
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
        MODEL_DIR+'lstm.current.hdf5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')


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
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
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

# Input layer
input_layer = Input(shape=(MAX_SENTENCE_LEN, ))
# Embedding layer
x = Embedding(MAX_FEATURES, EMBED_SIZE)(input_layer)
# LSTM layer
lstm_out_size = 100
x = LSTM(lstm_out_size, return_sequences=True)(x)
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

batch_size = 64
epochs = 4

# train start from the last checkpoint
model.load_weights(MODEL_DIR+'lstm.current.hdf5')

model.fit(
        train_tokens,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        callbacks = [callback_model,
            early_stop_model,
            checkpoint_model],
        validation_split=0.1)

# load the current best model
model.load_weights(MODEL_DIR+'lstm.current.hdf5')

test_labels = model.predict(test_tokens)

submission = pd.read_csv(DATA_DIR + 'sample_submission.csv')
submission[list_classes] = test_labels

submission.to_csv(RESULT_DIR + "lstm_result.csv", index=False)

#model.summary()
