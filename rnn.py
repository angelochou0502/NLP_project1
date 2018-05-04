import json 
import re
import numpy as np
import pickle
import os
import sys
import keras
from keras.layers.core import Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.layers import Embedding , Input, Conv1D , GlobalMaxPooling1D, Flatten , Bidirectional , LSTM
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping , ModelCheckpoint

MAX_SEQUENCE_LENGTH = 15
GLOVE_DIR = "/Users/angelocsc/Desktop/NTU/NLPLab/glove.6B/" #define the directory store glove
EMBEDDING_DIM = 100

def get_json(train_file):
	with open(train_file , 'r') as f:
		train_dics = json.load(f)
	return train_dics

def preprocess(sentence):
	words = sentence.split()
	word_list = []
	for word in words:
		if(word[0] == '#' or word[0] == '@' or word[0] == '$'):
			continue
		if(re.search(r'\d', word)):
			continue
		else:
			word_list.append(word)
	sentence = " ".join(word_list)
	return sentence

def classify(score): #bullish（看漲）:[1,0,0]  bearish（下跌）:[0,1,0]   natural:[0,0,1]
	score = float(score)
	if(score > 0.2):
		return [1,0,0]
	elif(score < -0.2):
		return [0,1,0]
	else:
		return [0,0,1]

def get_snippet_label(train_dics):
	train_snippet = []
	train_label = []
	for train_dic in train_dics:
		snippet = train_dic["snippet"]
		label = classify(train_dic["sentiment"])
		if(isinstance(snippet,list)):
			for sentence in snippet:
				train_snippet.append(preprocess(sentence))
				train_label.append(label)
		else:
			train_snippet.append(preprocess(snippet))
			train_label.append(label)
	train_snippet = np.array(train_snippet)
	train_label = np.array(train_label)
	return [train_snippet , train_label]

def create_rnn_model(embedding_layer):
	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype = 'int32')
	embedded_sequences = embedding_layer(sequence_input)
	x = Bidirectional((LSTM(100 , return_sequences = True)) , merge_mode = 'sum')(embedded_sequences)
	x = Bidirectional((LSTM(100 , return_sequences = True)) , merge_mode = 'sum')(x)
	x = GlobalMaxPooling1D()(x)
	#x = Flatten()(x)
	#x = Dense(128, activation='relu')(x)
	preds = Dense( 3, activation='softmax')(x)

	model = Model(sequence_input, preds)
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['acc'])

	'''#create model to see the output 
				get_1st_layer_output = K.function([model.layers[0].input],[model.layers[1].output])
				layer_output = get_1st_layer_output([embedded_sequences])[0]'''
	return model 

def get_embeddinglayer(word_index):
	embeddings_index = {}
	f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        # words not found in embedding index will be all-zeros.
	        embedding_matrix[i] = embedding_vector
	embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM,weights=[embedding_matrix],\
								input_length=MAX_SEQUENCE_LENGTH, trainable=False)
	return embedding_layer

def main():
	#get tarining json 
	train_file = "data/training_set.json"
	train_dics = get_json(train_file)

	#get train data and train label
	[train_snippet , train_label] = get_snippet_label(train_dics)

	#train_snippet -> word id
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(train_snippet)
	sequences = tokenizer.texts_to_sequences(train_snippet ) # len > 15 : only 7 ; len = 0 : 49
	word_index = tokenizer.word_index
	#save tokenizer
	with open('data/tokenizer.pickle' , 'wb') as handle:
		pickle.dump(tokenizer , handle)

	#pad to the same length
	train_data = pad_sequences(sequences ,maxlen = MAX_SEQUENCE_LENGTH)
	print('Shape of data tensor:', train_data.shape)
	print('Shape of label tensor:', train_label.shape)

	# split the data into a training set and a validation set
	indices = np.arange(train_data.shape[0])
	np.random.shuffle(indices)
	data = train_data[indices]
	labels = train_label[indices]
	nb_validation_samples = int(0.1 * data.shape[0])

	x_train = data[:-nb_validation_samples]
	y_train = labels[:-nb_validation_samples]
	x_val = data[-nb_validation_samples:]
	y_val = labels[-nb_validation_samples:]

	#preparing embedding layer with word_index
	embedding_layer = get_embeddinglayer(word_index)

	#create rnn model
	rnn_model = create_rnn_model(embedding_layer)
	esCallBack = EarlyStopping(monitor='val_loss' , min_delta= 0 , patience = 5 , verbose = 0 , mode = 'auto')
	os.makedirs('./model/%s' %(sys.argv[1]))
	chCallBack = ModelCheckpoint('./model/%s/weights.{epoch:02d}-{val_loss:.4f}.h5' %(sys.argv[1]) , monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	rnn_model.fit(x_train , y_train , validation_data = (x_val , y_val) , epochs = 15 , batch_size = 10 , callbacks = [esCallBack , chCallBack])
	#rnn_model.save('model/rnn.h5')

if __name__ == "__main__":
	main()