import sys
import os
import rnn
import pickle
import glob
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 15

def get_indices(test_label):
	index_1 = []
	index_2 = []
	index_3 = []
	for index, label in enumerate(test_label):
		label = list(label)
		if(label == [1,0,0]):
			index_1.append(index)
		elif(label == [0,1,0]):
			index_2.append(index)
		else:
			index_3.append(index)
	return [index_1 , index_2 , index_3]

def get_pre_ans(predict , index , answer):
	num = 0
	for label in predict[index]:
		if(list(label) == answer):
			num += 1
	return num

def evaluate(predict , test_label):
	#eval precision and recall
	#precision:
	[index_1 , index_2 , index_3] = get_indices(test_label)
	num_1 = len(index_1)
	num_2 = len(index_2)
	num_3 = len(index_3)
	print('real answer:' , num_1 , num_2 , num_3)
	[num_1_pre , num_2_pre , num_3_pre] = get_indices(predict)
	print('predict answer:' , len(num_1_pre) , len(num_2_pre) , len(num_3_pre))
	num_1_ans = get_pre_ans(predict , index_1 , [1,0,0])
	num_2_ans = get_pre_ans(predict , index_2 , [0,1,0])
	num_3_ans = get_pre_ans(predict , index_3 , [0,0,1])
	print('rigth answer:' , num_1_ans , num_2_ans , num_3_ans)

	#calculate macro-f1 score and miro-f1 score
	precision_1 = num_1_ans / len(num_1_pre)
	precision_2 = num_2_ans / len(num_2_pre)
	precision_3 = num_3_ans / len(num_3_pre)
	recall_1 = num_1_ans / num_1
	recall_2 = num_2_ans / num_2
	recall_3 = num_3_ans / num_3

	precision_macro = (precision_1 + precision_2 + precision_3) / 3
	recall_macro = (recall_1 + recall_2 + recall_3) / 3
	macro_f1 = 2 * (precision_macro * recall_macro / (precision_macro + recall_macro))

	precision_micro = (num_1_ans + num_2_ans + num_3_ans) / (len(num_1_pre) + len(num_2_pre) + len(num_3_pre))
	recall_micro = (num_1_ans + num_2_ans + num_3_ans) / (num_1 + num_2 + num_3)
	micro_f1 = 2 * (precision_micro * recall_micro / (precision_micro + recall_micro))

	print('micro_f1:' , micro_f1)
	print('macro_f1:' , macro_f1)

	return

def label(predict):
	for i , ans in enumerate(predict):
		index = np.argmax(ans)
		if(index == 0):
			predict[i] = [1,0,0]
		elif(index == 1):
			predict[i] = [0,1,0]
		else:
			predict[i] = [0,0,1]
	return predict

def main():
	#get best weight
	list_of_files = glob.glob('./model/rnn/*')
	latest_file = max(list_of_files , key = os.path.getctime)
	model = load_model(latest_file)

	with open('data/tokenizer.pickle' , 'rb') as handle:
		tokenizer = pickle.load(handle)

	#get test json
	test_file = "data/test_set.json"
	test_dic = rnn.get_json(test_file)
	[test_snippet , test_label] = rnn.get_snippet_label(test_dic)

	#word sequence to word ids
	sequences = tokenizer.texts_to_sequences(test_snippet )
	test_data = pad_sequences(sequences ,maxlen = MAX_SEQUENCE_LENGTH)
	
	#predict 
	predict = model.predict(test_data)
	predict = label(predict)

	#evaluate
	evaluate(predict , test_label)


if __name__ == "__main__":
	main()