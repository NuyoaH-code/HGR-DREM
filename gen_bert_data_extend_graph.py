import numpy as np
import os
import random
import json
import pickle
from nltk.tokenize import WordPunctTokenizer
import argparse
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer

#from sklearn.externals import joblib   #解决pickle不能存储大数据的问题
tokenizer = BertTokenizer.from_pretrained('./bert/bert-base-uncased/', do_lower_case=True)

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type = str, default =  "data")
# parser.add_argument('--out_path', type = str, default = "prepro_data_bert/new")
parser.add_argument('--out_path', type = str, default = "new_data")

# rank_path = './result/GraphCNN_multihead_bert_afterpretrain_hop2_rank_checkpointtrain_rank.pkl'
# rank_list = pickle.load(open(rank_path,'rb'))

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
case_sensitive = False

char_limit = 16
train_distant_file_name = os.path.join(in_path, 'train_distant.json')
train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')

rel2id = json.load(open(os.path.join(out_path, 'rel2id.json'), "r"))
id2rel = {v:u for u,v in rel2id.items()}
json.dump(id2rel, open(os.path.join(out_path, 'id2rel.json'), "w"))
fact_in_train = set([])
fact_in_dev_train = set([])
pronoun_list = []
with open ('pronoun_list.txt','r') as f:
	for line in f:
		pronoun_list.append(line.strip().lower())

relation_type = {(2, 2): 13535, 
                 (4, 2): 3673,
				 (1, 2): 3146, 
				 (5, 4): 2269, 
				 (4, 1): 1955, 
				 (4, 3): 1848, 
				 (5, 1): 1698, 
				 (4, 4): 1552,
				 (5, 3): 1448, 
				 (5, 2): 1402, 
				 (5, 5): 1367, 
				 (4, 5): 1117, 
				 (1, 1): 860, 
				 (1, 4): 480, 
				 (2, 1): 458, 
				 (2, 4): 391, 
				 (1, 3): 357, 
				 (1, 5): 260,
				 (2, 5): 212, 
				 (2, 3): 152}

def init(data_file_name, rel2id, max_length = 512, is_training = True, suffix=''):
	count_all_label_rel = 0


	max_exist_sentence_num = 0
	max_sentence_length = 0
	max_coexist_sentence_num = 0
	max_coexist_sentence_length = 0

	ori_data = json.load(open(data_file_name))

	char2id = json.load(open(os.path.join(out_path, "char2id.json")))
	word2id = json.load(open(os.path.join(out_path, "word2id.json")))
	ner2id = json.load(open(os.path.join(out_path, "ner2id.json")))

	# saving new data
	print("Saving files")
	if is_training:
		name_prefix = "train"
	else:
		name_prefix = "dev"

	Ma = 0
	Ma_e = 0
	data = []
	unk_num = 0
	k_num = 0
	#intrain = notintrain = notindevtrain = indevtrain = 0
	storage_size = 10000
	for i_t in tqdm(range(len(ori_data))):
		i = i_t
		item = {}
		Ls = [0]
		L = 0
		document = []
		for x in ori_data[i]['sents']:
			document.extend(x)
			L += len(x)
			Ls.append(L)   #start position of sentence  句子开始位置
		all_sentence_num = len(Ls) -1
		#document_id = []
		document_char_id = []
		document_bert = ['[CLS]']
		index_id = []
		for j, word in enumerate(document):
			word = word.lower()
			start = len(document_bert)
			new_word = tokenizer.tokenize(word)
			document_bert.extend(new_word)
			end = len(document_bert)
			index_id.append((start,end))
			# if word in word2id:
			# 	document_id.append(word2id[word])
			# else:
			# 	document_id.append(word2id['UNK'])              #未知词代替所有未登录词
			for w in new_word:
				word_char_id = np.zeros((char_limit))
				for c_idx, k in enumerate(list(w)):              #处理字符向量
					if c_idx>=char_limit:
						break
					word_char_id[c_idx] = char2id.get(k, char2id['UNK'])
				document_char_id.append (word_char_id)  
		#document_bert = document_bert[:511]
		document_bert.append('[SEP]')
		document_id = tokenizer.convert_tokens_to_ids(document_bert)      
		item['document'] = document_id
		item['document_char'] = document_char_id

		title = ori_data[i]['title']
		title_id = []
		title_char_id = []
		for j, word in enumerate(title):
			word = word.lower()
			if word in word2id:
				title_id.append(word2id[word])
			else:
				title_id.append(word2id['UNK'])              #未知词代替所有未登录词      
			word_char_id = np.zeros((char_limit))
			for c_idx, k in enumerate(list(word)):              #处理字符向量
				if c_idx>=char_limit:
					break
				word_char_id[c_idx] = char2id.get(k, char2id['UNK'])
			title_char_id.append (word_char_id)
        
		item['title'] = title
		item['title_char'] = title_char_id
		articleGraph = nx.DiGraph()          #创建联通图，无向图
		vertexSet =  ori_data[i]['vertexSet']   #第i篇文档的实体集合
		max_exist_sentence_num_i =0

		node_sent_dict = {}		#用于记录每个结点表示的实体在document中出现的句子
		for j in range(len(vertexSet)):       #遍历实体
			sent_list = []
			articleGraph.add_node(j,exist_sentence = [],exist_sentence_id = [],exist_pos=[],type = [])         #创建图中的实体节点
			if len(vertexSet[j]) > max_exist_sentence_num:            #一个实体最多出现在多少个句子中，实际上是mention数量
				max_exist_sentence_num = len(vertexSet[j])
			if len(vertexSet[j]) > max_exist_sentence_num_i:            #一个实体最多出现在多少个句子中，实际上是mention数量，本图中
				max_exist_sentence_num_i = len(vertexSet[j])
			for k in range(len(vertexSet[j])):              #遍历第J个实体的所有mention
				if ner2id[vertexSet[j][k]['type']] not in articleGraph.nodes[j]['type']:
					articleGraph.nodes[j]['type'].append (ner2id[vertexSet[j][k]['type']])
				vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])
				sent_list.append(int(vertexSet[j][k]['sent_id']))
				sent_id = vertexSet[j][k]['sent_id']
				dl = Ls[sent_id]
				pos1 = vertexSet[j][k]['pos'][0]
				pos2 = vertexSet[j][k]['pos'][1]
				vertexSet[j][k]['pos'] = (index_id[pos1+dl][0], index_id[pos2+dl-1][1])    #句子中的位置改为整个文档中的位置
				articleGraph.nodes[j]['exist_sentence'].append((index_id[Ls[sent_id]][0],index_id[Ls[sent_id+1]-1][1]))          #出现在第几个句子中,句子的开头和结尾位置
				articleGraph.nodes[j]['exist_sentence_id'].append(sent_id)
				# articleGraph.nodes[j]['exist_sentence'].append(sent_id)          #出现在第几个句子中
				articleGraph.nodes[j]['exist_pos'].append((index_id[pos1+dl][0],index_id[pos2+dl-1][1]))           #出现在整个文档的哪个位置
				if len(ori_data[i]['sents'][sent_id]) > max_sentence_length:         #最大的句子长度
					max_sentence_length = len(ori_data[i]['sents'][sent_id])
			node_sent_dict[j] = sent_list
		articleGraph.graph['max_entity_exist_num'] = max_exist_sentence_num_i
		articleGraph.graph['all_sentence_num'] = all_sentence_num
		# print('node_sent_dict',node_sent_dict)
		assert len(vertexSet) == len(node_sent_dict)
		item['node_sent_dict'] = node_sent_dict
		document_length = len(document_id)
		document_pos = np.zeros ((document_length))
		document_ner = np.zeros ((document_length))
		for idx, vertex in enumerate(vertexSet, 1):
			for v in vertex:
				document_pos[v['pos'][0]:v['pos'][1]] = idx                    #实体位置
				document_ner[v['pos'][0]:v['pos'][1]] = ner2id[v['type']]      #实体类别标签
		item['document_pos'] = document_pos
		item['document_ner'] = document_ner
		#ori_data[i]['vertexSet'] = vertexSet
		item['vertexSet'] = vertexSet
		# print('vertexSet',vertexSet)
		labels = ori_data[i].get('labels', [])    #三元组集合
		count_all_label_rel += len(labels)
		item['labels'] = labels
		# train_triple = set([])
		# new_labels = []
		label_matrix = np.zeros((len(vertexSet),len(vertexSet),len(rel2id)))
		label_evidence = {}
		for label in labels:
			rel = label['r']
			evidence = label['evidence']
			assert(rel in rel2id)
			label['r'] = rel2id[label['r']]        #关系对应的id
			label_matrix[label['h'], label['t'],label['r']] = 1     #正例对应的位置是label id else 0 (Na)
			evidence_label = np.zeros(all_sentence_num)
			for e in evidence:
				evidence_label[e] = 1
			label_evidence[(label['h'], label['t'],label['r'])] = evidence_label
		for h_i in range (len(vertexSet)):
			for t_j in range (len(vertexSet)):
				label_sum = label_matrix[h_i,t_j,:]
				if label_sum.sum() == 0:
					label_matrix[h_i,t_j,0] = 1
		item['label_matrix'] = label_matrix
	    
		#遍历所有节点,创建图中的边(在同一个句子当中就建立一条边)
		max_coexist_sentence_length_i = 0
		max_coexist_sentence_num_i =0
		label_mask = []

		for node_j in articleGraph.nodes():
			for node_k in articleGraph.nodes():
				if node_j != node_k:
					common_sentences = []
					common_sentences_poses = []
					common_next_sentences = []
					common_next_sentences_poses = []
					sents_j = articleGraph.nodes[node_j]['exist_sentence']
					poses_j = articleGraph.nodes[node_j]['exist_pos']
					sents_k = articleGraph.nodes[node_k]['exist_sentence']
					poses_k = articleGraph.nodes[node_k]['exist_pos']
					for pos_j,sent_j in zip(poses_j,sents_j):
						for pos_k,sent_k in zip(poses_k,sents_k):
							if sent_j == sent_k:
								common_sentences.append(sent_j)
								common_sentences_poses.append(pos_j+pos_k)
								if sent_j[1]-sent_j[0] > max_coexist_sentence_length:         #最大的句子长度
									max_coexist_sentence_length = sent_j[1]-sent_j[0]
								if sent_j[1]-sent_j[0] > max_coexist_sentence_length_i:         #最大的句子长度，本图中
									max_coexist_sentence_length_i = sent_j[1]-sent_j[0]
								# if len(ori_data[i]['sents'][sent_j]) > max_coexist_sentence_length:         #最大的句子长度
								#  	max_coexist_sentence_length = len(ori_data[i]['sents'][sent_j])
							if sent_j[1] == sent_k[0]:   # sent j is in front of sent k
								flag = False
								for word in document_bert[sent_j[0]:sent_k[1]]:
									if word.lower() in pronoun_list:
										flag = True
										break
								if flag:
									common_next_sentences.append((sent_j[0],sent_k[1]))
									common_next_sentences_poses.append(pos_j+pos_k)
							if sent_j[0] == sent_k[1]:   # sent j is in front of sent k
								flag = False
								for word in document_bert[sent_k[0]:sent_j[1]]:
									if word.lower() in pronoun_list:
										flag = True
										break
								if flag:
									common_next_sentences.append((sent_k[0],sent_j[1]))
									common_next_sentences_poses.append(pos_j+pos_k)

					if common_sentences != []:
						articleGraph.add_edge(node_j,node_k,sentences = common_sentences,position = common_sentences_poses)
						if len(common_sentences) > max_coexist_sentence_num:              #统计共同出现的句子个数（实际上可能两个实体在同一个句子中多次出现，算多个句子）
							max_coexist_sentence_num = len(common_sentences)
						if len(common_sentences) > max_coexist_sentence_num_i:              #统计共同出现的句子个数（实际上可能两个实体在同一个句子中多次出现，算多个句子，本图中）
							max_coexist_sentence_num_i = len(common_sentences)
					elif common_next_sentences != []:
						articleGraph.add_edge(node_j,node_k,sentences = common_next_sentences,position = common_next_sentences_poses)
						if len(common_next_sentences) > max_coexist_sentence_num:              #统计共同出现的句子个数（实际上可能两个实体在同一个句子中多次出现，算多个句子）
							max_coexist_sentence_num = len(common_next_sentences)
						if len(common_next_sentences) > max_coexist_sentence_num_i:              #统计共同出现的句子个数（实际上可能两个实体在同一个句子中多次出现，算多个句子，本图中）
							max_coexist_sentence_num_i = len(common_next_sentences)
					
					types_j = articleGraph.nodes[node_j]['type']
					types_k = articleGraph.nodes[node_k]['type']
					flag = False
					for type_j in types_j:
						for type_k in types_k:
							if (type_j,type_k) in relation_type:
								label_mask.append((node_j,node_k))
								flag = True
								break
						if flag:
							break

		articleGraph.graph['max_sentence_length'] = max_coexist_sentence_length_i
		articleGraph.graph['max_sentence_num'] = max_coexist_sentence_num_i
		edge_num = len(articleGraph.edges)
		if not edge_num:
			nx.draw(articleGraph)
			plt.savefig('./graph_fig/'+ name_prefix + suffix + '_zeroEdge_'+str(i) +'.jpg')
			plt.show()
			plt.close()
		#item['na_triple'] = na_triple
		item['Ls'] = Ls                               #每个句子start位置
		#item['sents'] = ori_data[i]['sents']
		item['graph'] = articleGraph
		item['label_mask'] = label_mask
		item['label_evidence'] = label_evidence
		data.append(item)

		Ma = max(Ma, len(vertexSet))                 #最多有多少个实体
		#Ma_e = max(Ma_e, len(ori_data[i]['labels']))        #最多右多少对关系
		if (i_t+1)%storage_size ==0 or i_t == len(ori_data)-1:
			# pickle.dump(data, open (os.path.join(out_path, name_prefix + suffix + '_low2high_' + str(int(i_t/storage_size)) + '.pkl'), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
			pickle.dump(data, open (os.path.join(out_path, name_prefix + suffix + str(int(i_t/storage_size)) + '.pkl'), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
			data = []

	print('count_all_label_rel',count_all_label_rel)
	print ('num:',unk_num,k_num)
	print ('data_len:', len(ori_data))               #打印初始的文档个数
	print("Finish saving")
	print ('实体最多mention数：%d' %max_exist_sentence_num)
	print ('最长句子长度为：%d' %max_sentence_length)
	print ('实体对最多的mention数：%d' %max_coexist_sentence_num)
	print ('实体对所在句子长度最长为：%d' %max_coexist_sentence_length)

init(train_distant_file_name, rel2id, max_length = 512, is_training = True, suffix='')
init(train_annotated_file_name, rel2id, max_length = 512, is_training = False, suffix='_train')
init(dev_file_name, rel2id, max_length = 512, is_training = False, suffix='_dev')
init(test_file_name, rel2id, max_length = 512, is_training = False, suffix='_test')


