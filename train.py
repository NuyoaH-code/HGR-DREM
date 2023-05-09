import config
# from config import Config
# from config import Config_bert
import models
import numpy as np
import os
import time
import datetime
import json
# from sklearn.metrics import average_precision_score
import sys
import os
import argparse



parser = argparse.ArgumentParser()
#训练BERT
parser.add_argument('--model_name', type = str, default = 'GraphCNN_multihead_bert_gate_cls', help = 'name of the model')
parser.add_argument('--save_name', type = str, default = 'GCGCN_BERT',help = 'save name for trained model')

parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')		##'dev_dev' when using dev dataset
#训练Glove
# parser.add_argument('--model_name', type = str, default = 'HDR_glove', help = 'name of the model')
# parser.add_argument('--save_name', type = str, default = 'HDR_GloVe1',help = 'save name for trained model')
#
# parser.add_argument('--train_prefix', type = str, default = 'dev_train')
# parser.add_argument('--test_prefix', type = str, default = 'dev_dev')		##'dev_dev' when using dev dataset
  

args = parser.parse_args()
model = {
	'GraphCNN_multihead_bert_gate_cls': models.GraphCNN_multihead_bert_gate_cls,		#model used bert
	'GCGCN_glove':models.GCGCN_glove,		#model used glove
}
#使用glove
# con = config.Config.Config(args)
# con.set_max_epoch(40)
# con.load_train_data()
# con.load_test_data()
# con.train(model[args.model_name], args.save_name)

#使用BERT
con =config.Config_bert.Config(args)
con.set_max_epoch(40)
con.load_train_data()
con.load_test_data()
con.gen_train_facts_anno()
con.gen_train_facts_distant()
con.train(model[args.model_name], args.save_name)
