from data_loader import msr_vtt_dataset
from encoder import ResTDconvE
from decoder import TDconvD
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import uuid
from nltk.translate.bleu_score import sentence_bleu
import os 
from torch.nn.utils import clip_grad_norm_

ID=uuid.uuid4()

def train(args):

	path="../checkpoints/4th_ckp_34a476c5-59d9-4b5e-857b-9262a2a2c729"
	new_path="../checkpoints/4th_ckp_34a476c5-59d9-4b5e-857b-9262a2a2c729_new"

	train_meta=msr_vtt_dataset(args.train_vocab,args.image_dir,"train",args.batch_size)
	#load checkpoint
	checkpoint = torch.load(path)
	checkpoint['args']=args
	checkpoint['w2i']=train_meta.w2i
	checkpoint['i2w']=train_meta.i2w
	
	#torch.save({'epoch': i_epoch+1,'args':args,'w2i':train_meta.w2i,'i2w':train_meta.i2w,'encoder': encoder.state_dict(),'decoder':decoder.state_dict(),'optimizer': optimizer.state_dict(),'in_BLEU@4':in_BLEU,'out_BLEU@4':out_BLEU }, os.path.join(args.ckp_dir,f'{i_epoch+1}th_ckp_{ID}'))
	torch.save(checkpoint,new_path)

if __name__=="__main__":
	
	parser = argparse.ArgumentParser(description='train TDconvED')
	parser.add_argument('--image_dir',default='../data/msr_vtt',help='directory for sampled images')
	parser.add_argument('--train_vocab',default='../data/msr_vtt/train_vocab.json',help='vocabulary file for training data')
	parser.add_argument('--test_vocab',default='../data/msr_vtt/test_vocab.json',help='vocabulary file for testing data')
	parser.add_argument('--batch_size',type=int,default=16,help='batch size')
	parser.add_argument('--encoder_dim',type=int,default=256,help='dimension for TDconvEncoder')
	parser.add_argument('--decoder_dim',type=int,default=256,help='dimension for TDconvDecoder')
	parser.add_argument('--encoder_layer',type=int,default=2,help='layer of TDconvEncoder')
	parser.add_argument('--decoder_layer',type=int,default=2,help='layer of TDconvDecoder')
	parser.add_argument('--embed_dim',type=int,default=256,help='dimension for word embedding')
	parser.add_argument('--attend_dim',type=int,default=256,help='dimension for attention')
	parser.add_argument('--device',type=str,default='cuda:0',help='default to cuda:0 if gpu available else cpu')
	parser.add_argument('--epoch',type=int,default=10,help='total epochs to train.')
	parser.add_argument('--lr',type=float,default=0.0001,help='learning rate for optimizer.')
	parser.add_argument('--log_dir',default='../logs/',help='directory for storing log files')
	parser.add_argument('--ckp_dir',default='../checkpoints/',help='directory for storing checkpoints.')
	parser.add_argument('--ckp_path',default='',help='the path to a checkpoint to be loaded to continue your training.')
	parser.add_argument('--BLEU_eval_ratio',type=float,default=0.005,help='proportion of data used to test model. 1 would use all the data to evaluate our model. But it will take a long time.')


	args = parser.parse_args()

	if not os.path.exists(args.ckp_dir):
		os.mkdir(args.ckp_dir)
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)

	train(args)
