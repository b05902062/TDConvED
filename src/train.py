from data_loader import msr_vtt_dataset
from encoder import ResTDconvE
from decoder import TDconvD
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import uuid
from nltk.translate.bleu_score import sentence_bleu

def train(args):
	
	output=open("./"+"_".join([str(args.lr),str(args.encoder_dim),str(args.decoder_dim),str(args.embed_dim)])+f"_{uuid.uuid4()}.txt","w+") 
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	#training
	train_meta=msr_vtt_dataset(args.train_vocab,args.image_dir,"train",args.batch_size)
	train=DataLoader(train_meta,batch_size=1,shuffle=False)

	#BLEU evaluation
	BLEU_train_meta=msr_vtt_dataset(args.train_vocab,args.image_dir,"train",1)
	BLEU_test_meta=msr_vtt_dataset(args.test_vocab,args.image_dir,"test",1)
	BLEU_train=DataLoader(BLEU_train_meta,batch_size=1,shuffle=False)
	BLEU_test=DataLoader(BLEU_test_meta,batch_size=1,shuffle=False)
	
	encoder = ResTDconvE(args).to(device)
	decoder = TDconvD(args.embed_dim, args.decoder_dim, args.encoder_dim, len(train_meta.w2i),device).to(device)

	criterion = nn.CrossEntropyLoss(ignore_index=0)
	params = list(decoder.parameters()) + list(encoder.parameters())
	optimizer = torch.optim.Adam(params, lr=args.lr)

	for i_epoch in range(args.epoch):
		for i_b,(images,sen_in,lengths) in enumerate(train):
			if i_b%500==1:
				break	
			images=images.squeeze(0).to(device)
			sen_in=sen_in.squeeze(0).to(device)
			#images batch*25*3*256*256 5d tensor.
			#sen_in is a 2d tensor of size batch*(max_len of this batch) containing word index.
			#len is a list(len=batch) of int(length of each sentence in this batch including <sos> and <eos>).

			features=encoder(images)	
			#feature is of size batch*g_sample*encoder_dim.
			outputs=decoder(features,sen_in,lengths)
			#feature is of size batch*(max_label_len-1)*vocab_size
			loss=criterion(outputs.reshape(-1,outputs.shape[2]),sen_in[:,1:].reshape(-1))
			if i_b%1 ==0:
				print(f"Epoch: {i_epoch+1}/{args.epoch} , Batch: {i_b+1}/{len(train)} Loss: {loss}")
				output.write(f"Epoch: {i_epoch+1}/{args.epoch} , Batch: {i_b+1}/{len(train)} Loss: {loss}\n")
			encoder.zero_grad()
			decoder.zero_grad()
			loss.backward()
			optimizer.step()

			if i_b==0:
				predict=decoder.predict(features)
				print(sen_in[:,1:],outputs.max(dim=2)[1])
				print(get_sentence(predict,train_meta))
		#calculate BLEU@4 score.
		in_BLEU=0
		for i_b,(images,_,_) in enumerate(BLEU_train):
			if i_b%500==1:
				break	
			images=images.squeeze(0).to(device)
			video_id=BLEU_train_meta.video_id[i_b][0]
			#images 1*25*3*256*256 5d tensor.
			#video_id is a string(name of this video).
			features=encoder(images)
			predict=decoder.predict(features)
			#predict is 2d tensor of size 1*max_predict.
			hypo=get_sentence(predict,train_meta)[0]
			#hypo is a list(len=# of words in this sentence, 0 to max_predict-1) of string. 
			#print("predict",predict,"hypo",hypo,"\n",train_meta.ref[video_id])
			in_BLEU+=sentence_bleu(train_meta.ref[video_id],hypo)
			
		out_BLEU=0
		for i_b,(images,_,_) in enumerate(BLEU_test):
			if i_b%500==1:
				break	
			images=images.squeeze(0).to(device)
			video_id=BLEU_test_meta.video_id[i_b][0]
			#images 1*25*3*256*256 5d tensor.
			#video_id is a string(name of this video).
			features=encoder(images)
			predict=decoder.predict(features)
			#predict is 2d tensor of size 1*max_predict.
			hypo=get_sentence(predict,train_meta)[0]
			#hypo is a list(len=# of words in this sentence) of string. 
			out_BLEU+=sentence_bleu(BLEU_test_meta.ref[video_id],hypo)
		
		print(f"Epoch: {i_epoch+1}/{args.epoch} , train BLEU@4: {in_BLEU/len(BLEU_train)} , test BLEU@4: {out_BLEU/30}")#len(BLEU_test)}")
		output.write(f"Epoch: {i_epoch+1}/{args.epoch} , train BLEU@4: {in_BLEU/len(BLEU_train)} , test BLEU@4: {out_BLEU/30}")#len(BLEU_test)}")




def get_sentence(sen_in,train_meta):
	#sen_in of size batch*sentence_len. sentence_len includes <pad>, <sos>,<eos> etc.
	sen=[]
	for s in sen_in:
		temp=[]
		for i in s:
			i=i.item()
			if i==train_meta.w2i["<eos>"]:
				break
			temp.append(i)
		#temp can be of length 1 to max_predict now. First one is always <sos>. There won't be <eos> in it now.
		sen.append([train_meta.i2w[str(i)] for i in temp if (i!=train_meta.w2i["<sos>"] and i!=train_meta.w2i['<pad>'])])

	#sen is a list(len=batch) of list(len=# of words in this sentence, from 0 to max_predict-1) of string.	
	return sen

if __name__=="__main__":
	
	parser = argparse.ArgumentParser(description='train TDconvED')
	parser.add_argument('--image_dir',default='../data/msr_vtt',help='directory for sampled images')
	parser.add_argument('--train_vocab',default='../data/msr_vtt/train_vocab.json',help='vocabulary file for training data')
	parser.add_argument('--test_vocab',default='../data/msr_vtt/test_vocab.json',help='vocabulary file for testing data')
	parser.add_argument('--batch_size',type=int,default=1,help='batch size')
	parser.add_argument('--encoder_dim',type=int,default=32,help='dimension for TDconvE')
	parser.add_argument('--decoder_dim',type=int,default=32,help='dimension for TDconvD')
	parser.add_argument('--embed_dim',type=int,default=32,help='dimension for word embedding')
	parser.add_argument('--device',type=str,default='cuda:0',help='default to cuda:0 if gpu available else cpu')
	parser.add_argument('--epoch',type=int,default=100,help='total epochs to train.')
	parser.add_argument('--lr',type=float,default=0.001,help='learning rate for optimizer.')
	#parser.add_argument('--attention_size',type=int,default=256,help='dimension for attention')



	args = parser.parse_args()
	train(args)
