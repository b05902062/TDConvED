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
	
	output=open(os.path.join(args.log_dir,"_".join([str(args.lr),str(args.encoder_dim),str(args.decoder_dim),str(args.embed_dim)])+f"_{ID}.txt"),"w") 
	

	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	print("using",device)

	#training
	train_meta=msr_vtt_dataset(args.train_vocab,args.image_dir,"train",args.batch_size,shuffle=True)
	train=DataLoader(train_meta,batch_size=1,shuffle=True)

	#BLEU evaluation
	BLEU_train_meta=msr_vtt_dataset(args.train_vocab,args.image_dir,"train",1)
	BLEU_test_meta=msr_vtt_dataset(args.test_vocab,args.image_dir,"test",1)
	BLEU_train=DataLoader(BLEU_train_meta,batch_size=1,shuffle=False)#don't modify batch and shuffle here.

	BLEU_test=DataLoader(BLEU_test_meta,batch_size=1,shuffle=False)#don't modify batch and shuffle here.

	
	encoder = ResTDconvE(args).to(device)
	
	#fix pretrained resnet
	for p in encoder.video_encoder.frame_encode.resnet.parameters():
		p.requires_grad=False

	decoder = TDconvD(args.embed_dim, args.decoder_dim, args.encoder_dim,args.attend_dim, len(train_meta.w2i),device,layer=args.decoder_layer).to(device)

	criterion = nn.CrossEntropyLoss(ignore_index=0)
	params = list(decoder.parameters()) + list(encoder.parameters())
	optimizer = torch.optim.Adam([p for p in params if p.requires_grad is True], lr=args.lr)
	start=0

	if args.ckp_path!='':
		#load checkpoint
		checkpoint = torch.load(os.path.join(args.ckp_path),map_location=args.device)
		encoder.load_state_dict(checkpoint['encoder'])
		decoder.load_state_dict(checkpoint['decoder'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		start=checkpoint['epoch']
		



	for i_epoch in range(start,args.epoch):
		for i_b,(images,sen_in,_) in enumerate(train):
			images=images.squeeze(0).to(device)
			sen_in=sen_in.squeeze(0).to(device)
			#images batch*25*3*256*256 5d tensor.
			#sen_in is a 2d tensor of size batch*(max_len of this batch) containing word index.

			features=encoder(images)	
			#feature is of size batch*g_sample*encoder_dim.
			outputs=decoder(features,sen_in)
			#outputs is of size batch*(max_label_len-1)*vocab_size
			loss=criterion(outputs.reshape(-1,outputs.shape[2]),sen_in[:,1:].reshape(-1))
			if i_b%args.loss_every ==0:
				print(f"Epoch: {i_epoch+1}/{args.epoch} , Batch: {i_b+1}/{len(train)} Loss: {loss}")
				output.write(f"Epoch: {i_epoch+1}/{args.epoch} , Batch: {i_b+1}/{len(train)} Loss: {loss}\n")
			encoder.zero_grad()
			decoder.zero_grad()
			loss.backward()
			#prevent exploding gradient.
			clip_grad_norm_([p for p in params if p.requires_grad is True],1)
			optimizer.step()
			
			#set batch to 1 when using.
			#use to debug. if predict is correct. Given the same word token it would produce the same word as decode at the position.
			"""
			if 1 or i_b==0:
				encoder.eval()
				decoder.eval()
				features=encoder(images)	
				outputs=decoder(features,sen_in,lengths)				
				print("answer, second word to last",sen_in[:,1:])
				print("decode, second to last",outputs.max(dim=2)[1])
				print("decode, second to last",get_sentence(outputs.max(dim=2)[1],train_meta.w2i,train_meta.i2w))
				(predict,prob)=decoder.predict(features,beam_size=1)
				print("predict, first to last",predict[0])
				print("predict,first to last",get_sentence(predict[0],train_meta.w2i,train_meta.i2w))
				print(get_BLEU(images,encoder,decoder,train_meta,BLEU_train_meta,i_b))
				encoder.train()
				decoder.train()
			"""
		#calculate BLEU@4 score.
		encoder.eval()
		decoder.eval()
		in_BLEU=0
		in_count=0
		out_BLEU=0
		out_count=0
		for i_b,(images,_,_) in enumerate(BLEU_train):
			#print(i_b,"/",int(len(BLEU_train)*args.BLEU_eval_ratio))
			if i_b%20!=0:
				#each video has 20 sentences. they would produce the same score. So we only calculate for one time.
				continue
			if i_b>int(len(BLEU_train)*args.BLEU_eval_ratio):
				break
			images=images.squeeze(0).to(device)
			in_BLEU+=get_BLEU(images,encoder,decoder,train_meta,BLEU_train_meta,i_b)
			in_count+=1

		for i_b,(images,_,_) in enumerate(BLEU_test):
			#print(i_b,"/",int(len(BLEU_test)*args.BLEU_eval_ratio))
			if i_b%20!=0:
				#each video has 20 sentences. they would produce the same score. So we only calculate for one time.
				continue
			if i_b>int(len(BLEU_test)*args.BLEU_eval_ratio):
				break
			images=images.squeeze(0).to(device)
			out_BLEU+=get_BLEU(images,encoder,decoder,train_meta,BLEU_test_meta,i_b)
			out_count+=1

		in_BLEU=in_BLEU/in_count
		out_BLEU=out_BLEU/out_count
		encoder.train()
		decoder.train()

		print(f"Epoch: {i_epoch+1}/{args.epoch} , train BLEU@4: {in_BLEU} , test BLEU@4: {out_BLEU}")
		output.write(f"Epoch: {i_epoch+1}/{args.epoch} , train BLEU@4: {in_BLEU} , test BLEU@4: {out_BLEU}\n")
		torch.save({'epoch': i_epoch+1,'args':args,'w2i':train_meta.w2i,'i2w':train_meta.i2w,'encoder': encoder.state_dict(),'decoder':decoder.state_dict(),'optimizer': optimizer.state_dict(),'in_BLEU@4':in_BLEU,'out_BLEU@4':out_BLEU }, os.path.join(args.ckp_dir,f'{i_epoch+1}th_ckp_{ID}'))


def get_BLEU(images,encoder,decoder,train_meta,score_meta,index):
		#images 1*g_sample*3*256*256 5d tensor.
		#video_id is a string(name of this video).

		video_id=score_meta.video_id[index][0]
		features=encoder(images)
		predict=decoder.predict(features)[0][0]#first sentence
		#predict is 2d tensor of size 1*max_predict.
		hypo=get_sentence(predict,train_meta.w2i,train_meta.i2w)[0]
		#hypo is a list(len=# of words in this sentence, 0 to max_predict-1) of string. 
		#score_meta.ref[video_id] is a list(len=total # of reference sentences for this video) of list(len=# of words in a string) of strings.
		####print("predict",predict,"\nhypo",hypo,"\nref",score_meta.ref[video_id])
		#return single sentence BLEU@4 score.
		return sentence_bleu(score_meta.ref[video_id],hypo)


def get_sentence(sen_in,w2i,i2w):
	#sen_in is 2d array of size batch*sentence_len. sentences includes <pad>, <sos>,<eos>,<unk> etc.
	sen=[]
	for s in sen_in:
		temp=[]
		for i in s:
			i=i.item()
			if i==w2i["<eos>"]:
				break
			temp.append(i)
		#temp can be of length 1 to max_predict now. First one is always <sos>. There won't be <eos> in it now.
		sen.append([i2w[str(i)] for i in temp if (i!=w2i["<sos>"] and i!=w2i['<pad>'])])

	#sen is a list(len=batch) of list(len=# of words in this sentence, from 0 to max_predict-1) of string.	
	return sen

if __name__=="__main__":
	
	parser = argparse.ArgumentParser(description='train TDconvED')
	parser.add_argument('--loss_every',type=int,default=20,help='print loss every loss_every batches.')
	parser.add_argument('--image_dir',default='../data/msr_vtt',help='directory for sampled images')
	parser.add_argument('--train_vocab',default='../data/msr_vtt/train_vocab.json',help='vocabulary file for training data')
	parser.add_argument('--test_vocab',default='../data/msr_vtt/test_vocab.json',help='vocabulary file for testing data')
	parser.add_argument('--batch_size',type=int,default=16,help='batch size')
	parser.add_argument('--encoder_dim',type=int,default=512,help='dimension for TDconvEncoder')
	parser.add_argument('--decoder_dim',type=int,default=512,help='dimension for TDconvDecoder')
	parser.add_argument('--encoder_layer',type=int,default=2,help='layer of TDconvEncoder')
	parser.add_argument('--decoder_layer',type=int,default=2,help='layer of TDconvDecoder')
	parser.add_argument('--embed_dim',type=int,default=512,help='dimension for word embedding')
	parser.add_argument('--attend_dim',type=int,default=512,help='dimension for attention')
	parser.add_argument('--device',type=str,default='cuda:0',help='default to cuda:0 if gpu available else cpu')
	parser.add_argument('--epoch',type=int,default=10,help='total epochs to train.')
	parser.add_argument('--lr',type=float,default=0.0001,help='learning rate for optimizer.')
	parser.add_argument('--log_dir',default='../logs/',help='directory for storing log files')
	parser.add_argument('--ckp_dir',default='../checkpoints/',help='directory for storing checkpoints.')
	parser.add_argument('--ckp_path',default='',help='the path to a checkpoint to be loaded to continue your training.')
	parser.add_argument('--BLEU_eval_ratio',type=float,default=1,help='proportion of data used to test model. 1 would use all the data to evaluate our model. But it will take a long time.')


	args = parser.parse_args()

	if not os.path.exists(args.ckp_dir):
		os.mkdir(args.ckp_dir)
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)

	train(args)
