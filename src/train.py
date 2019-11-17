from data_loader import msr_vtt_dataset
from encoder import ResTDconvE
from decoder import TDconvD
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import uuid


def train(args):
	
	output=open("./"+"_".join([str(args.lr),str(args.encoder_dim),str(args.decoder_dim),str(args.embed_dim)])+f"_{uuid.uuid4()}.txt","w+") 
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	msr_vtt=msr_vtt_dataset(args.data_dir,"train",args.batch_size)
	data = DataLoader(msr_vtt,batch_size=1,shuffle=True)
	
	encoder = ResTDconvE(args).to(device)
	decoder = TDconvD(args.embed_dim, args.decoder_dim, args.encoder_dim, len(msr_vtt.w2i)).to(device)

	criterion = nn.CrossEntropyLoss(ignore_index=0)
	params = list(decoder.shifted_conv1.parameters())+list(decoder.shifted_conv2.parameters())+list(decoder.linear2.parameters())+list(decoder.linear1.parameters()) + list(encoder.parameters())
	optimizer = torch.optim.Adam(params, lr=args.lr)

	for i_epoch in range(args.epoch):
		for i_b,(images,sen_in,lengths) in enumerate(data):
			
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
				print(f"Epoch: {i_epoch+1}/{args.epoch} , Batch: {i_b+1}/{len(data)} Loss: {loss}")
				output.write(f"Epoch: {i_epoch+1}/{args.epoch} , Batch: {i_b+1}/{len(data)} Loss: {loss}\n")
			encoder.zero_grad()
			decoder.zero_grad()
			loss.backward()
			optimizer.step()





if __name__=="__main__":
	
	parser = argparse.ArgumentParser(description='train TDconvED')
	parser.add_argument('--data_dir',default='./data',help='directory for sampled images')
	parser.add_argument('--batch_size',type=int,default=16,help='batch size')
	parser.add_argument('--encoder_dim',type=int,default=32,help='dimension for TDconvE')
	parser.add_argument('--decoder_dim',type=int,default=32,help='dimension for TDconvD')
	parser.add_argument('--embed_dim',type=int,default=32,help='dimension for word embedding')
	parser.add_argument('--device',type=str,default='cuda:0',help='default to cuda:0 if gpu available else cpu')
	parser.add_argument('--epoch',type=int,default=100,help='total epochs to train.')
	parser.add_argument('--lr',type=float,default=0.001,help='learning rate for optimizer.')
	#parser.add_argument('--attention_size',type=int,default=256,help='dimension for attention')



	args = parser.parse_args()
	train(args)
