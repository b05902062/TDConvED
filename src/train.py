from data_loader import msr_vtt_dataset
form encoder import TDconvE
from decoder import TDconvD
import torch
import torch.nn as nn
import argparse

def train(args):
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	msr_vtt=msr_vtt_dataset(args.data_dir,split="train",args.batch_size)
	data = DataLoader(msr_vtt,batch_size=1,shuffle=True)
	
	encoder = TDconvE(args.encode_dim).to(device)
	decoder = TDconvD(args.embed_size, args.hidden_size, len(vocab)).to(device)

	criterion = nn.CrossEntropyLoss(ignore_index=0)
	params = list(decoder.parameters()) + list(encoder.parameters())
	optimizer = torch.optim.Adam(params, lr=args.learning_rate)

	for i_epoch in range(args.epoch):
		for i_b,images,sen_in,len in tqdm(enumerate(data))
			
			images=images.unsqueeze(0).to(device)
			sen_in=sen_in.unsqueeze(0).to(device)

			#images batch*25*3*256*256 5d tensor.
			#sen_in is a 2d tensor of size batch*(max_len of this batch) containing word index.
			#len is a list(len=batch) of int(length of each sentence in this batch including <sos> and <eos>).

			features=encoder(images)	








if __name__=="__main__":
	
	parser = argparse.ArgumentParser(description='download and sample images')
	parser.add_argument('--data_dir',default='./data',help='directory for sampled images')
	parser.add_argument('--batch_size',type=int,default=16,help='batch size')
	parser.add_argument('--encode_size',type=int,default=256,help='dimension for TDconvE')
	parser.add_argument('--decode_size',type=int,default=256,help='dimension for TDconvD')
	parser.add_argument('--',type=int,default=256,help='dimension for TDconvD')
	#parser.add_argument('--attention_size',type=int,default=256,help='dimension for attention')



	args = parser.parse_args()
	train(args)
