from data_loader import msr_vtt_dataset
form encoder import TDconvE
from decoder import TDconvD
import torch

def train(args):


	msr_vtt=msr_vtt_dataset(args.data_dir,split="train")
	data = DataLoader(msr_vtt,batch_size=16,shuffle=True)
	
	encoder = TDconvE(args.encode_dim).to(device)
	decoder = TDconvD(args.embed_size, args.hidden_size, len(vocab)).to(device)

	criterion = nn.CrossEntropyLoss()
	params = list(decoder.parameters()) + list(encoder.parameters())
	optimizer = torch.optim.Adam(params, lr=args.learning_rate)

	for i_epoch in range(args.epoch):
		for i_b,images, in tqdm(enumerate(data))
