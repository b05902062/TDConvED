import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from encoder import TDconvE
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torchvision import transforms
import nltk
nltk.download('punkt')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if not os.path.exists("./log"):
	os.makedirs("./log")

def main(args):
	# Create model directory
	log=open("./log/"+"_".join(str(i) for i in [args.embed_size,args.hidden_size,args.batch_size,args.learning_rate])+".log","w+")
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	
	# Image preprocessing, normalization for the pretrained resnet
	transform = transforms.Compose([ 
		transforms.RandomCrop(args.crop_size),
		transforms.RandomHorizontalFlip(), 
		transforms.ToTensor(), 
		transforms.Normalize((0.485, 0.456, 0.406), 
							 (0.229, 0.224, 0.225))])
	
	# Load vocabulary wrapper
	with open(args.vocab_path, 'rb') as f:
		vocab = pickle.load(f)
	
	# Build data loader
	data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
							 transform, args.batch_size,
							 shuffle=True, num_workers=args.num_workers) 

	# Build the models
	encoder = TDconvE(args.embed_size).to(device)
	decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
	 
	# Loss and optimizer
	criterion = nn.CrossEntropyLoss(ignore_index=0)
	params = list(decoder.parameters()) + list(encoder.parameters())
	optimizer = torch.optim.Adam(params, lr=args.learning_rate)
	
	# Train the models
	total_step = len(data_loader)
	for epoch in range(args.num_epochs):
		for i, (images, captions, lengths) in enumerate(data_loader):
			# Set mini-batch dataset
			images = images.to(device)
			captions = captions.to(device)
			
			# Forward, backward and optimize
			features = encoder(images)
			outputs = decoder(features, captions, lengths)
			loss = criterion(outputs.reshape(-1,outputs.shape[2]), captions.reshape(-1))
			decoder.zero_grad()
			encoder.zero_grad()
			loss.backward()
			optimizer.step()

			# Print log info
			if i % args.log_step == 0:
				outputs=outputs.max(dim=2)[0].type(torch.cuda.LongTensor)
				we,total_w=WER(outputs,captions,lengths)
				log.write(f'Epoch [{epoch}/{args.num_epochs}], Step [{i}/{total_step}], Loss: {loss.item():.4f}, WER: {100*we/total_w:5.4f}%\n') 
				print(f'Epoch [{epoch}/{args.num_epochs}], Step [{i}/{total_step}], Loss: {loss.item():.4f}, WER: {100*we/total_w:5.4f}%') 
				
			# Save the model checkpoints
			if (i+1) % args.save_step == 0:
				torch.save(decoder.state_dict(), os.path.join(
					args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
				torch.save(encoder.state_dict(), os.path.join(
					args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))

def WER(output,target,lengths):
	#output and input of size batch*max_length_in_length
	#length is a list of size batch.
	print(output.shape,target.shape)
	total_w=0
	we=0
	for i in range(max(lengths)):
		count=0
		for o in lengths:
			if o>i:
				count+=1
		for o in range(count):
			total_w+=1
			if output[o][i]!=target[o][i]:
				we+=1
	return (we,total_w)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
	parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
	parser.add_argument('--vocab_path', type=str, default='/tmp/data/vocab.pkl', help='path for vocabulary wrapper')
	parser.add_argument('--image_dir', type=str, default='/tmp/data/resized2014', help='directory for resized images')
	parser.add_argument('--caption_path', type=str, default='/tmp/data/annotations/captions_train2014.json', help='path for train annotation json file')
	parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
	parser.add_argument('--save_step', type=int , default=5000, help='step size for saving trained models')
	
	# Model parameters
	parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
	parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
	parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
	
	parser.add_argument('--num_epochs', type=int, default=5)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=2)
	parser.add_argument('--learning_rate', type=float, default=0.001)
	args = parser.parse_args()
	print(args)
	main(args)
