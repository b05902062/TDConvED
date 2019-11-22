import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


g_decode_kernel_size= 7#odd


class TDconvD(nn.Module):
	def __init__(self, embed_size,decode_dim,encode_dim,vocab_size,device): 
		super(TDconvD, self).__init__()
		self.relu=nn.ReLU()		
		self.embed = nn.Embedding(vocab_size, embed_size)
		
		self.linear1 = nn.Linear(embed_size+encode_dim, decode_dim)
		
		self.shifted_conv1 = Shifted_conv(decode_dim)
		self.shifted_conv2 = Shifted_conv(decode_dim)

		self.linear2 = nn.Linear(decode_dim, vocab_size)
		self.device=device
	def forward(self, encode, captions, lengths):   

		#for image captioning.
		if(len(encode.shape)==2):
			encode = encode.unsqueeze(dim=1)

		#encode = batch * g_sample * encode_dim 3d tensor.
		#caption = batch * max_label_length(includes <sos>,<eos>) 2d tensor.
		#lengths = list(len=batch) of int(length of above captions including <sos>,<eos>,<unk>,but not <pad>).

		embeddings = self.embed(captions)
		#batch * max_label_length * embed_size
		
		encode = encode.mean(dim=1,keepdim=True)
		#encode of size batch * 1 * encode_dim  
			   
		encode = encode.expand(encode.shape[0],embeddings.shape[1],encode.shape[2])
		
		encode = torch.cat((encode, embeddings), 2) #batch * max_label_length * (embed_siencodee + encode_dim)	  
		encode = self.linear1(encode)				#batch*max_label*decode_dim
		encode = encode.transpose(1,2)			  #batch * decode_dim* max_lable_len
	  
		encode = self.shifted_conv1(encode)			  #batch * decode_dim*max_label_len
		encode = self.shifted_conv2(encode)			  #batch * decode_dim * max_label_len
		encode=encode.transpose(1,2) 
		
		encode = self.linear2(encode)
		##outputs = self.softmax(encode)#crossentropy would do it again. Seems to cause problem.
		#print("captions",captions[:,1:])
		#print("predict",encode.max(dim=2)[1][:,:-1])
		#we don't want the last prediction, the one predicted by <eos> of the longest sentence.
		#outputs of size batch*(max_label_len-1)*vocab_size. Use the first to the last-1, thus cutting the last one from result, of label to predict from the second word to the last word.
		return encode[:,:-1,:]

	def predict(self,features, max_predict_length=20):
		#max_predict_length is the max length to predict including <sos> so we actually only run it max_predict_length-1 time..
		if(len(features.shape)==2):
			features =features.unsqueeze(1)
		
		#features 3d tensor of size batch*g_sample*encode_dim
		
		concat = features.mean(dim=1,keepdim=True).expand(features.shape[0],g_decode_kernel_size,features.shape[2]) #concat batch* g_decode_kernel_size * encode_dim  
		#left side padded with kernel size -1 0.
		sampled_ids = torch.zeros([features.shape[0],max_predict_length+(g_decode_kernel_size-1)]).type(torch.LongTensor).to(self.device)
		sampled_ids[:,g_decode_kernel_size]=1#<sos>
		for i in range(max_predict_length-1):
			embedding=self.embed(sampled_ids[:,i:i+g_decode_kernel_size])#batch*g_decode_kernel_size*enbed_dim
			cnn=torch.cat((concat,embedding),dim=2)
			cnn=self.linear1(cnn)#batch*g_decode_kernel_size*decode_dim
			cnn=cnn.transpose(1,2)
			cnn=self.shifted_conv1(cnn)
			cnn=self.shifted_conv2(cnn)#batch*decode_dim*g_decode_kernel_size
			cnn=cnn.transpose(1,2)[:,g_decode_kernel_size-1,:]#batch*decode_dim. we want the last one from conv.
			cnn=self.linear2(cnn)
			outputs = F.softmax(cnn,dim=1)
			sampled_ids[:,i+g_decode_kernel_size]=outputs.max(dim=1)[1]
		return sampled_ids[g_decode_kernel_size-1:]#batch*max_predict_length containing label id.
def sig_gate(x):
	sig = nn.Sigmoid()
	out_dim = int(x.shape[1]/2)
	#print('sig_x = \n', sig(x[:,:,out_dim:])
	x = x[:,:out_dim,:]*sig(x[:,out_dim:,:])
	return x   

class Shifted_conv(nn.Module):
	def __init__(self,decode_dim): 
		super(Shifted_conv, self).__init__()
		self.relu=nn.ReLU()
		self.shifted_conv = nn.Conv1d(decode_dim, 2*decode_dim, kernel_size = g_decode_kernel_size, padding = g_decode_kernel_size-1) 
		#first shifted convolutional block of the decoder #only left side of input padding with "k-1" zero  #We
	
	def forward(self, Wi):   
		Wi_temp=Wi.clone()
		#batch * decode_dim * max_lable_length
		Wi = self.shifted_conv(Wi)	  #batch * 2decode_dim * (max_label_len+g_decode_kernel_size-1)
		
		Wi=Wi[:,:,:-(g_decode_kernel_size-1)]
		Wi=sig_gate(Wi)+Wi_temp		
		#return batch * decode_dim*max_label_len
		return Wi


