import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


g_decode_kernel_size= 7#odd


class TDconvD(nn.Module):
	def __init__(self, embed_size,decode_dim,encode_dim,vocab_size,device,layer=2): 
		super(TDconvD, self).__init__()
		self.relu=nn.ReLU()		
		self.embed = nn.Embedding(vocab_size, embed_size)
		
		self.linear1 = nn.Linear(embed_size+encode_dim, decode_dim)
		self.layer=layer
		self.shifted_convs=nn.ModuleList()
		for i in range(self.layer):
			self.shifted_convs.append(Shifted_conv(decode_dim))

		self.linear2 = nn.Linear(decode_dim, vocab_size)
		self.device=device
	def forward(self, encode, captions, lengths):   

		#for image captioning.
		if(len(encode.shape)==2):
			encode = encode.unsqueeze(dim=1)

		#encode = batch * g_sample * encode_dim 3d tensor.
		#caption = batch * max_label_length(includes <sos>,<eos>) 2d tensor.
		#lengths = 1*batch 2d tensor(length of above captions including <sos>,<eos>,<unk>,but not <pad>).

		embeddings = self.embed(captions)
		#batch * max_label_length * embed_size
		
		encode = encode.mean(dim=1,keepdim=True)
		#encode of size batch * 1 * encode_dim  
			   
		encode = encode.expand(encode.shape[0],embeddings.shape[1],encode.shape[2])
		
		encode = torch.cat((encode, embeddings), 2) #batch * max_label_length * (embed_siencodee + encode_dim)	  
		encode = self.linear1(encode)
		#batch*max_label*decode_dim now
		
		encode = encode.transpose(1,2)
		#batch * decode_dim* max_lable_len now
		for shifted_conv in self.shifted_convs:  
			encode = shifted_conv(encode)
		#batch * decode_dim*max_label_len now

		encode=encode.transpose(1,2) 
		
		encode = self.linear2(encode)
		#print("captions",captions[:,1:])
		#print("predict",encode.max(dim=2)[1][:,:-1])
		#we don't want the last prediction, the one predicted by <eos> of the longest sentence.
		#outputs of size batch*(max_label_len-1)*vocab_size. Use the first to the last-1, thus cutting the last one from result, of label to predict from the second word to the last word.
		
		return encode[:,:-1,:]

	def predict(self,features, max_predict_length=20):
		#max_predict_length is the max length to predict including <sos> so we actually only run it max_predict_length-1 time..
		if(len(features.shape)==2):
			features =features.unsqueeze(1)
		
		#features 3d tensor of size batch*g_sample*encode_dim now
		
		concat = features.mean(dim=1,keepdim=True)
		#video of size batch*1*encode_dim  

		#output is of size batch*max_predict_length
		output=torch.zeros([features.shape[0],max_predict_length]).type(torch.LongTensor).to(self.device)
		output[:,0]=1#<sos>

		#last version is incorrect. seems that we have to do it layer by layer.
		#state initialization of size (1+layer)*batch*decode_dim*kernel_size
		state=torch.zeros([self.layer+1,features.shape[0],features.shape[2],g_decode_kernel_size]).to(self.device)
		
		#before each iteration (layer+1)*batch*decode_dim*[:(g_decode_kernel_size-1)] of state should be ready. We add newly predicted (layer+1)*batch*decode_dim*[g_decode_kernel_size-1] every time.
		for i in range(max_predict_length-1):
			
			#before first shifted_conv. First layer state. 
			add=torch.cat((concat,self.embed(output[:,i].reshape(features.shape[0],1))),dim=2)
			add=self.linear1(add)
			state[0,:,:,g_decode_kernel_size-1]=add

			for i_layer,shifted_conv in enumerate(self.shifted_convs):
				#use the last one of conv output to update next layer.
				state[i_layer+1,:,:,g_decode_kernel_size-1]=shifted_conv(state[i_layer])[:,:,g_decode_kernel_size-1]
				

			predict=state[self.layer,:,:,g_decode_kernel_size-1]
			#predict of size batch*decode_dim now.
			predict=self.linear2(predict)
			#predict of size batch*vocab_size now.
			output[:,i+1]=predict.max(dim=1)[1]

			#prepare state for next iteration. It is like shifting towards the end by one time step.
			state[:,:,:,:g_decode_kernel_size-1]=state[:,:,:,1:]
		#output 2d int tensor of size batch*max_predict_len containing word index. first one is always <sos>.
		return output



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


