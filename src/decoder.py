import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import math

g_decode_kernel_size= 3#odd


class TDconvD(nn.Module):
	def __init__(self, embed_size,decode_dim,encode_dim,attend_dim,vocab_size,device,layer): 
		super(TDconvD, self).__init__()
		self.relu=nn.ReLU()		
		self.decode_dim=decode_dim
		self.embed = nn.Embedding(vocab_size, embed_size)
		self.linear_decode_att=nn.Linear(decode_dim,attend_dim)
		self.linear_encode_att=nn.Linear(encode_dim,attend_dim)
		self.linear_prob=nn.Linear(attend_dim,1)
		self.linear1 = nn.Linear(embed_size+encode_dim, decode_dim)
		self.layer=layer
		self.shifted_convs=nn.ModuleList()
		for i in range(self.layer):
			self.shifted_convs.append(Shifted_conv(decode_dim))

		self.linear2 = nn.Linear(encode_dim+decode_dim, vocab_size)
		self.device=device
		self.tanh=nn.Tanh()
		self.softmax=nn.Softmax(dim=1)
	def forward(self, features, captions):   

		#for image captioning.
		if(len(features.shape)==2):
			features = features.unsqueeze(dim=1)

		#features = batch * g_sample * encode_dim 3d tensor.
		#caption = batch * max_label_length(includes <sos>,<eos>) 2d tensor.
		#lengths = 1*batch 2d tensor(length of above captions including <sos>,<eos>,<unk>,but not <pad>).

		embeddings = self.embed(captions)
		#batch * max_label_length * embed_size
		
		encode = features.mean(dim=1,keepdim=True)
		#encode of size batch * 1 * encode_dim  
			   
		encode = encode.expand(encode.shape[0],embeddings.shape[1],encode.shape[2])
		
		encode = torch.cat((encode, embeddings), 2) #batch * max_label_length * (embed_dim + encode_dim)	  
		encode = self.linear1(encode)
		#batch*max_label*decode_dim now
		
		encode = encode.transpose(1,2)
		#batch * decode_dim* max_lable_len now
		for shifted_conv in self.shifted_convs:  
			encode = shifted_conv(encode)
		#batch * decode_dim*max_label_len now

		#we don't want the last prediction, the one predicted by <eos> of the longest sentence.
		#Use the first label to the last-1 label, thus cutting the last one from encode, to predict from the second word to the last word.
		encode=encode.transpose(1,2)[:,:-1,:] 
		#batch * (max_label_len-1)*decode_dim now

		#Take a simpler approach to do attention. Doing it one by one.
		#temp of size batch*(max_label_len-1)*encode_dim
		temp=torch.zeros([features.shape[0],encode.shape[1],features.shape[2]]).to(self.device)
		for i in range(encode.shape[1]):
			temp[:,i,:]=self.attention(features,encode[:,i,:])

		encode=torch.cat((temp,encode),dim=2)
		#encode batch*(max_label_len-1)*(encode_dim+decode_dim) now.
		encode=self.linear2(encode)

		#print("captions",captions[:,1:])
		#print("predict",encode.max(dim=2)[1])
		#encode of size batch*(max_label_len-1)*vocab_size.
		
		return encode

	def attention(self,encode,decode):
		#Take a simpler approach to do this. Doing it one by one.

		#encode of batch * g_sample * encode_dim 3d tensor.
		#decode of batch *decode_dim
		features=self.linear_encode_att(encode)
		predict=self.linear_decode_att(decode).reshape(decode.shape[0],1,decode.shape[1])
		#features of batch * g_sample * attend_dim 3d tensor.
		#predict of batch *1* attend_dim

		#broadcast
		features=features+predict
		#features of batch * g_sample * attend_dim 3d tensor now.
		featrues=self.tanh(features)
		features=self.linear_prob(features).squeeze(dim=2)
		features=self.softmax(features).unsqueeze(dim=2)
		encode=(encode*features).sum(dim=1)
		#return is of size batch*encode_dim
		return encode
				

	#this is a wrapper for self.__predict(), which is the original self.predict().
	def predict(self,features,return_top=1,beam_size=5, max_predict_length=20):
		#return_top used to indicate how many sentences to return.
		

		#max_predict_length is the max length to predict including <sos> so we actually only run it max_predict_length-1 time..
		if(len(features.shape)==2):
			features =features.unsqueeze(1)
		
		#features 3d tensor of size 1*g_sample*encode_dim now
		#we only support a batch size of 1.	

		concat = features.mean(dim=1,keepdim=True)
		#concat of size 1*1*encode_dim  

		#beam_state init (self,layer,decode_dim,kernel,device,max_predict_len)
		beam_list=[beam_state(self.layer,self.decode_dim,g_decode_kernel_size,self.device,max_predict_length)]#initalize with one instance.
		final_list=[]
		#first one is <sos> so we predict for (max_predict_length-1) time.
		for i in range(max_predict_length-1):
			if len(beam_list)==0:
				break
			new_beam_list=[]
			new_beam_list_not_final=[]
			for b in beam_list:
				#state and new_state of size 1*(layer+1)*decode_dim*g_decode_kernel_size.
				#prob of 1d tensor of size vocab_size. containing predicted probability distribution over words.
				new_state,prob=self.__predict(b.state,b.get_last_word(),features,concat)
				prob_id=prob.argsort(descending=True)
				for a in range(beam_size):
					#every single beam_state object is completely independent. The same as doing a copy.deepcopy.
					#Seems like state would continue to hold gradient information. But prob list or word tensor no longer contain any gradient.
					new_beam_list.append(b.copy().add(new_state,prob_id[a].item(),prob[prob_id[a]].item()))
				
			for b in new_beam_list:
				if b.word[i+1] == 2:#<eos>
					final_list.append(b)
				else:
					new_beam_list_not_final.append(b)
			beam_list=sorted(new_beam_list_not_final,key=lambda x: x.avg_log_prob(),reverse=True)[:beam_size]

		final_list.extend(beam_list)
		beam_list=[]
		score_list=[]
		for i in range(min(return_top,len(final_list))):
			beam_list.append(sorted(final_list,key=lambda x: x.avg_log_prob(),reverse=True)[i].word.unsqueeze(dim=0))
			score_list.append(sorted(final_list,key=lambda x: x.avg_log_prob(),reverse=True)[i].avg_log_prob())
		#return a list of 2d tensor of size 1*max_predict_len containing word index. first token of each tensor is always <sos>.
		#return a list of score for the sentences in the same order. score=log(prob_of_entire_sentence)/(#_of_words_in_the_sentence-1). For example,"<sos> i sleep" with probability 1*0,05*0.03. score=log(1*0.05*0.03)/2.
		return beam_list,score_list
		
			
	#used to produce next word.
	def __predict(self,state,last_word,features,concat):
		state=state.clone()
		#state passed in will be left intact.
		#state of size (layer+1)*1*decode_dim*g_decode_kernel_size. (layer+1)*1*decode_dim*[:(g_decode_kernel_size-1)] should be ready.
		#last_word is an 1d tensor of size 1. word index.
		#features is 3d tensor of size 1*g_sample*encode_dim.
		#concat of size 1*1*encode_dim. 
		
		#We return state with different id() from the one passed in that is ready for the next prediction. it is of the same size a the one passed in.
		#return the probabiliry distribution over all words.
			
		#before first shifted_conv. First layer state. 
		add=torch.cat((concat,self.embed(last_word.reshape(1,1))),dim=2)
		#add of size 1*1*(encode_dim+embed_dim) now.
		add=self.linear1(add)
		#add of size 1*1*(decode_dim) now.
		
		state[0,:,:,g_decode_kernel_size-1]=add.reshape(1,-1)#1*decode_dim

		for i_layer,shifted_conv in enumerate(self.shifted_convs):
			#use the last one of conv output to update next layer.
			#input and output of shifted_conv are all 1*decode_dim*g_decode_kernel_size
			state[i_layer+1,:,:,g_decode_kernel_size-1]=shifted_conv(state[i_layer])[:,:,g_decode_kernel_size-1]
			

		predict=state[self.layer,:,:,g_decode_kernel_size-1]
		#predict of size 1*decode_dim now.
		#features of size 1*g_sample*encode_dim.
		combine=self.attention(features,predict)
		#combine of size 1*encode_dim now.
		combine=torch.cat((combine,predict),dim=1)
		combine=self.linear2(combine)
		#combine of size 1*vocab_size now.
		#softmax along dim=1.
		combine=self.softmax(combine)
		#combine of size 1*vocab_size now.
		combine=combine.squeeze(dim=0)
		
		#combine is 1d tensor of size vocab_size. containing probability for each word.
		#prepare state for next iteration. It is like shifting towards the end by one time step.
		state[:,:,:,:g_decode_kernel_size-1]=state[:,:,:,1:]
	
		return state,combine

class beam_state():
	def __init__(self,layer,decode_dim,kernel,device,max_predict_len):
		self.layer=layer
		self.decode_dim=decode_dim
		self.kernel=kernel
		self.device=device
		self.max_predict_len=max_predict_len

		self.state=torch.zeros([layer+1,1,decode_dim,kernel]).to(device)
		self.prob=[]
		self.word=torch.zeros([max_predict_len]).type(torch.LongTensor).to(device)
		self.word[0]=1#<sos>
	def add(self,state,new_word,new_prob):
		assert (len(self.prob)+1)<self.max_predict_len, "exceed max predict length"
		#update state this way so that they won't become the same object.
		self.state[:,:,:,:]=state[:,:,:,:]
		self.word[len(self.prob)+1]=new_word
		self.prob.append(new_prob)
		return self
	def avg_log_prob(self):
		log_prob=0
		for p in self.prob:
			log_prob+=math.log(p)
		return log_prob/len(self.prob)
	def get_last_word(self):
		return self.word[len(self.prob)]

	def copy(self):
		#deep copy don't work so we copy our self. This should do the job done by copy.deepcopy().
		new_beam=beam_state(self.layer,self.decode_dim,self.kernel,self.device,self.max_predict_len)
		#deep copy all elements.
		new_beam.state=self.state.clone()
		new_beam.word=self.word.clone()
		new_beam.prob=self.prob.copy()
		return new_beam

		
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


