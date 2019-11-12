import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import time


#batch_size = 64
g_encode_dim = 512
g_decode_dim = 512
g_decode_kernel_size= 7#odd
g_video_sample = 1


#(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
#def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):

    
class TDconvD(nn.Module):
    def __init__(self, embed_size,decode_dim,encode_dim,vocab_size,max_predict_length=20): 
        super(TDconvD, self).__init__()
        self.relu=nn.ReLU()        
        self.embed = nn.Embedding(vocab_size, embed_size)       #Wi + embeddings of the input
        
        self.linear1 = nn.Linear(embed_size+encode_dim, decode_dim)
        
        self.shifted_conv1 = Shifted_conv(decode_dim)
        self.shifted_conv2 = Shifted_conv(decode_dim)

        self.linear2 = nn.Linear(decode_dim, vocab_size)
        self.max_predict_length=max_predict_length
        self.softmax=nn.Softmax(dim=2)
    def forward(self, encode, captions, lengths):   

        #for image captioning.
        if(len(encode.shape)==2):
            encode = encode.unsqueeze(dim=1)

        #z = batch * g_video_sample * encoder_dim 3d tensor.
        #caption = batch * max_label_length(includes <sos>,<eos>) 2d tensor.
        #lengths = list(len=batch) of int.

        embeddings = self.embed(captions)   #batch * max_label_length * embed_size
        encode = encode.mean(dim=1,keepdim=True) #z batch * 1 * encode_dim  
               
        encode = encode.expand(encode.shape[0],embeddings.shape[1],encode.shape[2])
        
        encode = torch.cat((encode, embeddings), 2) #batch * max_label_length * (embed_siencodee + encode_dim)      
        encode= self.linear1(encode)                #batch*max_label*decode_dim
        self.relu(encode)
        encode = encode.transpose(1,2)              #batch * decode_dim* max_lable_len

      
        #encode = self.shifted_conv1(encode)              #batch * decode_dim*max_label_len
        encode = self.shifted_conv2(encode)              #batch * decode_dim * max_label_len
        encode=encode.transpose(1,2) 
        
        encode = self.linear2(encode)
        #print("encode",encode[0][1][:100],encode.shape)
        outputs = self.softmax(encode)
        #print("outputs",outputs[0][1][:100],outputs.shape)
        #print("captions",captions)
        #print("predict",outputs.max(dim=2)[1][:,:-1])
        #time.sleep(1)
	#we don't want the last prediction, the one predicted by <eos> of the longest sentence.
        #outputs of size batch*(max_label_len-1)*vocab_size
        return outputs[:,:-1,:]

    def sample(self,z, states=None):
        """Generate captions for given image features using greedy search."""
        if(len(features.shape)==2):
            z =z.unsqueeze(1)
        
        #features 3d tensor of size 1*g_video_sample*encode_dim
        
        concat = z.mean(dim=1,keepdim=True).expand(1,3,z.shape[2]) #concat 1* 3 * encode_dim  
        sampled_ids = torch.zeros([1,self.max_predict_length+3])
        sampled_ids[0][2]=1
        for i in range(self.max_predict_length):
            embedding=self.embed(sampled_ids[:,i:i+2])#1*3*enbed_dim
            cnn=torch.cat((cancat,embedding),dim=2)
            cnn=self.linear1(cnn)#1*3*decode_dim
            cnn=cnn.transpose(1,2)
            cnn=self.shifted_conv1(cnn)
            cnn=self.shifted_conv2(cnn)
            cnn=cnn.transpose(1,2)[:,2,:]#1*1*decode_dim. we want the third from conv.
            cnn.linear2(cnn)
            outputs = F.softmax(z,dim=2)
            sampled_ids[i+3]=outputs.argmax()
            if sampled_ids[0][i+3]==2:
                break
        return sampled_ids[3:]
       
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
        self.shifted_conv1 = nn.Conv1d(decode_dim, 2*decode_dim, kernel_size = g_decode_kernel_size, padding = g_decode_kernel_size-1) 
        #first shifted convolutional block of the decoder #only left side of input padding with "k-1" zero  #We
    
    def forward(self, Wi):   
        Wi_temp=Wi.clone()
        #batch * decode_dim * max_lable_length
        Wi = self.shifted_conv1(Wi)      #batch * 2decode_dim * max_label_len+(g_decode_kernel_size-1)
        
        Wi=Wi[:,:,:-(g_decode_kernel_size-1)]
        self.relu(Wi) 
        Wi=sig_gate(Wi)+Wi_temp        
        #return batch * decode_dim*max_label_len
        return Wi


