import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch
import math



g_encode_dim=512
g_encode_kernel=3#odd
g_video_sample=1

class frame_feature_extraction(nn.Module):
	def __init__(self,model,transfer_dim):
		super(frame_feature_extraction, self).__init__()
		
		if model=='resnet18':
			self.resnet=models.resnet18(pretrained=True)
		self.fc=nn.Linear(1000,transfer_dim)

	def forward(self,x):
		x=self.resnet(x)
		x=self.fc(x)
		return x

class video_feature_extraction(nn.Module):
	def __init__(self,encode_dim):
		super(video_feature_extraction, self).__init__()
		self.encode_dim=encode_dim
		self.frame_encode=frame_feature_extraction("resnet18",encode_dim)
	def forward(self,x):
		#x is a 5d tensor of size batch*g_video_sample*channel*height*width
		x=x.reshape(-1,x.shape[2],x.shape[3],x.shape[4])
		x=self.frame_encode(x)
		x=x.reshape(-1,g_video_sample,self.encode_dim)
		#return batch*g_video_sample*encode_dim
		return x

class TDconv(nn.Module):
	def __init__(self,encode_dim):
		super(TDconv, self).__init__()

		self.deform_conv=nn.Conv1d(1,g_encode_kernel,kernel_size=g_encode_kernel*encode_dim,padding=math.floor(g_encode_kernel/2)*encode_dim,stride=encode_dim)
		self.attention_conv=nn.Conv1d(1,2*encode_dim,kernel_size=g_encode_kernel*encode_dim,stride=g_encode_kernel*encode_dim)
		self.relu=nn.ReLU()
		self.sig=nn.Sigmoid()
	def forward(self,x):
		#input x is of size batch*g_video_sample*encode_dim
		x_filter=x.clone()
		x_temp=x.clone()

		x_filter=x_filter.reshape(x_filter.shape[0],1,-1)#return batch*1*(encode_dim*g_video_sample) 3d tensor.
		x_filter=self.deform_conv(x_filter)#return batch*g_encode_kernel*g_video_sample
		x_filter=x_filter.transpose(1,2)
		#x_filter is of size batch*g_video_sample*g_encode_kernel now
		
		x_filter=x_filter.reshape(x_filter.shape[0],g_encode_kernel*g_video_sample,1).expand(-1,-1,g_video_sample)
		a=torch.tensor([o  for i in range(0,g_video_sample) for o in range(i-math.floor(g_encode_kernel/2),i+math.floor(g_encode_kernel/2)+1)]).type(torch.cuda.FloatTensor).reshape(1,-1,1).expand(x_filter.shape).cuda()
		b=torch.from_numpy(np.arange(g_video_sample)).type(torch.cuda.FloatTensor).reshape(1,1,-1).expand(x_filter.shape).cuda()
		#x_filter, a, and b are now all of size batch*(g_encode_kernel*g_video_sample)*g_video_sample 3d tensor.


		x_filter=x_filter+a
		x_filter=self.sig(x_filter)*(g_video_sample-1)##modification
		print("x_filter_before",x_filter[0][0])#delete
		x_filter=x_filter-b
		x_filter=1-torch.abs(x_filter)
		#print("x_filter_before",x_filter,"gradient",x_filter.grad)#delete
		x_filter=self.relu(x_filter)
		
		print("x_filter",x_filter[0][0])#delete
		
		x=x.unsqueeze(dim=1)#return batch*1*g_video_sample*encode_dim
		x_filter=x_filter.unsqueeze(dim=3)#return  batch*(g_encode_kernel*g_video_sample)*g_video_sample*1
		x=(x*x_filter).sum(dim=2)#member-wise multiplication broadcast
		#x is now batch*(g_encode_kernel*g_video_sample)*encode_dim
		x=x.reshape(x.shape[0],1,-1)
		#x is now batch*(encode_dim*g_encode_kernel*g_video_sample)
		x=self.attention_conv(x)
		#x is now batch*(2*encode_dim)*g_video_sample 3d
		x=x.transpose(1,2)
		x=sig_gate(x)
		#x is now batch*g_video_sample*encode_dim
		x=x+x_temp

		return x


def sig_gate(x):
	sig=nn.Sigmoid()
	out_dim=int(x.shape[2]/2)
	x=x[:,:,:out_dim]*sig(x[:,:,out_dim:])
	return x

class TDconvE(nn.Module):
	def __init__(self,encoder_dim):
		super(TDconvE, self).__init__()
		self.video_encoder=video_feature_extraction(encoder_dim)
		self.TDenc1=TDconv(encoder_dim)
		self.TDenc2=TDconv(encoder_dim)

	def forward(self,x):
		unsqueeze=False	
		if len(x.shape)==4:

			unsqueeze=True
			x=x.unsqueeze(dim=1)
	#video of size batch*g_video_sample*channel*height*width
		x=self.video_encoder(x)#return batch*g_video_sample*encode_dim
		x=self.TDenc1(x)#return batch*g_video_sample*encode_dim
		x=self.TDenc2(x)#return batch*g_video_sample*encode_dim
		if unsqueeze==True:
			x=x.squeeze(dim=1)
		return x	




if __name__=='__main__':
	
	encoder=TDconvE(g_encode_dim)
	#video of size batch*g_video_sample*channel*height*width
	encoding=encoder(video)#return batch*g_video_sample*g_encode_dim
	
