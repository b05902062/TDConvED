from torch.utils.data import Dataset, DataLoader
import json
import os
from skimage import io
import torch 

g_sample=25

class msr_vtt_dataset(Dataset):
	def __init__(self,vocab_file,image_dir,split,batch):
		with open(vocab_file,"r") as f:
			vocab=json.load(f)

		self.batch=batch
		self.split=os.path.join(image_dir,split)
		self.ref=vocab['ref']
		self.w2i=vocab['w2i']
		self.i2w=vocab['i2w']

		sen_in=vocab['sen_in']
		video_id=vocab['video_id']
		self.sen_in=[]
		self.video_id=[]

		#we are generating batch our self so that we don't need to cut variable length sentences.
		for i in range(len(sen_in)//batch):
			self.sen_in.append(sen_in[batch*i:batch*(i+1)])
			self.video_id.append(video_id[batch*i:batch*(i+1)])
		if (len(sen_in)%batch) != 0:
			self.sen_in.append(sen_in[- (len(sen_in)%batch) :])
			self.video_id.append(video_id[- (len(sen_in)%batch) :])

		#self.sen_in is a list(len=number of batch) of list(len=batch size except that the last batch can be smaller) list(len=length of that very sentence) of word index(int).
		#self.video_id is a list(len the same as first len of self.sen_in) of list(len=the second len of self.sen_in.) of string(video name).
		#print(self.sen_in[:5])
		#print(self.video_id[:5])
	

	def __getitem__(self,idx):
		
		images=torch.zeros(len(self.sen_in[idx]),g_sample,3,256,256)	
		for i_batch,video_id in enumerate(self.video_id[idx]):
			images_list=os.listdir(os.path.join(self.split,video_id))
			for i_sample,image in enumerate(images_list):
				images[i_batch][i_sample]=torch.from_numpy(io.imread(os.path.join(os.path.join(self.split,video_id),image)).transpose(2,0,1))

		lengths=torch.tensor([len(i) for i in self.sen_in[idx]]).type(torch.LongTensor)
		max_len=max(lengths)
		sen=torch.zeros(images.shape[0],max_len).type(torch.LongTensor)
		for i_batch,sen_in in enumerate(self.sen_in[idx]):
			sen[i_batch][:lengths[i_batch]]=torch.tensor(sen_in)
		#return images batch*25*3*256*256 5d tensor.
		#sen is a 2d tensor of size batch*(max_len of this batch) containing word index padded with 0.
		#lengghs is a list(len=batch) of int(length of each sentence in this batch including <sos> and <eos>).
		#video_id is a list of string indicating each sample'scorresponding video name.
		return	images,sen,lengths

	def __len__(self):
		return len(self.sen_in)



if __name__=="__main__":
	msr_vtt=msr_vtt_dataset("/tmp3/data/","train",17)
	print(msr_vtt[5])
	print(msr_vtt[-1])

