import argparse
from skimage import io
import os
import torch
from acquire_images import sample_image
from encoder import ResTDconvE
from decoder import TDconvD
from train import get_sentence

g_sample=25

def generate_caption(video_path,output_dir,ckp_path,device,return_top=5,beam_size=5,max_predict_length=20,return_prob=True):
		
	device = torch.device(device if torch.cuda.is_available() else 'cpu')
	
	checkpoint = torch.load(ckp_path)
	args=checkpoint['args']
	args.device=device
	w2i=checkpoint['w2i']
	i2w=checkpoint['i2w']

	encoder = ResTDconvE(args).to(args.device)
	decoder = TDconvD(args.embed_dim, args.decoder_dim,args.encoder_dim,args.attend_dim, len(w2i),args.device,layer=args.decoder_layer).to(args.device)
	
	encoder.load_state_dict(checkpoint['encoder'])
	decoder.load_state_dict(checkpoint['decoder'])
	encoder.eval()
	decoder.eval()

	sample_image(video_path,output_dir,start=113,end=130)
	#images 1*g_sample*3*256*256 5d tensor.
	images=get_images(video_path,output_dir).to(args.device)
	features=encoder(images)	

	(predict,prob)=decoder.predict(features,return_top=return_top,beam_size=beam_size,max_predict_length=max_predict_length)
	#return a list of 2d tensor of size 1*max_predict_len containing word index. first token of each tensor is always <sos>.
	#return a list of score for the sentences in the same order. score=log(prob_of_entire_sentence)/(#_of_words_in_the_sentence-1). For example,"<sos> i sleep" with probability 1*0,05*0.03. score=log(1*0.05*0.03)/2.

	sentences=[]
	for i in predict:
		sentences.extend(get_sentence(i,w2i,i2w))
	
	#sentences is a list(len=batch) of list(len=# of words in this sentence, from 0 to max_predict-1) of string.	
	if return_prob==False:		 
		return sentences
	else:
		return sentences,prob



def get_images(video_path,output_dir):
	
	output_dir=os.path.join("/".join(video_path.split('/')[:-1]),output_dir)
	#output_dir becomes the path to the directory containing sampled images.

	images=torch.zeros([1,g_sample,3,256,256])	
	images_list=os.listdir(output_dir)
	for i_sample,image in enumerate(images_list):
		images[0][i_sample]=torch.from_numpy(io.imread(os.path.join(output_dir,image)).transpose(2,0,1))

	#return 1*g_sample*3*256*256 5d tensor.
	return images

if __name__=="__main__":
	
	parser = argparse.ArgumentParser(description='generate caption')
	parser.add_argument('--video_path',help='path to video file')
	parser.add_argument('--output_dir',default='sampled_images',help='output_dir is the name of the folder that will contain the resultihg sampled images in the same directory as video. For example, videoi_path="/tmp3/TDConvED/hello_world.mp4". output_dir="hello_world". The resulting images will be stored in /tmp3/TDConvED/hello_world.')
	parser.add_argument('--ckp_path',help='path to a checkpoint(model) used to generate captions.')
	parser.add_argument('--device',default='cpu',help='the device used to run the model.e.g. cpu, cuda:0, etc.')

	args = parser.parse_args()

	predict=generate_caption(args.video_path,args.output_dir,args.ckp_path,args.device)
	print(predict)	
	#print("predict, first to last",predict[0])
	#print("predict,first to last",get_sentence(predict[0],train_meta))
