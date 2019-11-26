import json
import argparse
import os

def build_word(args):
	msr_vtt_file=args.file
	image_dir=args.image_dir
	output_dir=args.output_dir
	min_count=args.min_count

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	with open(msr_vtt_file,"r") as f:
		msr_vtt=json.load(f)
	video_downloaded=os.listdir(os.path.join(image_dir,msr_vtt['videos'][0]['split']))
	ref={}
	for s in msr_vtt['sentences']:
		if s['video_id'] in video_downloaded:
			if s['video_id'] not in ref.keys():
				ref[s['video_id']]=[[i for i in s['caption'].strip().split()]]	
			else:
				ref[s['video_id']].append([i for i in s['caption'].strip().split()])

	sen=[]
	video_id=[]
	for s in msr_vtt['sentences']:
		if s['video_id'] in video_downloaded:
			video_id.append(s['video_id'])
			sen.append(s['caption'])
	word_count={}
	for s in sen:
		for w in s.strip().split():
			if w not in word_count.keys():
				word_count[w]=1
			else:
				word_count[w]+=1
	word_count={w:c for w,c in word_count.items() if c >= min_count}
	
	#word to index
	w2i={}
	w2i['<pad>']=len(w2i)#0
	w2i['<sos>']=len(w2i)#1
	w2i['<eos>']=len(w2i)#2
	w2i['<unk>']=len(w2i)#3
	for w in word_count.keys():
		w2i[w]=len(w2i)

	#index to word
	i2w={i:w for w,i in w2i.items()}

	sen_in=[]
	for s in sen:
		sen_temp=[w2i['<sos>']]
		for w in s.strip().split():
			if w not in w2i.keys():
				sen_temp.append(w2i['<unk>'])
			else:
				sen_temp.append(w2i[w])
		sen_temp.append(w2i['<eos>'])
		sen_in.append(sen_temp)

	#sen_in is a list(len=number of total sentences for all videos) of list(len=len of that sentence) of word index(int). All index is in i2w.
	#video_id is a list(len=first len of sen_in) of string(the video this sentence corresponds to).
	#ref is a dictionary whose keys are video_id(string) and values are captions for the video. Value is a list(len=# of sentences for the video) of list(len=length of the sentence) of strings.
	#w2i is a mapping from word(string) to index(int).
	#i2w is a inverse mapping of w2i. Though int to string above, json convert int key to string key. So this is string to string.
	
	vocab={'w2i':w2i,'i2w':i2w,'ref':ref,'video_id':video_id,'sen_in':sen_in}
	with open(os.path.join(output_dir,f"{msr_vtt['videos'][0]['split']}_vocab.json"),"w") as f:
		json.dump(vocab,f)
		


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='preprocess data')
	parser.add_argument('--file',default='../data/msr_vtt/train.json',help='path to msr vtt json file')
	parser.add_argument('--image_dir',default='../data/msr_vtt',help='directory to downloaded images.')
	parser.add_argument('--output_dir',default='../data/msr_vtt',help='output directory for preprocessed data.')
	parser.add_argument('--min_count',default=20,type=int,help='turn words occur less than min_count into <unk>.')

	args = parser.parse_args()
	build_word(args)
