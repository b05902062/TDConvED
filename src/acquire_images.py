import argparse
import os
import json
from multiprocessing import Pool
from pytube import YouTube
import ffmpeg
from itertools import repeat

g_sample=25


def download_and_sample(args):

	msr_vtt=args.file
	output_dir=args.output_dir



	with open(msr_vtt,"r") as f:
		msr_vtt=json.load(f)['videos']

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	split=msr_vtt[0]['split']
	output_dir=os.path.join(output_dir,split)
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	"""
	for i in msr_vtt[:2]:
		__download_sample(i,output_dir)

	"""
	p=Pool(10)
	p.starmap(__download_sample,zip(msr_vtt,repeat(output_dir)))
	
	p.close()
	p.join()
	
	
def __download_sample(video,output_dir):

	try:
		yt = YouTube(video['url'])
		yt.streams.filter(file_extension='mp4').first().download(output_path=output_dir,filename=video['video_id'])
		__sample_image(video,output_dir)
		os.remove(os.path.join(output_dir,video['video_id']+'.mp4'))

	except:
		if os.path.isfile(os.path.join(output_dir,video['video_id']+'.mp4')):
			os.remove(os.path.join(output_dir,video['video_id']+'.mp4'))
		return

def __sample_image(video,output_dir):

	video_name=os.path.join(output_dir,video['video_id']+".mp4")
	probe = ffmpeg.probe(video_name)
	video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
	if video_stream is None:
		return

	output_dir=os.path.join(output_dir,video['video_id'])
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
		
	len=float(video_stream['duration'])
	start=video['start time']
	end=video['end time']
	if end<0 or end>len or start<0 or start>len or end<start:
		return
	for i in range(0,g_sample):
		time=start+(end-start)*i/g_sample
		(
			ffmpeg
			.input(video_name, ss=time)
			.filter('scale', 256, 256)
			.output(os.path.join(output_dir,video['video_id']+f"_{i}.jpg"), vframes=1)
			.global_args('-loglevel', 'error')
			.global_args('-y')
			.run()
		)
	

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='download and sample images')
	parser.add_argument('--file',help='path to msr vtt json file')
	parser.add_argument('--output_dir',default='./data',help='output directory for sampled images')

	args = parser.parse_args()
	download_and_sample(args)
