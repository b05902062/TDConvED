import argparse
import os
import json
from multiprocessing import Pool
from pytube import YouTube
import ffmpeg
from itertools import repeat
import shutil

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
	
	#delete video files that's somehow left behind.
	dir=[i for i in os.listdir(output_dir) if i.find('.mp4')!=-1]
	for i in dir:
		os.remove(os.path.join(output_dir,i))
	#delete directory that somehow doesn't have g_sample images.
	dir=os.listdir(output_dir)
	for i in dir:
		if len(os.listdir(os.path.join(output_dir,i)))!=g_sample:
			shutil.rmtree(os.path.join(output_dir,i))	
			
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

	video_path=os.path.join(output_dir,video['video_id']+".mp4")
	start=video['start time']
	end=video['end time']
	sample_image(video_path,video['video_id'],start=start,end=end)


	
def sample_image(video_path,output_dir,start=None,end=None):
	#video_path is the path to video file.
	#output_dir is the name of the folder that will contain the resultihg sampled images in the same directory as video. For example, videoi_path="/tmp3/TDConvED/hello_world.mp4". output_dir="hello_world". The resulting images will be stored in /tmp3/TDConvED/hello_world.
	#start is a float specify the start of the video in second.
	#end is a float specify the end of the video in second.	

	output_abs_dir=os.path.join("/".join(video_path.split("/")[:-1]),output_dir)
	if not os.path.exists(output_abs_dir):
		os.mkdir(output_abs_dir)

	probe = ffmpeg.probe(video_path)
	video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
	if video_stream is None:
		print(video_name,"error")
		return

	len=float(video_stream['duration'])
	if start==None:
		start=float(video_stream['start_time'])
	if end==None:
		end=len


	if end<0 or end>len or start<0 or start>len or end<start:
		print(video_name,"error")
		return
	for i in range(0,g_sample):
		time=start+(end-start)*i/g_sample
		(
			ffmpeg
			.input(video_path, ss=time)
			.filter('scale', 256, 256)
			.output(os.path.join(output_abs_dir,output_dir+f"_{i}.jpg"), vframes=1)
			.global_args('-loglevel', 'error')
			.global_args('-y')
			.run()
		)

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='download and sample images')
	parser.add_argument('--file',default='../data/msr_vtt/train.json',help='path to msr vtt json file')
	parser.add_argument('--output_dir',default='../data/msr_vtt',help='output directory for sampled images')
	args = parser.parse_args()
	assert os.path.exists(args.output_dir),"--output_dir doesn't exist."
	download_and_sample(args)
