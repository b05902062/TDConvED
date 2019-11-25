import argparse
from acquire_images import sample_image




if __name__=="__main__":
	
	parser = argparse.ArgumentParser(description='generate caption')
	parser.add_argument('--video_file',help='path to video file')
	parser.add_argument('--output_dir',default='sampled_images',help='output directory for sampled images')

	args = parser.parse_args()

	sample_image(args.video_file,args.output_dir)
