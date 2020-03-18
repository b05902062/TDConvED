# This is an implemetation of TDConvED for video captioning.
By Wang Zihmin,李旭峰 2019

## discription
work is going on.

This implementation is based on the paper TDconvED by Jingwen Chen et al. at https://arxiv.org/abs/1905.01077v1?fbclid=IwAR3PIjrHeMBZYcXfPm6J6mIkndjihtIlqsAjQopD_g-TlVvuwZWzEBMf-1Y.





## Usage:
#### Clone this repository:
        git clone https://github.com/b05902062/TDConvED.git
        cd TDConvED
        cd src


#### Download msr vtt dataset:

        #defualt is to download training dataset
        python3 acquire_images.py
        
        #download testing dataset
        python3 acquire_images.py --file ../data/msr_vtt/test.json
downlod videos and sample images from videos according to msr vtt json file.

The process takes about 2.5hrs and 1hrs on our machine. It takes about 1.3GB and 0.6GB of disk space respectively.

Some warning messeage regarding downloading or sampling should be acceptable.

#### Build vocabulary:

        #default is to build training vocabulary.
        python3 build_vocab.py
        
        #build testing vocabulary
        python3 build_vocab.py --min_count 1 --file ../data/msr_vtt/test.json
Organize all data into proper format.

#### Training:
        #train with default parameters
        python3 train.py
        
        
        #Some result. Times it with 100 to get the result as percentage as in the paper.
        Epoch: 1/10 , train BLEU@4: 0.2504181822774608 , test BLEU@4: 0.19647498017409695
        Epoch: 2/10 , train BLEU@4: 0.336280684330615 , test BLEU@4: 0.2034428793154184
        Epoch: 3/10 , train BLEU@4: 0.37172188638671383 , test BLEU@4: 0.18659927072256907
        Epoch: 4/10 , train BLEU@4: 0.40931475958462166 , test BLEU@4: 0.19302179117317217
        Epoch: 5/10 , train BLEU@4: 0.4533808649546446 , test BLEU@4: 0.1881313664760037
        Epoch: 6/10 , train BLEU@4: 0.47106283492848516 , test BLEU@4: 0.16356477544284143
        Epoch: 7/10 , train BLEU@4: 0.4973070155332076 , test BLEU@4: 0.16795545108050405
        Epoch: 8/10 , train BLEU@4: 0.5160734514139727 , test BLEU@4: 0.1647126360258762
        Epoch: 9/10 , train BLEU@4: 0.5173238094007665 , test BLEU@4: 0.15077383177620193
        Epoch: 10/10 , train BLEU@4: 0.5149565933986803 , test BLEU@4: 0.13222158469164846

The logs will be stored in TDConvED/logs and checkpoints will be stored in TDConvED/checkpoints by default. 

Some warning messeage from BLEU evaluation should be acceptable.

        #to see how to modify some parameters' settings.
        python3 train.py -h



#### Use the trained model to generate some description about a video:
First, you need to have a video. You can also download one from youtube.

        #We provide a simple way to download video from youtube.
        #launch python or write in a python file. We launch python directly here.
        python3
        
        #import function
        from acquire_images import download_video
        
        #replace the url to download a video you want.
        #download_video(url,output_dir,video_name)
        #url is a string. url to a youtube video to be downloaded.
        #output_dir is a string. a directory to store downloaded video.
        #video_name is a string. the name without extension for the newly downloaded video.
        #E.g. video_name='hello', the video would be named as 'hello.mp4' and stored in output_dir.
        download_video('https://www.youtube.com/watch?v=3nmcs4G_KLw','../data/','robot')
        
We can launch python or write in a file to generate caption. We also launch python here.

        python3
        
        #import function
        from generate_caption import generate_caption
        
        
        #generate_caption(video_path,ckp_path)
        #video_path is a string, help='path to video file'
        #ckp_path is a string, help='path to a checkpoint(model) used to generate captions. checkpoint can be found in ../checkpoints by default. Check logs in ../logs to choose the one with highest BLEU score'
	#A sample checkpoint can be found at a google drive link:https://drive.google.com/file/d/1SCoV2QRMQ-iJ7Vu9JygEnaGvRl0fX80v/view?usp=sharing
        generate_caption('../data/robot.mp4',<specify_a_checkpoint_path>)

        #you can specify some parameters.
        #generate_caption() returns a list(len=return_top) of list(len=# of words in this sentence) of string.
        #start is a float specifying the start time in seconds of the segment of the video on which you want to generate caption.
        #end is a float specifying the end time in seconds of the segment of the video on which you want to generate caption.
        #return_top specify the number of sentences with highest possibility to return.
        #beam_size is the window size for beam search higher beam size can be better but slower.
        generate_caption('../data/robot.mp4',<checkpoint_path>,start=11,end=28,beam_size=20,return_top=5)
        
        

        
For further understanding of the implementation, you can see the heavily annotated code in /src or see the original paper.
        



