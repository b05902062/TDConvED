# This is an implemetation of TDConvED for video captioning.
By Wang Zihmin,李旭峰 2019

## discription
work is going on.

This implementation is based on the paper TDconvED at https://arxiv.org/abs/1905.01077v1?fbclid=IwAR3PIjrHeMBZYcXfPm6J6mIkndjihtIlqsAjQopD_g-TlVvuwZWzEBMf-1Y.





## usage:
#### downlod videos and sample images from videos:

        cd src
        
        #defualt is to download training dataset
        python3 acquire_images.py
        
        #download testing dataset
        python3 acquire_images.py --file ../data/msr_vtt/test.json

The process takes about 2.5hrs and 1hrs on our machine. It takes about 1.2GB and 0.6GB of disk space respectively.

#### build vocabulary:

        #default is to build training vocabulary.
        python3 build_vocab.py
        
        #build testing vocabulary
        python3 build_vocab.py --min_count 1 --file ../data/msr_vtt/test.json
        
#### training
        python train.py