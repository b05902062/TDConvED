# This is an implemetation of TDConvED for video captioning.
By Wang Zihmin,李旭峰 2019

This implementation is based on the paper TDconvED at https://arxiv.org/abs/1905.01077v1?fbclid=IwAR3PIjrHeMBZYcXfPm6J6mIkndjihtIlqsAjQopD_g-TlVvuwZWzEBMf-1Y.

## discription

work is going on.

## usage:
#### downlod videos and sample images from videos:

        python3 acquire_images.py --file <path_to_json_file>
        
        e.g. python3 acquire_images.py --file ./videodatainfo_2017.json

the process takes about 4hrs on our machine and takes 2GB of disk space.

#### build vocabulary:

        python3 build_vocab.py --file <path_to_json_file>
        
        e.g. python3 build_vocab.py --file ./videodatainfo_2017.json