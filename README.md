# convert_TFRecord
Convert script from jpeg to TFRecord for TF-Slim image classification library

<https://github.com/tensorflow/models/tree/master/slim>

#Usage
First, make the image data directory structure as below.

    root_dir/
    + tulip00.jpg
    + Lexus03.jpg
    + sub_dir/
       + Dandelion.jpg
    ... 
    + list.csv #label list  

The csv file format is as below.  

    tulip00.jpg, flower
    Lexus03.jpg, car
    sub_dir/Dandelion.jpg, flower
    ...

Second, run the script. For example, as below.  

    python convert_tfrecord.py \
        --input_dir root_dir \
        --num_data 500 \
        --validations 2500

num_data : Number of data per file.  
validations : Number of validation data.  

