#for an example
#prompt 0-1
#instruction 0-9
python prompt_caption.py \
    --prompt 1 \
    --instruction -1 \
    --dataset mvsa-s \
    --answer_name "mplug-zs-prompt-caption" \
    --model_path '/home/jncsnlp/lxf/mPLUG-Owl2'\
    --tsv_path "/home/jncsnlp/lxf/datasets/mvsa-s" \
    --answer_path "/home/jncsnlp/lxf/answer/mvsa-s/" \
    --image_path "/home/jncsnlp/lxf/dataset/MVSA_Single/data/" 

python prompt_caption.py \
    --prompt 1 \
    --instruction -1 \
    --dataset t2015 \
    --answer_name "mplug-zs-prompt-caption" \
    --model_path '/home/jncsnlp/lxf/mPLUG-Owl2'\
    --tsv_path "/home/jncsnlp/lxf/datasets/t2015" \
    --answer_path "/home/jncsnlp/lxf/answer/t2015/" \
    --image_path "/home/jncsnlp/lxf/dataset/IJCAI2-119_data/twitter2015_images/" 

python prompt_caption.py \
    --prompt 1 \
    --instruction -1 \
    --dataset mvsa-m \
    --answer_name "mplug-zs-prompt-caption" \
    --model_path '/home/jncsnlp/lxf/mPLUG-Owl2'\
    --tsv_path "/home/jncsnlp/lxf/datasets/mvsa-m" \
    --answer_path "/home/jncsnlp/lxf/answer/mvsa-m/" \
    --image_path "/home/jncsnlp/lxf/dataset/MVSA_Multiple/data/"

python prompt_caption.py \
    --prompt 1 \
    --instruction -1 \
    --dataset t2017 \
    --answer_name "mplug-zs-prompt-caption" \
    --model_path '/home/jncsnlp/lxf/mPLUG-Owl2'\
    --tsv_path "/home/jncsnlp/lxf/datasets/t2-117" \
    --answer_path "/home/jncsnlp/lxf/answer/t2017/" \
    --image_path "/home/jncsnlp/lxf/dataset/IJCAI2-119_data/twitter2017_images/"
python prompt_caption.py \
    --prompt 1 \
    --instruction -1 \
    --dataset masad \
    --answer_name "mplug-zs-prompt-caption" \
    --model_path '/home/jncsnlp/lxf/mPLUG-Owl2'\
    --tsv_path "/home/jncsnlp/lxf/datasets/masad" \
    --answer_path "/home/jncsnlp/lxf/answer/masad/" \
    --image_path "/home/jncsnlp/lxf/dataset/MASAD_imgs/"
