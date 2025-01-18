python prompt_caption.py \
    --dataset mvsa-s \
    --tsv_path "/home/jncsnlp/lxf/datasets/mvsa-s" \
    --image_path "/home/jncsnlp/lxf/dataset/MVSA_Single/data/" \
    --answer_path "/home/jncsnlp/lxf/answer/mvsa-s/"  \
    --answer_name "adapter-zs-prompt-caption" \
    --prompt 1 \
    --instruction -1

python prompt_caption.py \
    --dataset mvsa-m \
    --tsv_path "/home/jncsnlp/lxf/datasets/mvsa-m" \
    --image_path "/home/jncsnlp/lxf/dataset/MVSA_Multiple/data/" \
    --answer_path "/home/jncsnlp/lxf/answer/mvsa-m/" \
    --answer_name "adapter-zs-prompt-caption" \
    --prompt 1 \
    --instruction -1

python prompt_caption.py \
    --prompt 1 \
    --instruction -1 \
    --dataset t2015 \
    --answer_name "adapter-zs-prompt-caption" \
    --tsv_path "/home/jncsnlp/lxf/datasets/t2015" \
    --answer_path "/home/jncsnlp/lxf/answer/t2015/" \
    --image_path "/home/jncsnlp/lxf/dataset/IJCAI2019_data/twitter2015_images/"       


python prompt_caption.py \
    --prompt 1 \
    --instruction -1 \
    --dataset t2017 \
    --answer_name "adapter-zs-prompt-caption" \
    --tsv_path "/home/jncsnlp/lxf/datasets/t2017" \
    --answer_path "/home/jncsnlp/lxf/answer/t2017/" \
    --image_path "/home/jncsnlp/lxf/dataset/IJCAI2019_data/twitter2017_images/"  

python prompt_caption.py \
    --prompt 1 \
    --instruction -1 \
    --dataset masad \
    --answer_name "adapter-zs-prompt-caption" \
    --tsv_path "/home/jncsnlp/lxf/datasets/masad" \
    --answer_path "/home/jncsnlp/lxf/answer/masad/" \
    --image_path "/home/jncsnlp/lxf/dataset/MASAD_imgs/"
