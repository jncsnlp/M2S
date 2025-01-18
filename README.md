# M2S ['A Novel MLLMs-based Two-stage Model for Zero-shot Multimodal Sentiment Analysis'](https://link.springer.com/chapter/10.1007/978-981-96-0119-6_41)

This is the implementation of our PRICAI '25 paper "A Novel MLLMs-based Two-stage Model for Zero-shot Multimodal Sentiment Analysis", Lu Hengyang, Li Xiaofei.


## Datasets
>
>Twitter-15 and Twitter-17: [`IJCAI2019_data.zip`](https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view) from https://github.com/jefferyYu/TomBERT
>
>MASAD: [`MASAD.zip`](https://drive.google.com/file/d/19YJ8vEYCb-uEKUqSGFmysUTvNzxhVKFE/view?usp=sharing) from https://github.com/12190143/MASAD
>
>MVSA-S and MVSA-M: [`MVSA-Single.zip`](https://portland-my.sharepoint.com/:u:/g/personal/shiaizhu2-c_my_cityu_edu_hk/Ebcsf1kUpL9Do_u4UfNh7CgBC19i6ldyYbDZwr6lVbkGQQ) and [`MVSA-multiple.zip`](https://portland-my.sharepoint.com/:u:/g/personal/shiaizhu2-c_my_cityu_edu_hk/EV4aaLrE-nxGs4ZNyZ8J_o8Bj6hui-PnU-FKYtG7S5r_xQ) from http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data


## MLLMs
We used four Multimodal Large Language Models for our task.
>
>LLaVA-v1.5-7B and LLaVA-v1.5-13B: from https://github.com/haotian-liu/LLaVA
>
>mPLUG-Owl2: from https://github.com/X-PLUG/mPLUG-Owl
>
>LLaMA-Adapter-V2:from https://github.com/ZrrSkywalker/LLaMA-Adapter
MLLMs can be downloaded from [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b/), [LLaVA-v1.5-13B](https://huggingface.co/liuhaotian/llava-v1.5-13b), [mPLUG-Owl2](https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b/), [LLaMA-Adapter-V2](https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth)

## Run
``` shell
sh prompt.sh
sh prompt_caption.sh

python ensemble_prompt.py
```

## Cite Information

Lu, H. Y., & Li, X. F. (2024, November). A Novel MLLMs-Based Two-Stage Model for Zero-Shot Multimodal Sentiment Analysis. In Pacific Rim International Conference on Artificial Intelligence (pp. 436-449). Singapore: Springer Nature Singapore.
```
@inproceedings{lu2024novel,
  title={A Novel MLLMs-Based Two-Stage Model for Zero-Shot Multimodal Sentiment Analysis},
  author={Lu, Heng-Yang and Li, Xiao-Fei},
  booktitle={Pacific Rim International Conference on Artificial Intelligence},
  pages={436--449},
  year={2024},
  organization={Springer}
}
```

Contact: 19850176872@163.com
