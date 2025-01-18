import torch
from PIL import Image
from transformers import TextStreamer
import csv
import time
import json
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def get_all_prompt1(i,j,texts,captions):
    #context="What's the sentiment of the text and the image?\nOptions:negative,positive,neutral.\nSelect one from options without explanation." 
    if j==0:
        context="What's the sentiment of the text and the image?\nOptions:negative,positive,neutral.\nSelect one from options without explanation." 
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    elif j==1:
        context="What's the sentiment of the text and the image?\nOptions:positive,neutral,negative.\nSelect one from options without explanation." 
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    elif j==2:
        context="What's the sentiment of the text and the image?\nOptions:neutral,negative,positive.\nSelect one from options without explanation." 
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    elif j ==3:
        context="What's the sentiment of the text and the image?\nOptions: A) negative B) positive C) neutral\nOutput a letter from options without explanation."
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    elif j ==4:
        context="What's the sentiment of the text and the image?\nOptions: A) positive B) neutral C) negative\nOutput a letter from options without explanation."
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    elif j ==5:
        context="What's the sentiment of the text and the image?\nOptions: A) neutral B) negative C) positive\nOutput a letter from options without explanation."
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    return prompt

def get_all_prompt2(i,j,aspects,texts,captions):
    #context="What's the sentiment of the text and the image?\nOptions:negative,positive,neutral.\nSelect one from options without explanation." 
    if j==0:
        context="Aspect:"+aspects[i]+".\n"+"According to the text and the image,what's the sentiment of aspect?\nOptions:negative,positive,neutral.\nSelect one from options without explanation." 
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    elif j==1:
        context="Aspect:"+aspects[i]+".\n"+"According to the text and the image,what's the sentiment of aspect?\nOptions:positive,neutral,negative.\nSelect one from options without explanation." 
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    elif j==2:
        context="Aspect:"+aspects[i]+".\n"+"According to the text and the image,what's the sentiment of aspect?\nOptions:neutral,negative,positive.\nSelect one from options without explanation." 
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    elif j ==3:
        context="Aspect:"+aspects[i]+".\n"+"According to the text and the image,what's the sentiment of aspect?\nOptions: A) negative B) positive C) neutral\nOutput a letter from options without explanation."
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    elif j ==4:
        context="Aspect:"+aspects[i]+".\n"+"According to the text and the image,what's the sentiment of aspect?\nOptions: A) positive B) neutral C) negative\nOutput a letter from options without explanation."
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    elif j ==5:
        context="Aspect:"+aspects[i]+".\n"+"According to the text and the image,what's the sentiment of aspect?\nOptions: A) neutral B) negative C) positive\nOutput a letter from options without explanation."
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    return prompt

def get_all_prompt3(i,j,aspects,texts,captions,):
    #context="What's the sentiment of the text and the image?\nOptions:negative,positive,neutral.\nSelect one from options without explanation." 
    if j==0:
        context="Aspect:"+aspects[i]+".\n"+"According to the text and the image,what's the sentiment of aspect?\nOptions:negative,positive.\nSelect one from options without explanation." 
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    elif j==1:
        context="Aspect:"+aspects[i]+".\n"+"According to the text and the image,what's the sentiment of aspect?\nOptions:positive,negative.\nSelect one from options without explanation." 
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    elif j ==2:
        context="Aspect:"+aspects[i]+".\n"+"According to the text and the image,what's the sentiment of aspect?\nOptions: A) negative B) positive\nOutput a letter from options without explanation."
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    elif j ==3:
        context="Aspect:"+aspects[i]+".\n"+"According to the text and the image,what's the sentiment of aspect?\nOptions: A) positive B) negative\nOutput a letter from options without explanation."
        prompt="Text:"+texts[i]+".\n"+"Image description:"+captions[i]+"\n"+context
    return prompt
#instruction from MMbigbench
def get_msa_instruction(i):
    MSA_instructs=[
    # 1 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Label:",
    # 2 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Question: {Question} Answer:",
    # 3
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\
    ### Instruction: Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} \
    ### Instruction: {Question} Options: {Options-1} ### Response: ",
    # 4
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} {Question}",
    # 5 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Question: {Question} Options: {Options-1} Answer:",
    # 6 
    "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\
    Human: Please perform {Task name} {Task definition} {Output format} Human: {Text input} Human: {Question} AI:",
    # 7 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Question: {Question} Options: {Options-2} Answer:",
    # 8 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input}",
    # 9
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
    ### Instruction: Please perform {Task name} {Task definition} {Output format} \
    ### Input: {Text input} ### Input: {Question} ### Response:",
    # 10 
    "User: Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Question: {Question} :<answer>"
    ]
    MSA_dict={
    "Task name": "multimodal Sentiment Analysis task.",
    "Task definition": "Given the text-image pair, assign a sentiment label from ['negative', 'neutral', 'positive'].",
    "Output format": "Return label only without any other text.",
    #"Text input": "hsc summer fun day toronto center island centre island Toronto
    "Question":"what is the sentiment about the text-image pair?",
    "Options-1":"(a) neutral (b) negative (c) positive" ,"Options-2": "neutral or negative or positive"
    }
    instruct=MSA_instructs[i]
    for j in MSA_dict:
        #print(MSA_dict[])
        #print("{"+j+"}")
        #print(MSA_dict[j])#MSA_dict[j]
        instruct=instruct.replace("{"+j+"}",MSA_dict[j])
    return instruct

def get_masa_instruction(i):
    MABSA_instructs=[
    # 1 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} Label:",
    # 2 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} Question: {Question} Answer:",
    # 3
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\
    ### Instruction: Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} \
    ### Instruction: {Question} Options: {Options-1} ### Response: ",
    # 4
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} {Question}",
    # 5 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} Question: {Question} Options: {Options-1} Answer:",
    # 6 
    "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\
    Human: Please perform {Task name} {Task definition} {Output format} Human: {Text input} Aspect:{Aspect input} Human: {Question} AI:",
    # 7 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} Question: {Question} Options: {Options-2} Answer:",
    # 8 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} ",
    # 9
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
    ### Instruction: Please perform {Task name} {Task definition} {Output format} \
    ### Input: {Text input} Aspect: {Aspect input}  ### Input: {Question} ### Response:",
    # 10 
    "User: Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} Question: {Question} :<answer>"
    ]
    MABSA_dict={
        "Task name": "multimodal aspect-based sentiment classification task.",
    "Task definition": "Given the text-image pair and the aspect, assign a sentiment label towards \"target\" from ['negative', 'neutral', 'positive'].",
    "Output format": "Return label only without any other text.",
    #"Text input": "hsc summer fun day toronto center island centre island Toronto, 'neutral'
    "Question":"what is the sentiment about the aspect based on the text-image pair?",
    "Options-1":"(a) neutral (b) negative (c) positive" ,"Options-2": "neutral or negative or positive"
    }
    instruct=MABSA_instructs[i]
    current_dict=MABSA_dict
    for j in current_dict:
        instruct=instruct.replace("{"+j+"}",current_dict[j])
    print(instruct)

def get_masad_instruction(i):
    MABSA_instructs=[
    # 1 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} Label:",
    # 2 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} Question: {Question} Answer:",
    # 3
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\
    ### Instruction: Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} \
    ### Instruction: {Question} Options: {Options-1} ### Response: ",
    # 4
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} {Question}",
    # 5 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} Question: {Question} Options: {Options-1} Answer:",
    # 6 
    "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\
    Human: Please perform {Task name} {Task definition} {Output format} Human: {Text input} Aspect:{Aspect input} Human: {Question} AI:",
    # 7 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} Question: {Question} Options: {Options-2} Answer:",
    # 8 
    "Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} ",
    # 9
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
    ### Instruction: Please perform {Task name} {Task definition} {Output format} \
    ### Input: {Text input} Aspect: {Aspect input}  ### Input: {Question} ### Response:",
    # 10 
    "User: Please perform {Task name} {Task definition} {Output format} Sentence: {Text input} Aspect: {Aspect input} Question: {Question} :<answer>"
    ]
    MABSA_dict={
        "Task name": "multimodal aspect-based sentiment classification task.",
    "Task definition": "Given the text-image pair and the aspect, assign a sentiment label towards \"target\" from ['negative', 'neutral', 'positive'].",
    "Output format": "Return label only without any other text.",
    #"Text input": "hsc summer fun day toronto center island centre island Toronto, 'neutral'
    "Question":"what is the sentiment about the aspect based on the text-image pair?",
    "Options-1":"(a) negative (b) positive" ,"Options-2": "negative or positive"
    }
    instruct=MABSA_instructs[i]
    current_dict=MABSA_dict
    for j in current_dict:
        instruct=instruct.replace("{"+j+"}",current_dict[j])
    print(instruct)

def get_msa_data(args):
    tsvpath=args.tsv_path
    test_paths=[tsvpath+'/test.tsv',tsvpath+"-caption/test.tsv"]
    imagepath=args.image_path
    ids=[]
    ocr=[]
    anps=[]
    texts=[]
    labels=[]
    images=[]
    captions=[]
    image_files=[]
    
    with open(test_paths[1], "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        lines = []
        for line in reader:
            lines.append(line)
        # remove the header row
        lines.pop(0)  
        for line in lines:
            text=line[3].lower()
            texts.append(text)
            ids.append(line[0])
            labels.append(line[1])
            images.append(line[2])
            captions.append(line[4].lower())
            image_files.append(imagepath+line[2])
            
    return ids,texts,images,image_files,captions,labels

def get_masa_data(args):
    base_path=args.tsv_path
    test_paths=[base_path+'/test.tsv',base_path+"-caption/test.tsv"]
    ids=[] 
    texts=[]
    labels=[]
    images=[]
    aspects=[]
    captions=[]
    image_files=[]
    
    map={"0":"negative","1":"neutral","2":"positive"}
    with open(test_paths[1], "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        lines = []
        for line in reader:
            lines.append(line)
        lines.pop(0)  
        # remove the header row
        for line in lines:
            text=line[3]#.lower()
            text=text.replace('$T$',line[4])
            texts.append(text.lower())
            ids.append(line[0])
            captions.append(line[5].lower())
            images.append(line[2])
            aspects.append(line[4].lower())
            labels.append(map[line[1]])
            image_files.append(args.image_path+line[2])

    return ids,texts,aspects,images,image_files,captions,labels

def get_masad_data(args):
    base_path=args.tsv_path
    test_paths=[base_path+'/test.tsv',base_path+"-caption/test.tsv"]
    ids=[]
    texts=[]
    labels=[]
    images=[]
    aspects=[]
    captions=[]
    image_files=[]
    with open(test_paths[1], "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        lines = []
        for line in reader:
            lines.append(line)
        lines.pop(0)  
        # remove the header row
        for line in lines:
            text=line[3].lower()
            texts.append(text)
            ids.append(line[0])
            captions.append(line[5].lower())
            images.append(line[2])
            aspects.append(line[4].lower())
            image_files.append(args.image_path+line[2])
            labels.append(line[1])
    return ids,texts,aspects,images,image_files,captions,labels

def main(args):  
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")

    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles

    dataset=args.dataset
    if dataset in ['mvsa-s','mvsa-m']:
        ids,texts,images,image_files,captions,labels=get_msa_data(args)
    elif dataset in ['t2015','t2017']:
        ids,texts,aspects,images,image_files,captions,labels=get_masa_data(args)
    else:
        ids,texts,aspects,images,image_files,captions,labels=get_masad_data(args)

    instruction_id=args.instruction
    if instruction_id>=0 and dataset in ['mvsa-s','mvsa-m']:
        instruction=get_msa_instruction(instruction_id)
        answername=args.answer_name+str(instruction_id)
    elif instruction_id>=0 and dataset in ['t2015','t2017']:
        instruction=get_masa_instruction(instruction_id)
        answername=args.answer_name+str(instruction_id)
    elif instruction_id>=0 and dataset in ['masad']:
        instruction=get_masad_instruction(instruction_id)
        answername=args.answer_name+str(instruction_id)
    else:
        answername=args.answer_name

    prompt_id=args.prompt
    answerpath=args.answer_path
    map={'mvsa-s':3,'mvsa-m':3,'t2015':3,'t2017':3,'masad':2}
    for j in range(map[dataset]): 
        ans_file1=open(answerpath+answername+str(j+(prompt_id if prompt_id>=0 else 0)*map[dataset])+".jsonl","w")
        for i in range(len(image_files)):
            if prompt_id>=0:
                if dataset in ['mvsa-s','mvsa-m']:
                    prompt=get_all_prompt1(i,j+prompt_id*3,texts,captions)
                elif dataset in ['t2015','t2017']:
                    prompt=get_all_prompt2(i,j+prompt_id*3,aspects,texts,captions)
                else:
                    prompt=get_all_prompt3(i,j+prompt_id*2,aspects,texts,captions)
            else:
                if dataset in ['mvsa-s','mvsa-m']:
                    prompt=instruction.replace("{Text input}",texts[i])
                else:
                    prompt=instruction.replace("{Text input}",texts[i])
                    prompt=prompt.replace("{Aspect input}",aspects[i])
                    prompt=prompt.replace("\"target\"",aspects[i])

            #image_file = '/home/jncsnlp/lxf/dataset/MVSA_Single/data/1.jpg' # Image Path
            image_file=image_files[i]
            image = Image.open(image_file).convert('RGB')
            max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
            image = image.resize((max_edge, max_edge))

            image_tensor = process_images([image], image_processor)
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            
            tokenizer.padding_side = 'left'
            if not hasattr(tokenizer, 'pad_token_id'):
                tokenizer.pad_token_id = tokenizer.eos_token_id
            prompt = 'USER: <|image|>'+prompt+'ASSISTANT: '
            #prompt='<|image|>'+query


            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            #temperature = 0.01
            max_new_tokens = 1

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0, #temperature,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    output_scores=True, 
                    return_dict_in_generate=True,
                    )

            outputs = tokenizer.decode(output_ids.sequences[0, input_ids.shape[1]:]).strip()
            logits=output_ids.scores
            #print(logits)
            #print(logits.shape())
            #[positive,negative,neutral]
            #tokenizer(["positive","negative","neutral"]).input_ids
            #[[1, 6374], [1, 8178], [1, 21104]]
            ll=[logits[0][0,6374].float(),logits[0][0,8178].float(),logits[0][0,21104].float()]
            ll=torch.tensor(ll).cpu()
            scores = ll.numpy()
            #np.array(ll).to(device)
            import numpy as np
            softmax = np.exp(scores) / np.sum(np.exp(scores))
            #print(outputs)
            ans_file1.write(json.dumps({"id": ids[i],
                                        "image":images[i],
                                        "prompt":prompt, 
                                        "answer": outputs,
                                        "probability":str(softmax),
                                        #"rethink_answer":answer,,                     
                                        "label":labels[i]})+"\n")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default = "/home/jncsnlp/lxf/llava-v1.5-7b")
    parser.add_argument('--dataset', type=str, default = "")
    parser.add_argument('--tsv_path', type=str, default = None)
    parser.add_argument('--image_path', type=str, default = None)
    parser.add_argument('--answer_path', type=str, default = None)
    parser.add_argument('--answer_name', type=str, default = None)
    parser.add_argument('--prompt', type=int, default = 0)
    parser.add_argument('--instruction', type=int, default = -1)
    args = parser.parse_args()
    main(args)