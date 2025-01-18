import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import transformers
from dataclasses import dataclass, field

from typing import List, Tuple

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self,args, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.args=args

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        print(image_file)
        print(qs)
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        
        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return index, input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


@dataclass
class DataCollatorForVisualTextGeneration(object):
    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=batch_first,
            padding_value=padding_value)
        print(input_ids.size())
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self,
                 batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        indices, input_ids, images = zip(*batch)
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        images = torch.stack(images, dim=0)
        print(input_ids.size())
        print(images.size())
        return indices, input_ids, images

# DataLoader
def create_data_loader(args,questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    dataset = CustomDataset(args,questions, image_folder, tokenizer, image_processor, model_config)
    collator = DataCollatorForVisualTextGeneration(tokenizer=tokenizer)
    data_loader = DataLoader(dataset, collate_fn=collator, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    questions = [json.loads(q) for q in open(args.question_file, "r")]
    #print(questions)    
    print(len(questions))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    #print(questions)
    #os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # set padding side to `left` for batch text generation
    model.config.tokenizer_padding_side = tokenizer.padding_side = "left"

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')


    data_loader = create_data_loader(
        args,
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    #count=0
    for indices, input_ids, image_tensor in tqdm(data_loader):
        #count=count+1
        #print(count)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids.to(device='cuda', non_blocking=True),
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        #print(output_ids)
        '''texts=[]
        for a in range(output_ids.size()[0]):
            outputs = tokenizer.decode(output_ids[a, input_ids.shape[1]:],skip_special_tokens=True).strip()
            texts.append(outputs)'''
        print(output_ids.size())
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        print(output_ids[:, input_token_len:].size())
        print(outputs)
        #print(texts)
            
        '''
        print(input_token_len)
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )'''
        '''print(args.max_new_tokens)
        print(output_ids[:, args.max_new_tokens:])
        outputs = tokenizer.batch_decode(output_ids[:, args.max_new_tokens:], skip_special_tokens=True)
            #print(outputs)'''
        #print(outputs)
        '''outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()'''
        
        for index, output in zip(indices, outputs):
            #print(index)
            #print(output)
            line = questions[index]
            idx = line["id"]
            cur_prompt = line["text"]
            label=line["label"]
            image=line["image"]
            #answer=line["answer"]
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"id": idx,
                                    "image":image,
                                    "prompt": cur_prompt,
                                    "answer": output.strip(),
                                    #"rethink_answer":answer,,
                                    "label":label})+"\n")
                                    #"model_id": model_name,
                                    #"metadata": {}}) + "\n")
        ans_file.flush()
        #return
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/jncsnlp/lxf/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/jncsnlp/lxf/dataset/IJCAI2019_data/twitter2015_images/")
    parser.add_argument("--question-file", type=str, default="/home/jncsnlp/lxf/data/t2015/test.jsonl")
    parser.add_argument("--answers-file", type=str, default="/home/jncsnlp/lxf/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=20)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    eval_model(args)
