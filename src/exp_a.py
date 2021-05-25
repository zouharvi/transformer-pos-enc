#!/usr/bin/env python3
import sys
sys.path.append("src")
from utils import TEXT_WIKIPEDIA
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2').to(DEVICE)
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(DEVICE)
model.eval()

choice_bag = "A B C D".split()
input_base = "A B C".split()


# for input_size in [16]:
for input_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    input_prefix = []
    # 1
    # for _ in range(input_size-len(input_base)):
    #     input_prefix.append(random.choice(choice_bag))
    # input_new = input_prefix+input_base
    # 2
    # while len(input_base) + len(input_prefix) + len(choice_bag) < input_size:
    #     input_prefix += list(choice_bag)
    # input_new = input_prefix+input_base
    # 3
    # while len(input_base) + len(input_prefix) + len(choice_bag)+1 < input_size:
    #     random.shuffle(choice_bag)
    #     input_prefix += list(choice_bag)+["|"]
    # random.shuffle(input_base)
    # input_new = input_prefix+input_base
    # 4
    # for _ in range(input_size-len(input_base)-len(choice_bag)-1):
    #     input_prefix.append("-")
    # input_new = input_prefix+"D B C A | D B C".split()
    # input_new = input_prefix+"A D B C | A D B".split()

    # 1,2,3,4
    # inputs = tokenizer(" ".join(input_new), return_tensors="pt").to(DEVICE)["input_ids"]
    # outputs = model.generate(inputs, max_length=len(input_new)+1)    
    # outputs_text = tokenizer.decode(outputs[0]).split()
    # if input_size >= 32:
    #     outputs_text = outputs_text[len(input_prefix)+len(input_base):]
    # outputs_text = outputs_text[len(input_new):]

    # 5
    # inputs = tokenizer(" ".join(TEXT_WIKIPEDIA.split()[-input_size+1:]), return_tensors="pt").to(DEVICE)["input_ids"]
    inputs = tokenizer(TEXT_WIKIPEDIA, return_tensors="pt").to(DEVICE)["input_ids"][:,-input_size+1:]
    print(inputs.shape)
    outputs = model.generate(inputs, max_length=1024)    
    outputs_text = tokenizer.decode(outputs[0])

    # 5
    end_index = outputs_text.index("188,855")
    print(outputs_text[end_index:].split()[:2])