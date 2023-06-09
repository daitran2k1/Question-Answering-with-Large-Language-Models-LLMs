import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel


model = None
tokenizer = None
generator = None
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def load_model(model_name, eight_bit=0, device_map="auto"):
    global model, tokenizer, generator

    print("Loading "+model_name+"...")

    #if device_map == "zero":
    #    device_map = "balanced_low_0"
    
    # config
    #gpu_count = torch.cuda.device_count()
    #print('gpu_count', gpu_count)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            'VietAI/gpt-neo-1.3B-vietnamese-news',
            #device_map=device_map,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_8bit=True,
            cache_dir="cache"    
            )#.cuda()
    
        model = PeftModel.from_pretrained(
            model, 
            model_name,
            torch_dtype=torch.float16,
            #device_map={'':0},
            device_map={"": device},
        )
        
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            'VietAI/gpt-neo-1.3B-vietnamese-news',
            #device_map=device_map,
            #device_map="auto",
            device_map={"": device},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_8bit=True,
            cache_dir="cache"    
            )
    
        model = PeftModel.from_pretrained(
            model, 
            model_name,
            torch_dtype=torch.float16,
            #device_map={'':0},
            device_map={"": device},
        )
        
    else:
        model = AutoModelForCausalLM.from_pretrained(
        'VietAI/gpt-neo-1.3B-vietnamese-news',
            #device_map=device_map,
            #device_map="auto",
            device_map={"": device},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_8bit=True,
            cache_dir="cache"    
            )
    
        model = PeftModel.from_pretrained(
            model, 
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device},
        )

    generator = model.generate

load_model("./lora-chatdoctor-5k/")

First_chat = "Nhân viên tổng đài của Viettel: Chào bạn, mình là nhân viên tổng đài của Viettel, mình có thể giúp gì cho bạn?"
print(First_chat)
history = []
history.append(First_chat)

def go():
    invitation = "Nhân viên tổng đài của Viettel: "
    human_invitation = "Khách hàng: "

    # input
    msg = input(human_invitation)
    print("")

    history.append(human_invitation + msg)

    fulltext = "Nếu bạn là nhân viên tổng đài của Viettel, hãy trả lời những câu hỏi về gói cước dựa trên mô tả của khách hàng. \n\n" + "\n\n".join(history) + "\n\n" + invitation
    #fulltext = "\n\n".join(history) + "\n\n" + invitation
    
    print('SENDING==========')
    print(fulltext)
    print('==========')

    generated_text = ""
    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
    in_tokens = len(gen_in)
    with torch.no_grad():
            generated_ids = generator(
                gen_in,
                max_new_tokens=200,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=0.5, # default: 1.0
                top_k = 50, # default: 50
                top_p = 1.0, # default: 1.0
                early_stopping=True,
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?

            text_without_prompt = generated_text[len(fulltext):]

    response = text_without_prompt

    response = response.split(human_invitation)[0]

    response.strip()

    print(invitation + response)

    print("")

    history.append(invitation + response)

while True:
    go()
