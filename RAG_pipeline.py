from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,

)
import torch

from logger import logtool
import re
import context




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logtool.write_log(f"Using device: {str(device)}", "RAG")

load_model = True

system_prompt = "You are an AI powered legal advisory chatbot named Law Sage. Only provide responses in a correct leagal context"
model_name = "meta-llama/Llama-2-7b-chat-hf"
hf_auth = "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"



if load_model:
    logtool.write_log("Loading model in nf4", "RAG")
    
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token = hf_auth,
        cache_dir="models/meta-llama/Llama-2-7b-chat-hf"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    logtool.write_log(f"Using model : {model_name}", "RAG")

    logtool.write_log("Loading tokenizer", "RAG")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    logtool.write_log("Creating RAG pipeline", "RAG")

    RAG_pipeline = pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        max_new_tokens=512,
        repetition_penalty=1.1
    )
else:
    logtool.write_log("Model already loaded", "RAG")
    logtool.write_log(f"Using model : {model_name}", "RAG")



def parse_generated_text(generated_text):
    generated_text_value = generated_text['generated_text']
    inst_pattern = r'\[INST\](.*?)\[/INST\]'
    inst_matches = re.findall(inst_pattern, generated_text_value)
    remaining_text = re.sub(inst_pattern, '', generated_text_value)
    return inst_matches, remaining_text.strip()


def get_RAG_response(query):
    logtool.write_log("Generating RAG response", "RAG")
    results = context.get_contex(query)
    ref = results[0]
    answer = RAG_pipeline(f"""[INST]  <<SYS>> {system_prompt} <</SYS>> Answer this question {query} based on the given context {ref} [/INST]""")
    generated_text_input = answer[0]
    logtool.write_log("Parsing response", "RAG")
    inst_contents, RAG_response = parse_generated_text(generated_text_input)
    return RAG_response


# query = "Give preamble of india constitution"
# print(get_RAG_response(query))