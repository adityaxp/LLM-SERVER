import re
import os
import sys
from llama_cpp import Llama
from logger import logtool


class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )

        sys.stdout = self.outnull_file        
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):        
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )

        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )

        self.outnull_file.close()
        self.errnull_file.close()

max_tokens = 512
temperature = 0.3
top_p = 0.1
echo = True
stop = ["Q", "\n"]
load_model = True
if load_model: 
    logtool.write_log("Loading model", "lawsage-v0.2-GGUF")
    law_sage_llama_model = Llama(model_path="models/lawsage-v0.2-GGUF/llama-2-7b-law-sage-v0.2.Q5_K_M.gguf")
    load_model = False
else:
    logtool.write_log("Model already loaded", "lawsage-v0.2-GGUF")

def lawsage_llama_cpp_pipeline(prompt):
    with suppress_stdout_stderr():

        logtool.write_log("Genrating response", "lawsage-v0.2-GGUF")
        model_output = law_sage_llama_model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
            stop=stop,
        )
        final_result = model_output["choices"][0]["text"].strip()
        return final_result


def parse_generated_text(generated_text):
    logtool.write_log("Parsing response", "lawsage-v0.2-GGUF")
    generated_text_value = generated_text
    inst_pattern = r'\[INST\](.*?)\[/INST\]'
    inst_matches = re.findall(inst_pattern, generated_text_value)
    remaining_text = re.sub(inst_pattern, '', generated_text_value)
    return inst_matches, remaining_text.strip()


def get_lawsage_llama_cpp_response(prompt):
    result = lawsage_llama_cpp_pipeline(prompt)
    _ , response = parse_generated_text(result)
    return response