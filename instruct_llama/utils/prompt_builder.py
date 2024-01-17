from typing import List, TypedDict, Literal, Tuple

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from instruct_llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str
    
Dialog = List[Message]

B_INST, E_INST = '[INST]', '[/INST]'
B_SYS, E_SYS = '<<SYS>>\n', '\n<</SYS>>\n\n'

def maybe_add_system_prompt(dialog: Dialog) -> Dialog:
    """Try to insert an empty system prompt at the beginning to make code consistent."""
    
    if dialog[0]["role"] != "system":
        dialog = [
            {
                "role": "system",
                "content": "",
            }
        ] + dialog
        
    return dialog        

def build_prompt_completion(dialog: Dialog, tokenizer: Tokenizer) -> Tuple[List[int], List[int]]:
    """Build prompt and completion pair following the Meta llama-2 format.
    
    Note we only build the training target completion if that last tole in the dialog is 'assistant'.
    
    Here are some examples of the format that llama-2 uses (before apply tokenization), note we inserted BOS and EOS into example for completeness:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} </s>"}
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST] {1st response} </s><s>[INST] {2nd user prompt} [/INST]", "completion": " {2nd response} </s>"}
    """
    
    assert dialog is not None and len(dialog) >= 1
    
    dialog = maybe_add_system_prompt(dialog)
    
    assert len(dialog) >= 2
    
    assert(
        dialog[0]['role'] == "system"
        and all([msg["role"] == "user" for msg in dialog[1::2]]) 
        and all([msg['role'] == 'assistant' for msg in dialog[2::2]])
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u ...)"
    )
    
    # store user-prompt: answer pairs so we can later add BOS, EOS tokens to each pair
    prompts = []
    for i in range(1, len(dialog), 2): # skip the first one since it's system prompt
        full_prompt = ""
        prompt = dialog[i]
        
        if i == 1:
            # add systemp prompt, note Meta llama-2 insert the system prompt inside the first user prompt
            # as in this format: [INST] <<SYS>>\n{systemp prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]
            sys_prompt = dialog[0]['content'].strip()
            if len(sys_prompt) > 0:
                sys_prompt = B_SYS + sys + E_SYS
            else:
                sys_prompt = ''
            full_prompt = f"{B_INST} {sys_prompt} {(prompt['content']).strip()} {E_INST}"
        else:
            full_prompt = f"{B_INST} {(prompt['content']).strip()} {E_INST}"
        
        # add the answer to the full prompt
        # here we skip the last answer by the assistant, since it's used for building the training target
        if i + 1 < len(dialog) - 1 and dialog[i+1]["role"] == "assistant":
            answer = dialog[i+1]
            full_prompt += f" {(answer['content']).strip()}"
            
        prompts.append(full_prompt)
        
    # Concatenate and tokenize the full prompt, note llama-2, we add BOS and EOS for every pair of user-prompt:answer, except for the last user-prompt
    prompt_tokens = []
    for i, prompt in enumerate(prompts):
        if i < len(prompts - 1):
            tokens = tokenizer.encode(prompt, bos=True, eos=True)
        else:
            tokens = tokenizer.encode(prompt, bos=True, eos=False)
            
        prompt_tokens.extend(tokens)
        
    # build completion tokens for training
    completion_tokens = None
    if dialog[-1]['role'] == "assistant":
        answer = dialog[-1]
        target = f" {(answer['content']).strip()}"
        completion_tokens = tokenizer.encode(target, bos=False, eos=True)
        
    return prompt_tokens, completion_tokens