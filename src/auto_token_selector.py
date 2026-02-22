from typing import Sequence, Tuple, Optional

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

def load_tokenizer(model_name : str):
    '''
        Load Tokenizer according to your choice
        You should take place your path here.
    '''
    name = model_name.lower

    if ("llama" in name) and AutoTokenizer is not None:
        try:
            tok = AutoTokenizer.from_pretrained(" Your path here ")
            return tok, "hf"
        except Exception as e:
            raise RuntimeError(f"Not Found HuggingFace tokenizer: {e}")
    
    if ("pythia" in name) and AutoTokenizer is not None:
        try:
            tok = AutoTokenizer.from_pretrained(" Your path here ")
            return tok, "hf"
        except Exception as e:
            raise RuntimeError(f"Not Found HuggingFace tokenizer: {e}")
    
    raise RuntimeError(f"Not load any tokenizer. Please Check your path or method. We only support two methods : 'llama' and 'pythia'.")

def count_tokens(text : str, tokenizer, backend_tag : str) -> int:
    if backend_tag == "openai":

        # prepare for the openai
        return len(tokenizer.encode(text))  
    elif backend_tag == "hf":
        return len(tokenizer.encode(text, add_special_tokens=False))
    else:
        raise ValueError(f"Not Found backend_tag: {backend_tag}")
    
def choose_bucket(
        n_tokens : int,
        buckets : Sequence[int] = (32, 64, 128, 256),
        safety : float = 1.1
) -> int:
    '''
        choose the best buckets
    '''
    need = int(n_tokens * safety + 0.9999)
    for b in sorted(buckets):
        if need <= b:
            return b
    return max(buckets)

def select_token_size(
    text: str,
    model_name: str,
    buckets: Sequence[int] = (32, 64, 128, 256),
    safety: float = 1.1,
    tokenizer=None,
    backend_tag: Optional[str] = None
) -> Tuple[int, int]:
    if tokenizer is None or backend_tag is None:
        tokenizer, backend_tag = load_tokenizer(model_name)
    
    n = count_tokens(text, tokenizer, backend_tag)
    b = choose_bucket(n, buckets=buckets, safety=safety)
    return n, b
