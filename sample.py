"""
Sample from a trained model
"""
import os
import pickle
import argparse
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from tunalab import compose_config, load_checkpoint, get_default_device, to_device

def main():
    # -----------------------------------------------------------------------------
    # Configuration setup
    parser = argparse.ArgumentParser(description='Sample from a trained GPT model')
    
    parser.add_argument('--init_from', type=str, default='resume', help='resume (from out_dir) or gpt2 variant')
    parser.add_argument('--out_dir', type=str, default='out', help='Directory to load checkpoint from')
    parser.add_argument('--start', type=str, default='\n', help='Start text')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='Max new tokens per sample')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--dtype', type=str, default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16', help='Data type')
    parser.add_argument('--compile', type=bool, default=False, help='Compile model')
    
    config = compose_config(parser)
    
    # -----------------------------------------------------------------------------

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    
    # Device handling
    if config.device:
        try:
            device = torch.device(config.device)
        except RuntimeError:
            device = get_default_device()
    else:
        device = get_default_device()
        
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    if config.init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
        
        # Load metadata first to get model args
        checkpoint = load_checkpoint(ckpt_path)
        
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        
        # Load weights into initialized model
        load_checkpoint(ckpt_path, model=model)
    elif config.init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(config.init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if config.compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if config.init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    start_text = config.start
    if start_text.startswith('FILE:'):
        with open(start_text[5:], 'r', encoding='utf-8') as f:
            start_text = f.read()
    start_ids = encode(start_text)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(config.num_samples):
                y = model.generate(x, config.max_new_tokens, temperature=config.temperature, top_k=config.top_k)
                print(decode(y[0].tolist()))
                print('---------------')

if __name__ == "__main__":
    main()
