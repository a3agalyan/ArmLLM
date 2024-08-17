import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from datasets import load_dataset
import evaluate
from tqdm import tqdm

# model_name = 'meta-llama/Meta-Llama-3-8B'
# model_name = 'google/gemma-2-9b'
# model_name = 'google/gemma-2-27b'
model_name = 'meta-llama/Meta-Llama-3-70B'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"
tokenizer.truncation_side = "left"
tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='cuda',
    quantization_config=quantization_config,
)

dataset = load_dataset("EdinburghNLP/xsum")
validation_dataset = dataset['validation'].shuffle().select(range(32)).map(lambda x: {'doclen': len(x['document']), **x}).sort('doclen')

rouge = evaluate.load('rouge')

batch_size = 8
for begin_idx in tqdm(range(0, len(validation_dataset), batch_size)):
    batch = validation_dataset[begin_idx:begin_idx+batch_size]
    batch_documents = batch['document']
    batch_summaries = batch['summary']

    batch_inputs = [
        document + ' TL;DR: '
        for document in batch_documents
    ]
    batch_inputs = tokenizer(
        batch_inputs,
        return_tensors='pt',
        padding=True,
    ).to('cuda')

    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model.generate(**batch_inputs, max_new_tokens=32, do_sample=False)

    for output, reference in zip(outputs, batch_summaries):
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        orig_input, tldr, prediction = decoded_output.partition(' TL;DR: ')
        rouge.add(references=reference, predictions=prediction)

print(rouge.compute())