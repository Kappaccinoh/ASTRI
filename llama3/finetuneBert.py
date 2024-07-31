from transformers import BertTokenizer, BertLMHeadModel
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from textwrap import dedent
from typing import Dict
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, BitsAndBytesConfig, pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, PeftModel, get_peft_model, prepare_model_for_kbit_training
import ast

new_model = "Bert-V1"
pad_token = "<|pad|>"
access_token = "a"
prompt_txt = "Imagine you are a machine that parses traffic notices. Return unique lists of streets from sentences inside the Notice. Here is an example output: {'CN1': '[香港仔大道', '奉天街'], 'EN1': ['Aberdeen Main Road', 'Fung Tin Street']}"
excel = 'Datasets/Clearways_train_EN_CN_New_Format.xlsx'

quant_config = BitsAndBytesConfig(
    load_in_4bit = True, bnb_4bit_quant_type = "nf4", bnb_4bit_compute_dtype = torch.bfloat16
)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertLMHeadModel.from_pretrained('bert-base-cased')


def format_example (row: Dict):
    prompt = dedent(
        f"""
    {str(row["Notice"])}
    """
    )

    messages = [
        {
            "role" : "system",
            "content" : prompt_txt
        },
        # {
        #     "role" : "system",
        #     "content" : "Imagine you are a machine that parses traffic notices. Return unique lists of streets from sentences inside . Here is an example output: {'EN1': ['Aberdeen Main Road', 'Fung Tin Street']}"
            
        # },
        {
            "role" : "user",
            "content" : prompt
        },
        {
            "role" : "assistant",
            "content" : str(row["Sample Output"])
        }
    ]

    return tokenizer.apply_chat_template(messages, tokenize = False)


def count_tokens(row : Dict) -> int:
    return len(
        tokenizer(
            row["text"],
            add_special_tokens = True,
            return_attention_mask = False
        )["input_ids"]
    )

df = pd.read_excel(excel , sheet_name='Sheet1')
df["text"] = df.apply(format_example, axis = 1)
df["token_count"] =  df.apply(count_tokens, axis = 1)

print(df.head())

print(df.text.iloc[0])


train, eval_test = train_test_split(df, test_size = 0.2, random_state = 42)
test, eval = train_test_split(eval_test, test_size = 0.5, random_state = 42)

train.sample(n=80).to_json("train.json", orient="records", lines = True)
test.sample(n=10).to_json("test.json", orient="records", lines = True)
eval.sample(n=10).to_json("validation.json", orient="records", lines = True)


dataset = load_dataset(
    "json",
    data_files = {"train" : "train.json", "test" : "test.json", "validation" : "validation.json"}
)

print(dataset)


pipe = pipeline(
    task = "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 256,
    return_full_text = False
)

def create_test_prompt(row):
    prompt = dedent(
        f"""
    {row["Notice"]}
    """
    )
    messages = [
        {
            "role" : "system",
            "content" : prompt_txt
            
        },
        # {
        #     "role" : "system",
        #     "content" : "Imagine you are a machine that parses traffic notices. Return unique lists of streets from sentences inside. Here is an example output: {'EN1': ['Aberdeen Main Road', 'Fung Tin Street']}"
            
        # },
        {
            "role" : "user",
            "content" : prompt
        }
    ]

    return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    

row = dataset["test"][0]
prompt = create_test_prompt(row)
# print(prompt)
outputs = pipe(prompt)
response = f"""
Ground truth: {row["Sample Output"]}
prediction: {outputs[0]["generated_text"]}
"""

# print(response)

rows = []
for row in tqdm(dataset["test"]):
    prompt = create_test_prompt(row)
    outputs = pipe(prompt)
    rows.append(
        {
        "Notice" : row["Notice"],
        "prompt" : prompt,
        "Sample Output" : row["Sample Output"],
        "untrained_prediction" : outputs[0]["generated_text"]
        }
    )

predictions_df = pd.DataFrame(rows)
predictions_df.to_excel("predictions.xlsx", index=False)


response_template = "<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer = tokenizer)

examples = [dataset["train"][0]["text"]]
encodings = [tokenizer(e) for e in examples]

dataloader = DataLoader(encodings, collate_fn = collator, batch_size = 1)

batch = next(iter(dataloader))
# print(batch.keys())
# print(batch["labels"])


lora_config = LoraConfig(
    r = 32,
    lora_alpha = 16,
    target_modules = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    ],
    lora_dropout = 0.05,
    bias = "none",
    task_type = TaskType.CAUSAL_LM
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


OUTPUT_DIR = "tests"

sft_confg = SFTConfig(
    output_dir = OUTPUT_DIR,
    dataset_text_field = "text",
    max_seq_length = 2048,
    num_train_epochs = 10,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps = 4,
    optim = "adamw_hf",
    eval_strategy = "steps",
    eval_steps = 0.2,
    save_steps = 0.2,
    logging_steps = 10,
    learning_rate = 1e-4,
    fp16 = True,
    save_strategy = "steps",
    warmup_ratio = 0.1,
    save_total_limit = 2,
    lr_scheduler_type = "linear",
    save_safetensors = True,
    dataset_kwargs = {
        "add_special_tokens": False,
        "append_concat_token" : False
    },
    seed = 42
)

trainer = SFTTrainer(
    model = model,
    args = sft_confg,
    train_dataset = dataset["train"],
    eval_dataset = dataset["validation"],
    tokenizer = tokenizer,
    data_collator = collator
)

# def evaluate_on_validation(model_dir):
#     generator = pipeline(
#         task = "text-generation",
#         model = model_dir,
#         tokenizer = model_dir,
#         max_new_tokens = 256,
#         return_full_text = False
#     )
#     total_loss = 0.0
#     criterion = nn.CrossEntropyLoss()
#     with torch.no_grad():
#         for row in tqdm(dataset["validation"]):
#             print(row)
#             input =  create_test_prompt(row)
#             labels = row["Sample output"]
#             outputs = generator(input)
#             loss = criterion(outputs[0]["generated_text"], labels)
#             total_loss += loss.item()

#     avg_loss = total_loss / len(dataset["validation"])
#     return avg_loss




# best_val_loss = float("inf")
# patience = 5 # Number of epochs without improvement before stopping
# for epoch in range(100):
#     if epoch >= 1:
#         trainer = SFTTrainer(
#         model = new_model,
#         args = sft_confg,
#         train_dataset = dataset["train"],
#         eval_dataset = dataset["validation"],
#         tokenizer = tokenizer,
#         data_collator = collator
#         )
#     trainer.train()
#     trainer.save_model(new_model)
#     val_loss = evaluate_on_validation(new_model)  # Implement your own validation evaluation function
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#     else:
#         patience -= 1
#         if patience == 0:
#             print("Early stopping triggered. Training halted.")
#             break
trainer.train()

trainer.save_model(new_model)

# tokenizer = AutoTokenizer.from_pretrained(new_model)
# model = AutoModelForCausalLM.from_pretrained(
#     new_model,
#     torch_dtype = torch.float16,
#     device_map = "auto"
# )

# model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of = 8)
# model = PeftModel.from_pretrained(model, new_model)
# model = model.merge_and_unload()

model = AutoModelForCausalLM.from_pretrained(
    new_model,
    quantization_config = quant_config,
    device_map = "auto"
)

pipe = pipeline(
    task = "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 256,
    return_full_text = False
)

row = dataset["test"][0]
prompt = create_test_prompt(row)
print(prompt)

response = f"""
Ground truth: {row["Sample Output"]}
prediction: {outputs[0]["generated_text"]}
"""

print(response)

predictions = []
for row in tqdm(dataset["test"]):
    prompt = create_test_prompt(row)
    outputs = pipe(prompt)
    predictions.append({
        "Llama Prediction" : outputs[0]["generated_text"],
        "Ground Truth" : row["Sample Output"]
        })

predictions_df = pd.DataFrame(predictions)
predictions_df.to_excel("predictions_llama.xlsx", index=False)


def eval_llama(df, gt, out):
    ground_truth = df[gt]
    llama_pred = df[out]
    score = 0
    count = 0

    for row_gt, row_pr in zip(ground_truth, llama_pred):
        gt_map = ast.literal_eval(row_gt)
        pr_map = ast.literal_eval(row_pr)
        flag = True
        if pr_map.keys() == gt_map.keys():
            for k in gt_map:
                if k not in pr_map or set(gt_map[k]) != set(pr_map[k]) or len(gt_map[k]) != len(pr_map[k]):
                    flag = False
        else:
            flag = False
                
        if flag == True:
            score += 1
        count += 1
                    
    return score / count

gt = "Ground Truth"
pred = "Llama Prediction"
  
accuracy = eval_llama(predictions_df, gt, pred)

print(f"Accuracy: {accuracy:.2f}")
