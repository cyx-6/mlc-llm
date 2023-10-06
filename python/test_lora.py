from mlc_chat import ChatModule
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import torch
from collections import defaultdict
import time


class TestResult:
    def __init__(self):
        self.success = 0
        self.failure = 0
        self.malformed = 0
        self.error = 0


def parse_response_for_action(response):
    # get the 'action' from the ai response
    action = response.split('"action": "')[1].split('"')[0]
    # remove anything after the number in the ai_action (i.e. TYPESUBMIT 82 "Go to Gmail" -> TYPESUBMIT 82)
    action_split = action.split(" ")
    if len(action_split) > 1:
        return action_split[0] + " " + action_split[1]
    else:
        return action_split[0]


def get_num_tokens(input_text, tokenizer):
    tokens = tokenizer(input_text.strip(), return_tensors="pt")["input_ids"]
    return len(tokens.flatten())


def analyze_results(data, ai_responses, results, tokenizer):
    for i, (line, ai_response) in enumerate(zip(data, ai_responses)):
        # print(ai_response)
        json_line = json.loads(line)
        prompt = json_line["prompt"]
        response = json_line["response"]
        # get the number of tokens in the prompt and print it
        num_tokens = get_num_tokens(prompt, tokenizer)
        print(f"[{i}] Evaluating prompt, Number of tokens in prompt: {num_tokens}")
        print(
            f'Match count: {results["totals"].success}, Fail count: {results["totals"].failure}, Error count: {results["totals"].error}'
        )
        if ai_response is None:
            results["totals"].error += 1
            results[i].error += 1
            print("ERROR: response was None.")
            continue
        try:
            try:
                ai_action = parse_response_for_action(ai_response)
            except Exception as e:
                results["totals"].failure += 1
                results["totals"].malformed += 1
                results[i].failure += 1
                results[i].malformed += 1
                print(f"FAILURE: Malformed reponse, {e}. Response was: {ai_response}")
                continue

            try:
                original_action = parse_response_for_action(response)
            except Exception as e:
                results["totals"].error += 1
                results[i].error += 1
                print(f"ERROR: Malformed groundtruth reponse, {e}. Response was: {ai_response}")
                continue

            # check if the actions match and keep a tally
            if ai_action == original_action:
                results["totals"].success += 1
                results[i].success += 1
                print(f"SUCCESS: AI action: {ai_action}, Original action: {original_action}")
            else:
                results["totals"].failure += 1
                results[i].failure += 1
                print(f"FAILURE: AI action: {ai_action}, Original action: {original_action}")

        except Exception as e:
            print(f"ERROR: Exception occured, {e}. Response was: {ai_response}")
            results["totals"].error += 1
            results[i].error += 1

    print(f'Number of success: {results["totals"].success}')
    print(f'Number of failure: {results["totals"].failure}')
    print(f'Number of failures due to malformed response: {results["totals"].malformed}')
    print(f'Number of errors: {results["totals"].error}')
    print()

    if len(data) > 0:
        print(
            f'Percentage of matches: {(results["totals"].success / (results["totals"].success + results["totals"].failure + results["totals"].error)) * 100}%'
        )
    else:
        print(f"Percentage of matches: 0%")
    return results


class Timer:
    def __init__(self):
        self.time = time.time()

    def get(self):
        current_time = time.time()
        print(current_time - self.time)
        self.time = current_time


def test_hf():
    timer = Timer()

    with open("/home/ycai/models/google_specific_data_train.jsonl", "r") as f:
        data = f.readlines()

    timer.get()

    data = data[:30]

    # model_id = "/opt/models/llama-2/llama-2-13b-otherside-2/"
    model_id = "/opt/models/otherside/llama-2-13b-otherside/"
    lora_id = "/home/ycai/models/checkpoint-220"
    # lora_id = "/home/ycai/models/checkpoint-280"
    # quantization_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_compute_dtype=torch.float16
    # )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # quantization_config=quantization_config,
        device_map="auto",
    )
    print("loaded base model")
    timer.get()
    model = PeftModel.from_pretrained(model, lora_id)
    model = model.merge_and_unload()
    print("loaded lora")
    timer.get()
    model.half()
    # model.to_bettertransformer() #Mabye comment this out if you can get it to fit in GPU mem

    response = []
    for idx, line in enumerate(data):
        prompt = json.loads(line)["prompt"]
        print(f"Index: {idx}, Response:")
        inputs = tokenizer(prompt.strip(), return_tensors="pt").to("cuda")

        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=400,
            do_sample=True,
            top_p=0.9,
            temperature=0.1,
        )
        output = output[0].to("cpu")
        response.append(tokenizer.decode(output))
        # print(tokenizer.decode(output))
    results = defaultdict(lambda: TestResult())
    results = analyze_results(data, response, results, tokenizer)
    for i in range(len(data)):
        print(
            f"Prompt {i}, Success: {results[i].success}, Failure: {results[i].failure}, Malformed: {results[i].malformed}, Error: {results[i].error}"
        )

    print(
        f"Total Success: {results['totals'].success}, Total Failure: {results['totals'].failure}, Total Malformed: {results['totals'].malformed}, Total Error: {results['totals'].error}"
    )
    timer.get()


def test_tvm():
    with open("/home/ycai/models/google_specific_data_train.jsonl", "r") as f:
        data = f.readlines()

    data = data[:30]
    model_id = "llama-2-13b-otherside-2-q4f16_1"
    # lora_id = "/home/ycai/models/checkpoint-220"
    lora_id = "/home/ycai/models/checkpoint-280"
    cm = ChatModule(model=model_id)
    cm.apply_lora(lora_id)

    response = []
    for idx, line in enumerate(data):
        prompt = json.loads(line)["prompt"]
        print(f"Index: {idx}, Response:")

        output = cm.generate(prompt=prompt.strip())
        print(output)
        print()
        response.append(output)
    results = defaultdict(lambda: TestResult())
    results = analyze_results(
        data,
        response,
        results,
        AutoTokenizer.from_pretrained("/opt/models/llama-2/llama-2-13b-otherside-2/"),
    )
    for i in range(len(data)):
        print(
            f"Prompt {i}, Success: {results[i].success}, Failure: {results[i].failure}, Malformed: {results[i].malformed}, Error: {results[i].error}"
        )

    print(
        f"Total Success: {results['totals'].success}, Total Failure: {results['totals'].failure}, Total Malformed: {results['totals'].malformed}, Total Error: {results['totals'].error}"
    )


if __name__ == "__main__":
    test_hf()
    # test_tvm()
