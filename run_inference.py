'''
Adapted from https://github.com/kojima-takeshi188/zero_shot_cot
'''

import argparse
import wandb
from utils import *

def kant_v1(args, x):
    """Apply the Kant-inspired method in a single shot and return the generated answer."""
    understand_prompt = "First, let's briefly understand this question step by step."
    plan_prompt = "Let's make a briefly plan to solve this question step by step."
    action_prompt = "Now, let's execute the plan step by step."
    check_prompt = "Check your answers and correct possible errors."
    kant_prompt =  (f"You are an assistant with a multi-level thinking structure. When you receive the problem, "
              f"don't solve it immediately. Follow the structured instructions: \n\n"
              f"{understand_prompt}: [Your understanding here.]\n"
              f"{plan_prompt}: [Your plan here.]\n"
              f"{action_prompt}: [Your solution here.]\n"
              f"{check_prompt}: [Your reflect here.]\n")
    input_mes = [{"role":"system", "content" : kant_prompt},
              {"role":"user", "content" : x}]
    return input_mes

def kant_v2(args, x):
    """Apply the Kant-inspired method in a single shot and return the generated answer."""
    cate_prompt = (
        "1.(First, briefly understand this question in the context of time and space step by step.)\n"
        "Quantity (What entitie/events and their quantitative relationships are related to the question?): [Your answer here]\n"
        "Quality (What intrinsic properties and external constraints of these entities/events are related to the question?): [Your answer here]\n"
        "Relation (What is the relationship between these entities/events?): [Your answer here]\n"
        "Modality (Is possibility/impossibility, inevitable/accidental involved in the entities/events related to the question?): [Your answer here]\n"
    )
    plan_prompt = (
        "2.Plan (Integrate the above analysis, make a briefly plan to solve this question step by step.)\n"
    )
    action_prompt = (
        "3.Action (Excute the plan step by step to solve the problem.) "
    )
    reflect_prompt = (
        "4.Review (Based on your experience, check the above answers for possible errors and revise them.)"
    )
    kant_prompt =  (f"You are an assistant operating under a Kantian-inspired multilevel thinking structure. When receiving a problem, do not solve it immediately. Instead, follow these structured instructions: \n\n"
                    f"{cate_prompt}\n"
                    f"{plan_prompt}: [Your plan here]\n"
                    f"{action_prompt}: [Your solution here]\n"
                    f"{reflect_prompt}: [Your reflect here]\n")

    input_mes = [{"role": "system", "content" : kant_prompt},
                 {"role": "user", "content" : x}]
    return input_mes


def main():
    args = parse_arguments()
    wandb.init(
        # set the wandb project where this run will be logged
        project="iclr",
        # track hyperparameters and run metadata
        config= args
    )
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    print("OPENAI_API_KEY:")
    print(os.getenv("OPENAI_API_KEY")[0:5] + '**********')
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder()
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    total = 0
    correct_list = []

    for i, data in enumerate(dataloader):
        if i < args.resume_id - 1:
            continue
        
        print('*************************')
        print("{}st data".format(i+1))
                
        # Prepare question template ...
        x, y = data
        x = "Q: " + x[0] + "\n" + "A:"
        y = y[0].strip()
                    
        if args.method == "zero_shot":
            x = x + " " + args.direct_answer_trigger
            x_dict =[{"role": "user", "content": x}]
        elif args.method == "zero_shot_cot":
            x = x + " " + args.cot_trigger
            x_dict =[{"role": "user", "content": x}]
        elif args.method == "kantv1":
            x_dict = kant_v1(args, x)
        elif args.method == "kantv2":
            x_dict = kant_v2(args, x)
        else:
            raise ValueError("method is not properly defined ...")
        print(x)
        # Answer experiment by generating text ...
        z = decoder.decode(args, x_dict)
        print(z)
        # Answer extraction for zero-shot-cot ...
        if args.method in ["zero_shot_cot","kantv1","kantv2"]:
            z = x + z + " " + args.direct_answer_trigger
            z_dict = [{"role": "user", "content": z}]
            pred = decoder.decode(args, z_dict)
        else:
            pred = z

        # Clensing of predicted answer ...
        pred = answer_cleansing(args, pred)
        
        # Choose the most frequent answer from the list ...
        print("pred : {}".format(pred))
        print("GT : " + y)
        print('*************************')
        
        # Checking answer ...
        correct = (np.array([pred]) == np.array([y])).sum().item()
        if(correct == 0):
            print("FALSE!")
        else:
            print("True!")
        correct_list.append(correct)
        total += 1 #np.array([y]).size(0)
        
        if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            break
            #raise ValueError("Stop !!")

        # Calculate accuracy ...
        accuracy = (sum(correct_list) * 1.0 / total) * 100
        print("accuracy : {}".format(accuracy))
        wandb.log({"acc": accuracy})
    wandb.finish()
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="multiarith", 
        choices=["gsm8k","aqua","commonsensqa","strategyqa", 
                 "gsm8k-hard","bbh-cj"], 
        help="dataset used for experiment"
    )

    parser.add_argument(
        "--resume_id", type=int, default=0, 
        help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    parser.add_argument(
        "--minibatch_size", type=int, default=1, choices=[1], 
        help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request"
    )
    
    parser.add_argument("--max_num_worker", type=int, default=0, 
                        help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--model", type=str, default="gpt3.5", 
        choices=["gpt3.5", "gpt4"], 
        help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    
    parser.add_argument(
        "--method", type=str, default="auto_cot",
        choices=["zero_shot", "zero_shot_cot", "kantv1", "kantv2"], 
        help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment/log", 
        help="output directory"
    )

    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, 
        help="sleep between runs to avoid excedding the rate limit of openai api"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, 
        help="temperature for GPT-3"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "gsm8k-hard":
        args.dataset_path = "./dataset/hard/gsm8k-hard.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bbh-cj":
        args.dataset_path = "./dataset/causal_judgement.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    args.cot_trigger = "Let's think step by step."
    
    return args

if __name__ == "__main__":
    main()