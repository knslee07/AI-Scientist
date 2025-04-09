import random
from tqdm import tqdm
from utils import call_api, load_model

# Set random seed for reproducibility
random.seed(2024)

## Step 1: Generate synthetic test examples
def generate_testset():
    test_data = [
        {
            "input": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "output": "Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72"
        },
        {
            "input": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "output": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10"
        },
        {
            "input": "Tim has 30 less apples than Martha, and Harry has half as many apples as Tim. If Martha has 68 apples, how many apples does Harry have?",
            "output": "Tim has 68-30 = <<68-30=38>>38 apples. Harry has 38/2 = <<38/2=19>>19 apples. #### 19"
        },
        {
            "input": "Four people lost a total of 103 kilograms of weight. The first person lost 27 kilograms. The second person lost 7 kilograms less than the first person. The two remaining people lost the same amount. How many kilograms did each of the last two people lose?",
            "output": "Second person = 27 - 7 = <<27-7=20>>20 kg 103 - 27 - 20 = <<103-27-20=56>>56 kg 56/2 = <<56/2=28>>28 kg The last two people each lost 28 kilograms of weight. #### 28"
        }
    ]
    return test_data

## Step 2: Implement the baseline method
def baseline_method(client, model_name, seed, question):
    # Zero-shot chain-of-thought
    prompt = f"Answer the following question: {question}\nThink step by step."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()

## Step 3: Implement the proposed method
def proposed_method(client, model_name, seed, question, print_all=False):
    intermediate_outputs = ""

    if print_all:
        print("question:\n", question)

    # Collaborative reasoning step 1: task decomposition
    prompt = f"Please break down the following task into smaller sub-tasks or steps: {question}"
    prompt_messages = [{"role": "user", "content": prompt}]
    decomposition, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += f"task decomposition:\n{decomposition}\n"
    if print_all:
        print("decomposition:\n", decomposition)

    # Collaborative reasoning step 2: sub-task information generation
    prompt = f"For each of the following sub-tasks, please generate relevant information or intermediate results:\n{decomposition}"
    prompt_messages = [{"role": "user", "content": prompt}]
    intermediate, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += f"sub-task results:\n{intermediate}\n"
    if print_all:
        print("intermediate:\n", intermediate)

    # Collaborative reasoning step 3: result combination
    prompt = f"Given the following intermediate results:\n{intermediate}, please combine them to generate the final answer for the task:\n{question}"
    prompt_messages = [{"role": "user", "content": prompt}]
    answer, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += f"result combination:\n{answer}\n"
    if print_all:
        print("initial answer:\n", answer)

    # Collaborative reasoning step 4: reflection and refinement
    prompt = (
        f"Given the task: {question}\nPlease reflect on the generated answer:\n{answer}.\n\n"
        "Are there any gaps or inconsistencies in the answer? If so, please identify and address them and give me an improved answer. "
        "If not, you don’t have to edit anything and can just return the original answer."
    )
    prompt_messages = [{"role": "user", "content": prompt}]
    final_answer, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += f"reflection and refinement:\n{final_answer}"
    if print_all:
        print("final answer:\n", final_answer)

    return final_answer.strip(), intermediate_outputs

## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    # Define evaluation criteria
    prompt = (
        f"Given the task: {question}\n"
        f"The baseline method produced the following output:\n{baseline_prediction}\n\n"
        f"The proposed new method produced the following output:\n{proposed_prediction}\n\n"
        "Now determine if the proposed method is better by checking if it has satisfied the following criteria:\n"
        "1. The proposed method’s output should produce all the intermediate components including: task decomposition, sub-task information generation, result combination, and reflection and refinement.\n"
        "2. The proposed method should provide a more detailed and comprehensive answer than the baseline method.\n"
        "Just tell me ’yes’ or ’no’ for whether the criteria are met, nothing else is needed."
    )
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)

    return response.strip().lower() == "yes"

## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    # Check correctness of the prediction
    prompt = (
        f"Given the following question and reference answer, determine if the prediction is correct. Just tell me ’yes’ or ’no’, nothing else is needed.\n\n"
        f"Question: {question}\n\nReference Answer: {gold_label}\n\nPrediction: {prediction}\n\n"
    )
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)

    return response.strip().lower() == "yes"

## Step 6: Define the function that runs the experiments to obtain model predictions and performance
def run_experiment(client, model_name, seed, testset):
    sample_size = len(testset)
    baseline_predictions = []
    proposed_predictions = []

    baseline_correctness = []
    proposed_correctness = []

    style_check = []

    for i in tqdm(range(sample_size)):
        question = testset[i]["input"].strip()
        gold_label = testset[i]["output"].strip()

        baseline_prediction = baseline_method(client, model_name, seed, question)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, question)

        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)

        baseline_correctness.append(output_evaluator(client, model_name, seed, question, gold_label, baseline_prediction))
        proposed_correctness.append(output_evaluator(client, model_name, seed, question, gold_label, proposed_prediction_final))

        style_check.append(style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction_intermediate))

    return baseline_correctness, proposed_correctness, style_check

## Step 7: Execute the experiments and compare performance
if __name__ == "__main__":
    testset = generate_testset()
    print(f"Simulated {len(testset)} test examples for evaluation.")

    model_name = "claude-3-opus-20240229"
    seed = 2024
    client = load_model(model_name)
    print(f"Using model: {model_name}")

    # Output correctness
    baseline_correctness, proposed_correctness, style_check = run_experiment(client, model_name, seed, testset)
    print(f"Baseline correctness: {sum(baseline_correctness) / len(baseline_correctness)}")
    print(f"Proposed correctness: {sum(proposed_correctness) / len(proposed_correctness)}")
    print(f"Style check pass rate: {sum(style_check) / len(style_check)}")
