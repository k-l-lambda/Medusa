
import argparse
import torch
from fastchat.model.model_adapter import get_conversation_template
from fastchat.llm_judge.common import load_questions
import numpy as np
import time
import os
import json
from tqdm import tqdm

from medusa.model.medusa_model import MedusaModel
from medusa.model.utils import reset_medusa_mode
from medusa.model.kv_cache import initialize_past_key_values



def reorg_answer_file (answer_file):
	"""Sort by question id and de-duplication"""
	answers = {}
	with open(answer_file, "r") as fin:
		for l in fin:
			qid = json.loads(l)["question_id"]
			answers[qid] = l

	qids = sorted(list(answers.keys()))
	with open(answer_file, "w") as fout:
		for qid in qids:
			fout.write(answers[qid])


def runMedusa (model, input_ids, temperature, max_tokens):
	eot = model.tokenizer.convert_tokens_to_ids("<|eot_id|>")

	torch.cuda.synchronize()
	t0 = time.time()

	with torch.inference_mode():
		gen = model.medusa_generate(
			input_ids,
			temperature=temperature,
			max_steps=max_tokens,
		)

		output_ids = None
		n_step = 0
		for batch in gen:
			output_ids = batch["output_ids"]

			n_step += 1

			if eot in output_ids.tolist():
				break
			if output_ids.shape[0] >= max_tokens:
				break

		output = model.tokenizer.decode(
			output_ids,
			skip_special_tokens=True,
			spaces_between_special_tokens=False,
			clean_up_tokenization_spaces=True,
		)

	torch.cuda.synchronize()
	t_end = time.time()

	return dict(
		output=output,
		duration=t_end - t0,
		n_tokens=output_ids.shape[0],
		n_step=n_step,
	)


def main (args):
	model = MedusaModel.from_pretrained(
		args.model,
		torch_dtype=torch.float16,
		low_cpu_mem_usage=True,
		device_map='auto',
	)
	model.eval()

	tokenizer = model.get_tokenizer()
	#print(f'{tokenizer.eos_token_id=}')

	questions = load_questions(args.question, args.question_begin, args.question_end)

	# warmup
	g = model.medusa_generate(
		torch.tensor([[128000]], dtype=torch.long, device=model.base_model.device),
		temperature=0,
		max_steps=1,
	)
	next(g)
	print("Warmup done.")

	for question in tqdm(questions):
		#print('question:', question)

		torch.manual_seed(0)
		turns = []
		idxs = []
		new_tokens = []
		wall_time = []
		for qs in question['turns']:
			#conv = get_conversation_template(args.model)
			#conv.append_message(conv.roles[0], qs)
			#conv.append_message(conv.roles[1], None)
			#prompt = conv.get_prompt()
			#input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.base_model.device)

			messages = [{
				"role": "user",
				"content": qs,
			}]
			prompt = tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True,
			)
			#print('prompt:', prompt)
			input_ids = torch.as_tensor(tokenizer([prompt], add_special_tokens=False).input_ids).to(model.base_model.device)

			o_medusa = runMedusa(model, input_ids, temperature=args.temperature, max_tokens=args.max_new_tokens)

			turns.append(o_medusa['output'])
			idxs.append(o_medusa['n_step'])
			new_tokens.append(o_medusa['n_tokens'])
			wall_time.append(o_medusa['duration'])

		choices = [{"index": 0, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time}]
		os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)
		with open(os.path.expanduser(args.answer_file), "a") as fout:
			ans_json = {
				"question_id": question["question_id"],
				#"answer_id": shortuuid.uuid(),
				"model_id": args.model,
				"choices": choices,
				"tstamp": time.time(),
			}
			fout.write(json.dumps(ans_json) + "\n")

	reorg_answer_file(args.answer_file)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, required=True, help="Model name or path.")
	parser.add_argument("--question", type=str, help="Question jsonl file")
	parser.add_argument("--answer_file", type=str, help="Path to the answer jsonl file")
	parser.add_argument(
		"--question-begin", type=int, help="A debug option. The begin index of questions.",
	)
	parser.add_argument(
		"--question-end", type=int, help="A debug option. The end index of questions."
	)
	#parser.add_argument("--original-infer", "-ori", action="store_true", help="Use original inference mode.")
	parser.add_argument(
		"--load-in-8bit", action="store_true", help="Use 8-bit quantization"
	)
	parser.add_argument(
		"--load-in-4bit", action="store_true", help="Use 4-bit quantization"
	)
	parser.add_argument(
		"--conv-template", type=str, default=None, help="Conversation prompt template."
	)
	parser.add_argument(
		"--conv-system-msg", type=str, default=None, help="Conversation system message."
	)
	parser.add_argument("--temperature", type=float, default=0)
	#parser.add_argument("--max-steps", type=int, default=512)
	parser.add_argument(
		"--max-new-tokens", type=int, default=512, help="The maximum number of new generated tokens.",
	)
	args = parser.parse_args()
	main(args)
