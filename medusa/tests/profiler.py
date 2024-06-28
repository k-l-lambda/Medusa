
import argparse
import torch
from fastchat.model.model_adapter import get_conversation_template
import numpy as np
import time

from medusa.model.medusa_model import MedusaModel
from medusa.model.utils import reset_medusa_mode
from medusa.model.kv_cache import initialize_past_key_values



def runMedusa (model, input_ids, temperature, max_steps):
	output = ''
	accepts = []

	t0 = time.time()

	with torch.inference_mode():
		gen = model.medusa_generate(
			input_ids,
			temperature=temperature,
			max_steps=max_steps,
		)

		for batch in gen:
			output = batch['text']
			t1 = batch["t1"]
			new_token = batch['new_token']
			accept_length = batch['accept_length']
			accepts.append(accept_length)

			mem=torch.cuda.memory_allocated(0)
			print(f'{accept_length}/{new_token},	mem:{mem:,}')

	t_end = time.time()

	return dict(
		output='> ' + output + '\n',
		accepts=accepts,
		ttft=t1 - t0,
		duration=t_end - t1,
		n_tokens=np.sum(accepts) + len(accepts),
		mem=mem,
	)


def runWO (model, input_ids, max_steps):
	output = ''

	t0 = time.time()

	with torch.inference_mode():
		input_len = input_ids.shape[1]

		(
			past_key_values,
			past_key_values_data,
			current_length_data,
		) = initialize_past_key_values(model.base_model)
		model.past_key_values = past_key_values
		model.past_key_values_data = past_key_values_data
		model.current_length_data = current_length_data

		reset_medusa_mode(model)

		n_tokens = 0

		logits = model(
			input_ids, past_key_values=model.past_key_values, output_orig=True, medusa_forward=False
		).logits

		t1 = time.time()

		while True:
			new_id = torch.argmax(logits[:, -1])[None, None]
			new_id_value = new_id.item()
			print(f'{new_id_value=},	{n_tokens}	mem:{torch.cuda.memory_allocated(0):,}')

			if new_id_value == model.tokenizer.eos_token_id or n_tokens >= max_steps:
				output = model.tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True, spaces_between_special_tokens=False, clean_up_tokenization_spaces=True)
				break

			input_ids = torch.cat([input_ids, new_id], dim=-1)

			n_tokens += 1
			logits = model(
				new_id,
				past_key_values=model.past_key_values,
				output_orig=True,
				medusa_forward=False,
				position_ids=None,
				output_attentions=False,
				use_cache=False,
			).logits

	t_end = time.time()

	return dict(
		output='> ' + output + '\n',
		ttft=t1 - t0,
		duration=t_end - t1,
		n_tokens=n_tokens,
		mem=torch.cuda.memory_allocated(0),
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

	queries = open(args.query, 'r').read().split('\n')
	queries = [q for q in queries if len(q) > 0]

	result_medusa = dict(
		output='',
		accepts=[],
		ttft=0,
		duration=0,
		n_tokens=0,
		mem=0,
	)

	result_wo = dict(
		output='',
		ttft=0,
		duration=0,
		n_tokens=0,
		mem=0,
	)

	for query in queries:
		print('query:', query)
		conv = get_conversation_template(args.model)

		conv.append_message(conv.roles[0], query)
		conv.append_message(conv.roles[1], None)
		prompt = conv.get_prompt()

		input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.base_model.device)

		o_medusa = runMedusa(model, input_ids, max_steps=args.max_steps, temperature=args.temperature)
		o_wo = runWO(model, input_ids, max_steps=o_medusa['n_tokens'])

		for k in result_medusa.keys():
			result_medusa[k] += o_medusa[k]
		for k in result_wo.keys():
			result_wo[k] += o_wo[k]

	tps_medusa = result_medusa['n_tokens'] / result_medusa['duration']
	tps_wo = result_wo['n_tokens'] / result_wo['duration']

	n_query = len(queries)

	print('------------------------')
	print(result_medusa['output'])
	print('------------------------')
	print(result_wo['output'])
	print('------------------------')
	print('\n\n')

	print('ttft:', result_medusa['ttft'], ':', result_wo['ttft'])
	print('tps:', tps_medusa / tps_wo, '=', tps_medusa, ':', tps_wo)
	print('mem:', f'{(result_medusa["mem"] - result_wo["mem"]) // n_query:,}	= ({result_medusa["mem"]:,} - {result_wo["mem"]:,}) / {n_query}')
	print('mean accept_length:', np.mean(result_medusa['accepts']))


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, required=True, help="Model name or path.")
	parser.add_argument("--query", type=str, default="./assets/eval_queries.txt", help="Prompt text input")
	parser.add_argument("--original-infer", "-ori", action="store_true", help="Use original inference mode.")
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
	parser.add_argument("--max-steps", type=int, default=512)
	parser.add_argument("--no-history", action="store_true")
	parser.add_argument(
		"--style",
		type=str,
		default="simple",
		choices=["simple", "rich", "programmatic"],
		help="Display style.",
	)
	parser.add_argument(
		"--multiline",
		action="store_true",
		help="Enable multiline input. Use ESC+Enter for newline.",
	)
	parser.add_argument(
		"--mouse",
		action="store_true",
		help="[Rich Style]: Enable mouse support for cursor positioning.",
	)
	parser.add_argument(
		"--debug",
		action="store_true",
		help="Print useful debug information (e.g., prompts)",
	)
	args = parser.parse_args()
	main(args)
