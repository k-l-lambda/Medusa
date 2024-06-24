
import argparse
import torch
from fastchat.model.model_adapter import get_conversation_template
import numpy as np

from medusa.model.medusa_model import MedusaModel
from medusa.model.kv_cache import initialize_past_key_values



def main (args):
	model = MedusaModel.from_pretrained(
		args.model,
		torch_dtype=torch.float16,
		low_cpu_mem_usage=True,
		device_map='auto',
	)

	tokenizer = model.get_tokenizer()
	conv = get_conversation_template(args.model)

	inp = "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients"
	conv.append_message(conv.roles[0], inp)
	conv.append_message(conv.roles[1], None)
	prompt = conv.get_prompt()

	input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.base_model.device)

	output = ''
	accepts = []

	if args.original_infer:
		input_len = input_ids.shape[1]

		while True:
			(
				past_key_values,
				past_key_values_data,
				current_length_data,
			) = initialize_past_key_values(model.base_model)

			accepts.append(0)

			medusa_logits, outputs, logits = model(
				input_ids, past_key_values=past_key_values, output_orig=True, medusa_forward=True
			)
			new_id = torch.argmax(logits[:, -1])
			new_id_value = new_id.item()
			print(f'{new_id_value=},	{len(accepts)}	mem:{torch.cuda.memory_allocated(0):,}')

			if new_id_value == model.tokenizer.eos_token_id:
				output = model.tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True, spaces_between_special_tokens=False, clean_up_tokenization_spaces=True)
				break

			input_ids = torch.cat([input_ids, new_id[None, None]], dim=-1)
	else:
		gen = model.medusa_generate(
			input_ids,
			temperature=args.temperature,
			max_steps=args.max_steps,
		)

		for batch in gen:
			output = batch['text']
			new_token = batch['new_token']
			accept_length = batch['accept_length']
			accepts.append(accept_length)

			print(f'{accept_length}/{new_token},	mem:{torch.cuda.memory_allocated(0):,}')

	print(f'{output=}')
	print(f'{np.mean(accepts)=}')


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, required=True, help="Model name or path.")
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
	parser.add_argument("--temperature", type=float, default=0.7)
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
