import typer
import json
from transformers import Conversation
from typing_extensions import Annotated
import httpx
import tqdm
import asyncio
import os
import pandas as pd

app = typer.Typer()


client = httpx.AsyncClient(timeout=None)

save_batch_size = 100


async def run(conv: Conversation, url: str):
    payload = {"model":"/models/Meta-Llama-3-8B-Instruct/", "messages": conv.messages, "temperature": 0}
    #print('payload:', payload)
    #print('request:', conv.messages)
    response = await client.post(url, json=payload)
    content = response.json()
    message = content["choices"][0]["message"]
    message.pop("tool_calls", None)
    #print('reply:', message)
    conv.add_message(message)




def fix_source(source):
    if source and (source[0]["from"] in ["gpt", "assistant"]):
        # Skip if GPT is first to talk
        source = source[1:]
    new_source = []
    for item in source:
        role = "assistant" if item["from"] == "gpt" else "user"
        content = item["value"]
        new_source.append({"role": role, "content": content})
    return new_source


async def recreate_conversation(conversation, sem, url):
    async with sem:
        conv = Conversation()
        try:
            for message in conversation[::2]:
                assert message["role"] == "user"
                conv.add_message(message)
                await run(conv, url)
        except Exception as e:
            print(e)
            exit(-1)
        return conv.messages


def load_conversation_json(filename):
    with open(filename, "r") as f:
        input_data = json.loads(f.read())
    return [fix_source(source["conversations"]) for source in input_data]


def load_conversation_parquet (filename):
    df = pd.read_parquet(filename)
    return df["conversation"]


@app.command()
def main(
    *,
    input_filename: Annotated[str, typer.Option("--input-filename")],
    output_filename: Annotated[str, typer.Option("--output-filename")],
    url: Annotated[str, typer.Option("--url")] = "http://localhost:8080/v1/chat/completions",
    concurrency: Annotated[int, typer.Option("--concurrency")] = 64
):
    sem = asyncio.Semaphore(concurrency)
    async def _main():
        if input_filename.endswith(".parquet"):
            conversations = load_conversation_parquet(input_filename)
        elif input_filename.endswith(".json"):
            conversations = load_conversation_json(input_filename)

        backup_output_filename = output_filename + ".bak"

        recreated_conversations = []
        if os.path.exists(output_filename):
            recreated_conversations = json.load(open(output_filename, "r"))
            conversations = conversations[len(recreated_conversations):]
            print(f'{len(recreated_conversations)} conversations loaded.')

        for i in tqdm.tqdm(range(0, len(conversations), save_batch_size), position=1):
            batch_conversations = conversations[i:i+save_batch_size]

            futures = []
            for conversation in batch_conversations:
                future = recreate_conversation(conversation, sem, url)
                futures.append(future)

            recreated_conversations_batch = await tqdm.asyncio.tqdm.gather(*futures, leave=False, position=2)
            recreated_conversations += recreated_conversations_batch

            if os.path.exists(output_filename):
                if os.path.exists(backup_output_filename):
                    os.remove(backup_output_filename)
                os.rename(output_filename, backup_output_filename)

            with open(output_filename, "w") as f:
                json.dump(recreated_conversations, f, indent=4)
    asyncio.run(_main())


if __name__ == "__main__":
    app()
