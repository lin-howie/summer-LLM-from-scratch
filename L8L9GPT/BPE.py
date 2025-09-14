import tiktoken

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = tokenizer.encode(text)

print(token_ids[:5])
