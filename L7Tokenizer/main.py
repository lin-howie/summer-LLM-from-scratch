from SimpleTokenizerV1 import SimpleTokenizerV1
from SimpleTokenizerV2 import SimpleTokenizerV2
import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# Here is the preprocessing of the vocabulary
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_words = sorted(set(preprocessed))
vocab = {token: integer for integer, token in enumerate(all_words)}


# Here is the vocabulary with the |unk| token and |endoftext| token
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(['<|unk|>', '<|endoftext|>'])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

print([item for i, item in enumerate(list(vocab.items())[-5:])])

# New text
new_tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello do you like tea?"
text2 = "Of course, I do! Let's see the other side of the world."
input_text = " <|endoftext|>  ".join((text1, text2))

p = new_tokenizer.encode(input_text)
print(p)
p1 = new_tokenizer.decoder(p)
print(p1)

