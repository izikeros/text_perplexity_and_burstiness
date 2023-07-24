import tiktoken
import math

# Exemplary text
text = "Another vital aspect to consider in the realm of text generation is perplexity. Essentially, perplexity refers to the amount of information contained within a text. Natural language possesses a remarkable redundancy that enables effective communication, even in noisy environments such as a crowded bar or an intimate dinner. This redundancy allows for the overall message to be comprehended, even if certain parts of the text are missing or obscured."

# Tokenize the text using tiktoken
tokens = tiktoken.tokenize(text)

# Calculate the total number of tokens
total_tokens = len(tokens)

# Create a dictionary to count the frequency of each token
token_freq = {}
for token in tokens:
    if token in token_freq:
        token_freq[token] += 1
    else:
        token_freq[token] = 1

# Calculate the probability of each token
token_probabilities = {token: freq / total_tokens for token, freq in token_freq.items()}

# Calculate the text probability by multiplying the probabilities of each token
text_probability = 1.0
for token in tokens:
    text_probability *= token_probabilities[token]

# Calculate the perplexity using the formula
perplexity = math.pow(text_probability, -1/total_tokens)

print(f"Perplexity of the text: {perplexity:.2f}")
