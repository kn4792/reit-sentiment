import pandas as pd

bigrams = pd.read_csv('data/processed/bigrams.csv')

ai_keywords = ['ai', 'generati' 'artificial', 'intelligence', 'technology',
                       'digital', 'software', 'system', 'tool', 'platform',
                       'chatgpt', 'gpt', 'machine', 'learni', 'algo', 'bing',
                       'bard', 'robot', 'robotic', 'openai', 'llm', 'model', 'gemini',
                       'copilot', 'neural', 'network', 'automat', 'predict', 'data']

ai_bigrams = bigrams[
    bigrams['word1'].str.contains('|'.join(ai_keywords), case=False, na=False) |
    bigrams['word2'].str.contains('|'.join(ai_keywords), case=False, na=False)
]

ai_bigrams.to_csv('data/processed/bigrams_ai_technology.csv', index=False)