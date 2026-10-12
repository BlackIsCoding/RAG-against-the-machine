import bm25s
import Stemmer
import math

def tokenize_chunks(chunks, query):
    en_stem = Stemmer.Stemmer("english")
    tokenized = bm25s.tokenize(chunks, stemmer=en_stem)
    query = bm25s.tokenize([query], stemmer=en_stem)

    vocab = tokenized.vocab
    query_tokens = [
        vocab[token]
        for token in query.vocab
        if token in vocab
    ]
    encoded_chunks = tokenized.ids

    return query_tokens, encoded_chunks

def build_tf(query, chunks):

    chunks_infos = []
    for chunk in chunks:
        tf_dict = {}
        for token in query:
            tf = chunk.count(token)
            if not tf:
                continue
            tf_dict[token] = tf

        chunks_infos.append((chunk, tf_dict))
    return chunks_infos


def build_df(query, chunks):
    df_dict = {}

    for token in query:
        df = sum(token in chunk for chunk in chunks)
        df_dict[token] = df
    return df_dict
        
def build_idf(query, df_dict, chunks):
    idf_dict = {}
    n = len(chunks)
    for token in query:
        df = df_dict[token]
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        idf_dict[token] = idf
    return idf_dict

def average_lenght(chunks):

    lenght = 0
    for chunk in chunks:
        lenght += len(chunk)
    avgdl = lenght / len(chunks) if len(chunks) else 0
    return len(chunks), avgdl


def bm25_score(tf_dict, idf_dict, dl, avgdl, k1=1.5, b=0.75):

    bm25_score = 0
    for token , tf in tf_dict.items():
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl /avgdl)
        score = idf_dict[token] * numerator / denominator
        bm25_score += score
    return bm25_score


def score_chunks(chunks_tf, chunk_tokens, idf_dict, avgdl, k1=1.5, b=0.75):
    scores = {}

    for i, (chunk, tf_dict) in enumerate(chunks_tf):
        dl = len(chunk_tokens[i])
        score = bm25_score(tf_dict, idf_dict, dl, avgdl, k1, b)
        scores[i] = score
    return scores

def return_topk(scores, top_k):
    return sorted(scores, key=scores.get, reverse=True)[:top_k]

chunks = ["john cena is one of the greatest",
          "Berserk is a manga by Kentaru miura",
          "Cr7 stands for Cristiano Ronaldo awesome",
          "Messi is argentian awesome"]
query = "berserk manga is awesome"


query_tokens, en_chunks = tokenize_chunks(chunks, query)


chunks_tf = build_tf(query_tokens, en_chunks)
df_dict = build_df(query_tokens, en_chunks)
idf_dict = build_idf(query_tokens, df_dict, en_chunks)
_, avgdl = average_lenght(en_chunks)

scores = score_chunks(chunks_tf, en_chunks, idf_dict, avgdl)
top_k = return_topk(scores, 3)

print(f"Query is : {query}\n")
print("retrieved sources:")
for i, k in enumerate(top_k):
    print(f"number {i + 1} is {chunks[k]}")