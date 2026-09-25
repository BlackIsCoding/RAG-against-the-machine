from src.markdown import Header

splitter = Header(["t.md"], 100)
chunks = splitter.split()
chunks = splitter.split_para()
chunks = splitter.split_lines()
chunks = splitter.split_sentence()
print("number of chunks", len(chunks), end='\n\n')

for chunk in chunks:
    print("headers ->", chunk['metadata'])
    print("start ->", chunk['start_char'])
    print("end ->", chunk['end_char'])
    print("file ->", chunk['file_path'])
    print("content ->", chunk['content'])
    print()