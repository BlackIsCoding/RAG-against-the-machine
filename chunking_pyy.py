import ast

class Python():
    def __init__(self, python_files, max_size):
        self.python_files = python_files
        self.max_size = max_size
        self.chunks = []

    def add_chunk(self, chunk_content, file, find_index, node):
        with open(file) as f:
            content = f.read()
        first_char = content.find(chunk_content, find_index)
        node_type = type(node).__name__ if node else None
        chunk = {"file_path": file, "first_char": first_char,
                 "last_char": first_char + len(chunk_content),
                 "text": chunk_content,
                 "type": node_type}
        return chunk

    def split_toplvl(self):
        chunks = []
        
        for file in self.python_files:
            find_index = 0
            
            with open(file) as f:
                content = f.read()
                
                if len(content) <= self.max_size:
                    chunking = self.add_chunk(content, file, find_index, None)
                    chunks.append(chunking)
                else:
                    tree = ast.parse(content)
                    for node in tree.body:
                        chunk = self.add_chunk(ast.get_source_segment(content, node), file, find_index, node)
                        find_index = chunk['last_char']
                        chunks.append(chunk)
        self.chunks = chunks
        return chunks

    def merge_toplvl(self):
        current_chunks = self.chunks
        merged_chunks = []
        current_group = []
        index = 0
        group_size = 0

        while index < len(current_chunks):
            chunk = current_chunks[index]
            current_size = chunk['last_char'] - chunk['first_char']

            if group_size + current_size <= self.max_size:
                current_group.append(chunk)
                group_size += current_size
                index += 1
            else:
                # Merge the current group

                if current_group:
                    # merge current_group
                    with open(current_group[0]['file_path']) as f:
                        content = f.read()

                    first = current_group[0]['first_char']
                    last = current_group[-1]['last_char']
                    text = content[first:last]

                    merged = self.add_chunk(
                        text,
                        current_group[0]['file_path'],
                        first,
                        None
                    )

                    merged_chunks.append(merged)

                # DON'T throw away chunk
                current_group = [chunk]
                group_size = current_size
                index += 1
                # Don't forget the last group
        if current_group:
            with open(current_group[0]['file_path']) as f:
                content = f.read()

            first = current_group[0]['first_char']
            last = current_group[-1]['last_char']
            text = content[first:last]

            merged = self.add_chunk(
                text,
                current_group[0]['file_path'],
                first,
                None
            )

            merged_chunks.append(merged)

        self.chunks = merged_chunks
        return merged_chunks
















file = "t.py"
pyt = Python([file], 100)
chunks = pyt.split_toplvl()
chunks = pyt.merge_toplvl()

for ch in chunks:
    print(ch)
    print()