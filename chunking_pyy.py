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
                 "type": node_type,
                 "node": node}
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
                        current_group[0]['node']
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
                current_group[0]['node']
            )

            merged_chunks.append(merged)

        self.chunks = merged_chunks
        return merged_chunks

    def split_internals(self):
        updated_chunks = []
        for chunk in self.chunks:
            if len(chunk['text']) <= self.max_size:
                updated_chunks.append(chunk)
            else:
                children = chunk['node'].body
                for child in children:
                    with open(chunk['file_path']) as f:
                        content = f.read()
                        chunk_content = ast.get_source_segment(content, child)
                        u_chunk = self.add_chunk(chunk_content, chunk['file_path'], chunk['first_char'], child)
                        u_chunk['parent'] =[chunk['node']]
                        if isinstance(chunk['node'], ast.FunctionDef):
                            u_chunk['hierarchy'] = [f"def {chunk['node'].name}"]
                        elif isinstance(chunk['node'], ast.ClassDef):
                            u_chunk['hierarchy'] = [f"class {chunk['node'].name}"]
                        if isinstance(child, ast.FunctionDef):
                            u_chunk['hierarchy'] += [f"def {child.name}"]
                        elif isinstance(child, ast.ClassDef):
                            u_chunk['hierarchy'] += [f"class {child.name}"]
                        u_chunk['identity'] = child
                        updated_chunks.append(u_chunk)
        self.chunks = updated_chunks
        return updated_chunks

    # def split_para(self):
    #     chunks = []

    #     for chunk in self.chunks:
    #         c_content = chunk['text']
    #         if len(c_content) <= self.max_size:
    #             chunks.append(chunk)
    #             continue
    #         paragraphs = c_content.split("\n\n")
    #         if not isinstance(paragraphs, list):
    #             continue
    #         find_position = 0
    #         for para in paragraphs:
    #             para = para.strip()
    #             if not para:
    #                 continue
    #             real_start = c_content.find(para, find_position)
    #             find_position = real_start + len(para)

    #             chunkk = self.add_chunk(para, chunk['file_path'], real_start, chunk['node'])
    #             if 'parent' in chunk:
    #                 chunkk['parent'] = [chunk['parent']]
    #                 chunkk['parent'].append(chunk['identity'])
    #             chunks.append(chunkk)
    #     self.chunks = chunks
    #     return chunks

    def split_lines(self):
        chunks = []

        for chunk in self.chunks:
            c_content = chunk['text']

            if len(c_content) <= self.max_size:
                chunks.append(chunk)
                continue

            lines = c_content.splitlines(keepends=True)

            current_lines = []
            current_size = 0
            sub_chunk_start = chunk['first_char']

            for line in lines:
                if not line.strip():
                    continue
                line_len = len(line)

                # Adding this line would exceed max_size
                if current_size + line_len > self.max_size and current_lines:

                    join_lines = ''.join(current_lines)

                    chunkk = {
                        "file_path": chunk['file_path'],
                        "first_char": sub_chunk_start,
                        "last_char": sub_chunk_start + len(join_lines),
                        "text": join_lines,
                        "type": chunk['type'],
                        "node": chunk['node']
                    }

                    # Preserve hierarchy information
                    if 'parent' in chunk:
                        chunkk['parent'] = chunk['parent'] + [chunk['identity']]
                        chunkk['hierarchy'] = chunk['hierarchy']


                    chunks.append(chunkk)

                    # Next chunk starts immediately after this one
                    sub_chunk_start += len(join_lines)

                    current_lines = [line]
                    current_size = line_len

                else:
                    current_lines.append(line)
                    current_size += line_len

            # Add the final group
            if current_lines:
                join_lines = ''.join(current_lines)

                chunkk = {
                    "file_path": chunk['file_path'],
                    "first_char": sub_chunk_start,
                    "last_char": sub_chunk_start + len(join_lines),
                    "text": join_lines,
                    "type": chunk['type'],
                    "node": chunk['node']
                }
                if 'parent' in chunk:
                    chunkk['parent'] = chunk['parent'] + [chunk['identity']]
                    chunkk['hierarchy'] = chunk['hierarchy']
                    # Inside split_lines, replace the hierarchy assignment logic with:

                chunks.append(chunkk)

        self.chunks = chunks
        return chunks

    def split_sentence(self):
        chunks = []
        checkpoint_chars = ".!?…;:, \t"

        for chunk in self.chunks:
            content = chunk["text"]

            if len(content) <= self.max_size:
                chunks.append(chunk)
                continue

            chars_hub = ""
            chunk_start = chunk["first_char"]
            line_length = 0
            checkpoint = None

            for char in content:
                chars_hub += char
                line_length += 1

                if char in checkpoint_chars:
                    checkpoint = line_length

                if line_length == self.max_size:

                    if checkpoint is not None:
                        piece = chars_hub[:checkpoint]

                        chunkk = self.add_chunk(
                            piece,
                            chunk['file_path'],
                            chunk_start,
                            chunk['node']
                        )

                        if 'parent' in chunk:
                            chunkk['parent'] = chunk['parent']

                        chunks.append(chunkk)

                        chars_hub = chars_hub[checkpoint:]
                        chunk_start += checkpoint
                        line_length -= checkpoint

                    else:
                        chunkk = self.add_chunk(
                            chars_hub,
                            chunk['file_path'],
                            chunk_start,
                            chunk['node']
                        )

                    if 'parent' in chunk:
                        chunkk['parent'] = chunk['parent']
                        chunkk['hierarchy'] = chunk['hierarchy']

                        chunks.append(chunkk)

                        chunk_start += line_length
                        chars_hub = ""
                        line_length = 0

                    checkpoint = None

            if chars_hub:
                chunkk = self.add_chunk(
                    chars_hub,
                    chunk['file_path'],
                    chunk_start,
                    chunk['node']
                )

                if 'parent' in chunk:
                    chunkk['parent'] = chunk['parent']
                    chunkk['hierarchy'] = chunk['hierarchy']

                chunks.append(chunkk)

        self.chunks = chunks
        return chunks


file = "t.py"
pyt = Python([file], 1)
chunks = pyt.split_toplvl()
chunks = pyt.merge_toplvl()
chunks = pyt.split_internals()
# chunks = pyt.split_para()
chunks = pyt.split_lines()
chunks = pyt.split_sentence()
for ch in chunks:
    if ch.get('hierarchy', 0):
        print(ch['hierarchy'])
    print(ch)
    print()