import ast

class Python():
    def __init__(self, python_files, max_size):
        self.python_files = python_files
        self.max_size = max_size
        self.chunks = []
        self.file_cache = {}

    def _get_file_content(self, file):
        """Helper to load a file into memory once and reuse it."""
        if file not in self.file_cache:
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    self.file_cache[file] = f.read()
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                self.file_cache[file] = ""
        return self.file_cache[file]

    @staticmethod
    def _compute_line_offsets(source: str) -> list[int]:
        """Compute the starting character index for each line."""
        offsets = [0]
        for line in source.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        return offsets

    def _pos_to_char(self, source: str, line_offsets: list[int], lineno: int, col_offset: int) -> int:
        """Convert 1-based (lineno, col_offset) to an absolute character index."""
        if lineno <= 0:
            return 0
        line_idx = lineno - 1
        if line_idx >= len(line_offsets):
            return len(source)
        return min(line_offsets[line_idx] + col_offset, len(source))

    def _get_node_span(self, source: str, line_offsets: list[int], node: ast.AST) -> tuple[int, int]:
        """Extract absolute start and end character indexes for an AST node using numbers."""
        start_line = getattr(node, "lineno", 1)
        start_col = getattr(node, "col_offset", 0)
        end_line = getattr(node, "end_lineno", start_line)
        end_col = getattr(node, "end_col_offset", start_col)

        start_idx = self._pos_to_char(source, line_offsets, start_line, start_col)
        end_idx = self._pos_to_char(source, line_offsets, end_line, end_col)
        return start_idx, end_idx

    def add_chunk_numeric(self, file, start_idx, end_idx, node):
            content = self._get_file_content(file)
            chunk_content = content[start_idx:end_idx]
            node_type = type(node).__name__ if node else None
            
            # If your tests expect inclusive bounds, use end_idx - 1 (handle empty chunks safely)
            inclusive_last = max(start_idx, end_idx - 1) if chunk_content else start_idx

            chunk = {
                "file_path": file,
                "first_char": start_idx,
                "last_char": inclusive_last, 
                "text": chunk_content,
                "type": node_type,
                "node": node
            }
            return chunk

    def split_toplvl(self):
        chunks = []
        
        for file in self.python_files:
            
            content = self._get_file_content(file)
            if not content.strip():
                continue
                
            if len(content) <= self.max_size:
                chunks.append(self.add_chunk_numeric(file, 0, len(content), None))
            else:
                line_offsets = self._compute_line_offsets(content)
                try:
                    tree = ast.parse(content, filename=file)
                    for node in tree.body:
                        start_idx, end_idx = self._get_node_span(content, line_offsets, node)
                        if start_idx < end_idx:
                            chunks.append(self.add_chunk_numeric(file, start_idx, end_idx, node))
                except SyntaxError:
                    chunks.append(self.add_chunk_numeric(file, 0, len(content), None))
                        
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
            same_file = not current_group or current_group[0]['file_path'] == chunk['file_path']
            
            if same_file and (group_size + current_size <= self.max_size):
                current_group.append(chunk)
                group_size += current_size
                index += 1
            else:
                if current_group:
                    first = current_group[0]['first_char']
                    last = current_group[-1]['last_char']
                    merged = self.add_chunk_numeric(
                        current_group[0]['file_path'],
                        first,
                        last,
                        current_group[0]['node']
                    )
                    merged_chunks.append(merged)

                current_group = [chunk]
                group_size = current_size
                index += 1

        if current_group:
            first = current_group[0]['first_char']
            last = current_group[-1]['last_char']
            merged = self.add_chunk_numeric(
                current_group[0]['file_path'],
                first,
                last,
                current_group[0]['node']
            )
            merged_chunks.append(merged)

        self.chunks = merged_chunks
        return merged_chunks

    def split_internals(self):
        updated_chunks = []
        
        for chunk in self.chunks:
            if len(chunk['text']) <= self.max_size or not hasattr(
                chunk['node'], 'body') or chunk['node'] is None:
                updated_chunks.append(chunk)
            else:
                children = chunk['node'].body
                content = self._get_file_content(chunk['file_path'])
                line_offsets = self._compute_line_offsets(content)
                
                for child in children:
                    start_idx, end_idx = self._get_node_span(content, line_offsets, child)
                    if start_idx >= end_idx:
                        continue
                        
                    u_chunk = self.add_chunk_numeric(chunk['file_path'], start_idx, end_idx, child)
                    u_chunk['parent'] = [chunk['node']]
                    u_chunk['hierarchy'] = []
                    
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

                if current_size + line_len > self.max_size and current_lines:
                    join_lines = ''.join(current_lines)
                    end_idx = sub_chunk_start + len(join_lines)
                    
                    chunkk = {
                        "file_path": chunk['file_path'],
                        "first_char": sub_chunk_start,
                        "last_char": end_idx,
                        "text": join_lines,
                        "type": chunk['type'],
                        "node": chunk['node']
                    }

                    if 'parent' in chunk:
                        chunkk['parent'] = chunk['parent'] + [chunk['identity']]
                        chunkk['hierarchy'] = chunk['hierarchy']

                    chunks.append(chunkk)
                    sub_chunk_start = end_idx
                    current_lines = [line]
                    current_size = line_len
                else:
                    current_lines.append(line)
                    current_size += line_len

            if current_lines:
                join_lines = ''.join(current_lines)
                end_idx = sub_chunk_start + len(join_lines)
                
                chunkk = {
                    "file_path": chunk['file_path'],
                    "first_char": sub_chunk_start,
                    "last_char": end_idx,
                    "text": join_lines,
                    "type": chunk['type'],
                    "node": chunk['node']
                }
                if 'parent' in chunk:
                    chunkk['parent'] = chunk['parent'] + [chunk['identity']]
                    chunkk['hierarchy'] = chunk['hierarchy']

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
                        end_idx = chunk_start + len(piece)
                        
                        chunkk = self.add_chunk_numeric(
                            chunk['file_path'],
                            chunk_start,
                            end_idx,
                            chunk['node']
                        )
                        if 'parent' in chunk:
                            chunkk['parent'] = chunk['parent']

                        chunks.append(chunkk)
                        chars_hub = chars_hub[checkpoint:]
                        chunk_start = end_idx
                        line_length -= checkpoint
                    else:
                        end_idx = chunk_start + len(chars_hub)
                        chunkk = self.add_chunk_numeric(
                            chunk['file_path'],
                            chunk_start,
                            end_idx,
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
                end_idx = chunk_start + len(chars_hub)
                chunkk = self.add_chunk_numeric(
                    chunk['file_path'],
                    chunk_start,
                    end_idx,
                    chunk['node']
                )
                if 'parent' in chunk:
                    chunkk['parent'] = chunk['parent']
                    chunkk['hierarchy'] = chunk['hierarchy']

                chunks.append(chunkk)

        self.chunks = chunks
        return chunks

    def chunks_adapt(self):
        chunks = self.chunks
        new = []

        for chunk in chunks:
            keys = list(chunk.keys())
            new_chunk = {}
            for key in keys:
                if key == 'text':
                    new_chunk['content'] = chunk[key]
                elif key == 'first_char':
                    new_chunk['start_char'] = chunk[key]
                elif key == 'last_char':
                    new_chunk['end_char'] = chunk[key]
                elif key == 'hierarchy':
                    new_chunk['metadata'] = chunk['hierarchy']
                else:
                    new_chunk[key] = chunk[key]
            if 'metadata' not in new_chunk:
                new_chunk['metadata'] = []
            new.append(new_chunk)
        return new