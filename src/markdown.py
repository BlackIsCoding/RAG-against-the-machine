class Header():
    def __init__(self, docs_files, max_size):
        self.docs_files = docs_files
        self.max_size = max_size
        self.chunks = []
    
    def add_chunk(self, content, headers, file_path, section_start):
        if content: # because for the first header i try to add an empty chunk 
            raw_text = ''.join(content)
            s_text = raw_text.lstrip()
            if s_text:
                lstrip_len = len(raw_text) - len(s_text)
                start_char = lstrip_len + section_start
                return({
                            "content": s_text,
                            "metadata": 
                            [f"Header {h} -> {headers[h]}" for h in headers],
                            "start_char": start_char,
                            "end_char": start_char + len(s_text) - 1,
                            "file_path": file_path
                        })
            return {}


    def chunk_header(self, chunks, content, hashs, title, section_start, headers, doc, offset):
        chunk = self.add_chunk(content, headers, doc, section_start)
        content.clear()
        if chunk: # because a full chunk can be \n\n\n\
            self.chunks.append(chunk)

        header_level = len(hashs)
        for level in list(headers):
            if level > header_level:
                del headers[level]
        s_title = title.strip()
        headers[header_level] = s_title
        return offset


    def split(self):
        docs_files = self.docs_files
        self.chunks: list[dict] = []

        for doc in docs_files:
            if doc.endswith(".md"):
                headers: dict[str, int] | None = {}
                content: list[str] = []
                offset = 0
                section_start = 0

                with open(doc) as f:
                    for line in f:
                        start = offset
                        offset += len(line) # newline included
                        s_line = line.strip()

                        if s_line.startswith("#"): # checking if this line is a header
                            parts = s_line.split(' ', 1)
                            if len(parts) == 2:
                                hashs, title = parts
                            is_header = (len(parts) == 2 and 1 <= len(hashs) <= 6 and set(hashs)== {'#'})

                            if is_header: # detecting a header
                                hashs, title = parts
                                section_start = self.chunk_header(
                                    self.chunks, content, hashs, title, section_start, headers, doc, offset)
                                continue

                        content.append(line) # append the line as long as it is a new header
                    
                    chunk = self.add_chunk(content, headers, doc, section_start)
                    content.clear()
                    if chunk:
                        self.chunks.append(chunk) # this is for the last chunk since i will hit EOF instead of a header
        return self.chunks


    def split_para(self):
        chunks = []

        for chunk in self.chunks:
            c_content = chunk['content']
            chunk_start = chunk['start_char']
            if len(c_content) <= self.max_size:
                chunks.append(chunk)
                continue
            paragraphs = c_content.split("\n\n")
            if not isinstance(paragraphs, list):
                continue
            find_position = 0
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                real_start = c_content.find(para, find_position)
                real_end = real_start + len(para)
                find_position = real_end

                chunkk = {
                    "content": para,
                    "metadata": chunk["metadata"].copy(),
                    "start_char": real_start + chunk_start,
                    "end_char": real_end + chunk_start - 1,
                    "file_path": chunk['file_path']
                }
                chunks.append(chunkk)
        self.chunks = chunks
        return chunks


    def split_lines(self):
        chunks = []

        for chunk in self.chunks:
            c_content = chunk['content']
            chunk_start = chunk['start_char']
            if len(c_content) <= self.max_size:
                chunks.append(chunk)
                continue
            lines = c_content.splitlines(keepends=True)
            chunk_lenght = 0
            current_lines = []
            sub_chunk_start = chunk['start_char']

            for line in lines:
                line_len = len(line)

                if chunk_lenght + line_len > self.max_size and current_lines: # if we hit the size_max and there are lines stored dump them as a chunk
                    # dump the chunk and initialize the next
                    join_lines = ''.join(current_lines)
                    chunkk = {"content": join_lines,
                              "metadata": chunk['metadata'],
                              "start_char": sub_chunk_start,
                              "end_char": sub_chunk_start + len(join_lines) - 1,
                              "file_path": chunk['file_path']}
                    chunks.append(chunkk)

                    sub_chunk_start += len(join_lines)
                    current_lines = [line]
                    chunk_lenght = len(line)

                else: # just append the line and add it's lenght
                    current_lines.append(line)
                    chunk_lenght += line_len
            
            if current_lines: # append the last chunk since it might skip the stop condition above
                join_lines = ''.join(current_lines)
                chunkk = {"content": join_lines,
                          "metadata": chunk['metadata'],
                          "start_char": sub_chunk_start,
                          "end_char": sub_chunk_start + len(join_lines) - 1,
                          "file_path": chunk['file_path']}
                chunks.append(chunkk)
        self.chunks = chunks
        return chunks

    def split_sentence(self):
        chunks = []
        checkpoint_chars = ".!?…;:, \t"

        for chunk in self.chunks:
            content = chunk["content"]

            if len(content) <= self.max_size:
                chunks.append(chunk)
                continue

            chars_hub = ""
            line_start = chunk["start_char"]
            line_length = 0
            checkpoint = None

            for char in content:
                chars_hub += char
                line_length += 1

                # Remember the latest checkpoint
                if char in checkpoint_chars:
                    checkpoint = line_length

                # We reached max_size
                if line_length == self.max_size:

                    # We found a checkpoint before reaching max_size
                    if checkpoint is not None:
                        piece = chars_hub[:checkpoint]

                        chunks.append({
                            "content": piece,
                            "metadata": chunk["metadata"].copy(),
                            "start_char": line_start,
                            "end_char": line_start + checkpoint - 1,
                            "file_path": chunk["file_path"]
                        })

                        # Keep the characters after the checkpoint
                        chars_hub = chars_hub[checkpoint:]
                        line_start += checkpoint
                        line_length -= checkpoint

                    # No checkpoint -> hard cut at max_size
                    else:
                        chunks.append({
                            "content": chars_hub,
                            "metadata": chunk["metadata"].copy(),
                            "start_char": line_start,
                            "end_char": line_start + line_length - 1,
                            "file_path": chunk["file_path"]
                        })

                        chars_hub = ""
                        line_start += line_length
                        line_length = 0

                    checkpoint = None

            # Add remaining characters
            if chars_hub:
                chunks.append({
                    "content": chars_hub,
                    "metadata": chunk["metadata"].copy(),
                    "start_char": line_start,
                    "end_char": line_start + line_length - 1,
                    "file_path": chunk["file_path"]
                })

        self.chunks = chunks
        return chunks