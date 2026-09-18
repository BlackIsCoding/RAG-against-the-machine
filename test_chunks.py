import re

from pydantic import BaseModel


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(BaseModel):
    source: MinimalSource
    text: str
    heading_path: list[str]
    size: int


def split_markdown(
    content: str,
    file_path: str,
) -> list[Chunk]:
    """Split Markdown into structure-aware chunks."""
    chunks: list[Chunk] = []

    # (heading level, heading text)
    heading_stack: list[tuple[int, str]] = []

    current_content: list[str] = []
    current_start = 0
    current_position = 0

    for line in content.splitlines(keepends=True):
        match = re.match(r"^(#{1,3})\s+(.+?)\s*$", line)

        if match:
            # Save the content before this heading.
            if current_content:
                text = "".join(current_content)
                clean_text = text.strip()

                if clean_text:
                    leading = len(text) - len(text.lstrip())
                    trailing = len(text.rstrip())

                    start = current_start + leading
                    end = current_start + trailing

                    chunks.append(
                        Chunk(
                            source=MinimalSource(
                                file_path=file_path,
                                first_character_index=start,
                                last_character_index=end - 1,
                            ),
                            text=clean_text,
                            heading_path=[
                                heading for _, heading in heading_stack
                            ],
                            size=len(clean_text),
                        )
                    )

                current_content = []

            # Update heading hierarchy.
            level = len(match.group(1))
            heading = match.group(2)

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            heading_stack.append((level, heading))

            current_position += len(line)
            current_start = current_position

        else:
            if not current_content:
                current_start = current_position

            current_content.append(line)
            current_position += len(line)

    # Save the final section.
    if current_content:
        text = "".join(current_content)
        clean_text = text.strip()

        if clean_text:
            leading = len(text) - len(text.lstrip())
            trailing = len(text.rstrip())

            start = current_start + leading
            end = current_start + trailing

            chunks.append(
                Chunk(
                    source=MinimalSource(
                        file_path=file_path,
                        first_character_index=start,
                        last_character_index=end - 1,
                    ),
                    text=clean_text,
                    heading_path=[
                        heading for _, heading in heading_stack
                    ],
                    size=len(clean_text),
                )
            )

    return chunks


def split_paragraphs(chunk: Chunk) -> list[Chunk]:
    """Split a chunk into paragraphs while preserving source offsets."""
    result: list[Chunk] = []

    # CHANGE 1: Remove the faulty lookbehind (?<!\n\n) 
    # and match continuous lines of text as a paragraph block
    for match in re.finditer(
        r"(?s)([^\n]+(?:\n[^\n]+)*)",
        chunk.text,
    ):
        paragraph = match.group(1).strip()

        if not paragraph:
            continue

        relative_start = match.start(1)

        leading = len(match.group(1)) - len(match.group(1).lstrip())
        trailing = len(match.group(1).rstrip())

        relative_start += leading
        relative_end = match.start(1) + trailing

        absolute_start = (
            chunk.source.first_character_index
            + relative_start
        )
        absolute_end = (
            chunk.source.first_character_index
            + relative_end
        )

        result.append(
            Chunk(
                source=MinimalSource(
                    file_path=chunk.source.file_path,
                    first_character_index=absolute_start,
                    last_character_index=absolute_end - 1,
                ),
                text=paragraph,
                heading_path=chunk.heading_path.copy(),
                size=len(paragraph),
            )
        )

    return result


def split_lines(chunk: Chunk) -> list[Chunk]:
    """Split a paragraph chunk into individual lines while preserving source offsets."""
    result: list[Chunk] = []
    current_pos = chunk.source.first_character_index

    for line in chunk.text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            current_pos += len(line)
            continue

        leading = len(line) - len(line.lstrip())
        start = current_pos + leading
        end = start + len(stripped)

        result.append(
            Chunk(
                source=MinimalSource(
                    file_path=chunk.source.file_path,
                    first_character_index=start,
                    last_character_index=end - 1,
                ),
                text=stripped,
                heading_path=chunk.heading_path.copy(),
                size=len(stripped),
            )
        )
        current_pos += len(line)

    return result


def split_by_sentences(chunk: Chunk) -> list[Chunk]:
    """Split an oversized chunk/line into sentences while preserving source offsets."""
    result: list[Chunk] = []

    for match in re.finditer(
        r"(?s)([^.!?]+[.!?]+(?:\s+|$)|[^.!?]+$)",
        chunk.text,
    ):
        sentence = match.group(0).strip()

        if not sentence:
            continue

        relative_start = match.start(0)

        leading = len(match.group(0)) - len(match.group(0).lstrip())
        trailing = len(match.group(0).rstrip())

        relative_start += leading
        relative_end = match.start(0) + trailing

        absolute_start = (
            chunk.source.first_character_index
            + relative_start
        )
        absolute_end = (
            chunk.source.first_character_index
            + relative_end
        )

        result.append(
            Chunk(
                source=MinimalSource(
                    file_path=chunk.source.file_path,
                    first_character_index=absolute_start,
                    last_character_index=absolute_end - 1,
                ),
                text=sentence,
                heading_path=chunk.heading_path.copy(),
                size=len(sentence),
            )
        )

    return result


MAX_CHUNK_SIZE = 40

with open("chunking_practice.md", encoding="utf-8") as f:
    content = f.read()

    # Step 1: Get initial markdown section chunks
    markdown_chunks = split_markdown(content, "chunking_practice.md")
    final_chunks = []

    for md_chunk in markdown_chunks:
        # Check Markdown Section Size
        if md_chunk.size <= MAX_CHUNK_SIZE:
            final_chunks.append(md_chunk)
            continue

        # Step 2: Fallback to Paragraphs
        paragraphs = split_paragraphs(md_chunk)
        for para in paragraphs:
            if para.size <= MAX_CHUNK_SIZE:
                final_chunks.append(para)
                continue

            # Step 3: Fallback to Lines
            lines = split_lines(para)
            for line in lines:
                if line.size <= MAX_CHUNK_SIZE:
                    final_chunks.append(line)
                    continue

                # Step 4: Fallback to Sentences
                sentences = split_by_sentences(line)
                for sentence in sentences:
                    final_chunks.append(sentence)

for i, chunk in enumerate(final_chunks, start=1):
    print(f"\n{'=' * 60}")
    print(f"CHUNK {i} (Size: {chunk.size})")
    print(f"{'=' * 60}")
    print("TEXT:", chunk.text)
    print("HEADING PATH:", chunk.heading_path)
    print("SOURCE:", chunk.source)