import fitz
from dataclasses import dataclass
import re
from collections import defaultdict

@dataclass
class SubtopicBlock:
    chapter: str
    title: str
    page_range: tuple
    text: str

class PDFChunker:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        self.index = {}
        # Load pages 5 to 10 (0-based: 4–9)
        for i in range(4, 10):
            self.index[i] = self.doc.load_page(i)

        # Extract text from each index page
        self.index_text_map = {}
        for idx, page in self.index.items():
            self.index_text_map[f"index_text{idx - 3}"] = page.get_text("blocks")

        print("Mapped pages in self.index and their text:")
        for idx in self.index:
            print(f"Page {idx + 1} mapped to self.index[{idx}]")
            print("Text:")
            print(self.index[idx].get_text("blocks"))

    def parse_toc_structure(self, last_page=208):
        all_blocks = []
        for page_idx, blocks in self.index_text_map.items():
            for block in blocks:
                x0 = block[0]
                text = block[4].strip()
                if not text:
                    continue
                for line in text.splitlines():
                    line = line.strip()
                    if not line or not re.search(r'\d{1,3}$', line):
                        continue
                    all_blocks.append({
                        "x0": x0,
                        "text": line
                    })

        # Step 1: Parse blocks into classified entries
        entries = []
        for block in all_blocks:
            x0 = block["x0"]
            line = block["text"]

            # Extract title and page
            match = re.match(r'^(.*?)\s*[.\s]{5,}\s*(\d{1,3})$', line)
            if not match:
                continue

            title = match.group(1).strip()
            page = int(match.group(2)) + 10  # offset applied

            if x0 <= 72:  # chapter
                chapter_match = re.match(r'^(Chapter\s+\d+)\s+(.*)', title)
                if chapter_match:
                    entries.append({
                        "type": "chapter",
                        "chapter_id": chapter_match.group(1),
                        "name": chapter_match.group(2).strip(),
                        "page": page
                    })
            else:  # subtopic
                entries.append({
                    "type": "subtopic",
                    "name": title,
                    "page": page
                })

        # Step 2: Precompute chapter positions
        chapter_indices = [i for i, e in enumerate(entries) if e["type"] == "chapter"]

        # Step 3: Build structured TOC
        toc = {}
        current_chapter = None

        for i, entry in enumerate(entries):
            this_page = entry["page"]

            if entry["type"] == "chapter":
                # Find the start of the next chapter
                next_chapter_index = next((j for j in chapter_indices if j > i), None)
                next_page = entries[next_chapter_index]["page"] if next_chapter_index is not None else last_page
                end_page = next_page - 1

                current_chapter = entry["chapter_id"]
                toc[current_chapter] = {
                    "name": entry["name"],
                    "page": {"start": this_page, "end": end_page},
                    "subtopics": []
                }

            elif entry["type"] == "subtopic" and current_chapter:
                # Subtopic end = start of next entry - 1 (or same chapter’s end)
                next_page = entries[i + 1]["page"] if i + 1 < len(entries) else toc[current_chapter]["page"]["end"] + 1
                end_page = next_page - 1

                toc[current_chapter]["subtopics"].append({
                    "name": entry["name"],
                    "page": {"start": this_page, "end": end_page}
                })

        return toc

    def extract_subtopic_and_chapter_chunks(self, toc):
        subtopic_chunks = []
        chapter_chunks = []

        for chapter_id, info in toc.items():
            chapter_start, chapter_end = info["page"]["start"], info["page"]["end"]
            chapter_text = ""

            # Extract chapter-level text
            for page_num in range(chapter_start - 1, chapter_end):  # 0-based
                page = self.doc.load_page(page_num)
                chapter_text += page.get_text("text") + "\n" # type: ignore 

            # Save full chapter
            chapter_chunks.append(SubtopicBlock(
                chapter=chapter_id,
                title=info["name"],
                page_range=(chapter_start, chapter_end),
                text=chapter_text.strip()
            ))

            # Extract subtopics inside chapter
            for sub in info["subtopics"]:
                sub_start, sub_end = sub["page"]["start"], sub["page"]["end"]
                sub_text = ""
                for page_num in range(sub_start - 1, sub_end):
                    page = self.doc.load_page(page_num)
                    sub_text += page.get_text("text") + "\n" # type: ignore

                subtopic_chunks.append(SubtopicBlock(
                    chapter=chapter_id,
                    title=sub["name"],
                    page_range=(sub_start, sub_end),
                    text=sub_text.strip()
                ))

        return subtopic_chunks, chapter_chunks




    def close(self):
        self.doc.close()
