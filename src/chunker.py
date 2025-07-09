import fitz
from dataclasses import dataclass
import re
from typing import List, Tuple
import json
@dataclass
class SubtopicBlock:
    chapter_id: str
    chapter_title: str
    subtopic_title: str
    page_range: Tuple[int, int]
    text: str


@dataclass
class ChapterBlock:
    chapter_id: str
    chapter_title: str
    page_range: Tuple[int, int]
    text: str
    subtopics: List[SubtopicBlock]

class PDFChunker:
    HEADING_FONT = "AGaramondPro-Bold"
    HEADING_COLOR = 31926  # Hex for detected heading color

    def __init__(self, pdf_path):
        print(f"Opening PDF: {pdf_path}")
        self.doc = fitz.open(pdf_path)

    def is_subtopic_heading(self, text, font_name, color_val):
        """
        Detect only subtopic headings with very specific criteria
        """
        # Must be uppercase and reasonable length
        if not (text == text.upper() and 3 <= len(text.split()) <= 8):
            return False
        
        # Skip obvious non-headings
        if text.isdigit() or text in ['•', '*', ':', '(', ')', '@']:
            return False
        
        # Skip chapter titles
        if text.startswith('CHAPTER '):
            return False
        
        # Skip very short or very long text
        if len(text) < 5 or len(text) > 50:
            return False
        
        # Specific font and color for subtopics
        return font_name == "Cambria-Bold" and color_val == 32183

    def parse_toc_structure(self, start_page=4, end_page=10, offset=10, last_page=208):
        print("Parsing TOC structure...")
        toc = {}
        index_pages = {i: self.doc.load_page(i) for i in range(start_page, end_page)}

        all_blocks = []
        for idx, page in index_pages.items():
            print(f"Reading index page {idx + 1}")
            for block in page.get_text("blocks"): # type: ignore
                x0 = block[0]
                text = block[4].strip()
                if not text:
                    continue
                for line in text.splitlines():
                    line = line.strip()
                    match = re.match(r'^(.*?)\s*[.\s]{5,}\s*(\d{1,3})$', line)
                    if not match:
                        continue

                    title = match.group(1).strip()
                    page_num = int(match.group(2)) + offset

                    chapter_match = re.match(r'^(Chapter\s+\d+)\s+(.*)', title)
                    if chapter_match:
                        chapter_id = chapter_match.group(1)
                        chapter_name = chapter_match.group(2).strip()
                        print(f"Detected chapter: {chapter_id} - {chapter_name} at page {page_num}")
                        toc[chapter_id] = {
                            "name": chapter_name,
                            "page": {"start": page_num, "end": last_page}
                        }

        chapter_ids = list(toc.keys())
        for i in range(len(chapter_ids)):
            if i < len(chapter_ids) - 1:
                toc[chapter_ids[i]]["page"]["end"] = toc[chapter_ids[i + 1]]["page"]["start"] - 1
            else:
                toc[chapter_ids[i]]["page"]["end"] = last_page

        print("TOC parsing completed. Chapters found:")
        for chapter_id, data in toc.items():
            print(f"  {chapter_id}: {data['name']} (pages {data['page']['start']} to {data['page']['end']})")

        return toc

    def extract_subtopic_and_chapter_chunks(self, toc):
        print("\nExtracting subtopic and chapter chunks...")
        subtopic_chunks = []
        chapter_chunks = []

        for chapter_id, info in toc.items():
            chapter_title = info["name"]
            chapter_start, chapter_end = info["page"]["start"], info["page"]["end"]
            chapter_text = ""

            print(f"\nProcessing {chapter_id}: {chapter_title} (pages {chapter_start}-{chapter_end})")

            subtopics = []
            current_subtopic_title = None
            current_buffer = []
            subtopic_start_page = chapter_start

            for page_num in range(chapter_start - 1, chapter_end):
                print(f"  Reading page {page_num + 1}")
                page = self.doc.load_page(page_num)
                blocks = page.get_text("dict")["blocks"] # type: ignore

                for block in blocks:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span["text"].strip()
                            if not text:
                                continue

                            font_name = span.get("font", "")
                            color_val = span.get("color", 0)

                            if text == text.upper() and len(text.split()) <= 8:
                                print(f"[DEBUG] Page {page_num+1}: '{text}' | Font: {font_name} | Color: {color_val}")
                            
                            # Use the specific subtopic heading detection
                            is_heading = self.is_subtopic_heading(text, font_name, color_val)

                            if is_heading:
                                print(f"    [HEADING DETECTED] '{text}' on page {page_num + 1} (Font: {font_name}, Color: {color_val})")
                                if current_subtopic_title:
                                    sub_text = " ".join(current_buffer).strip()
                                    print(f"      -> Saving subtopic: {current_subtopic_title} (pages {subtopic_start_page}-{page_num + 1})")
                                    sub_block = SubtopicBlock(
                                        chapter_id=chapter_id,
                                        chapter_title=chapter_title,
                                        subtopic_title=current_subtopic_title,
                                        page_range=(subtopic_start_page, page_num + 1),
                                        text=sub_text
                                    )
                                    subtopic_chunks.append(sub_block)
                                    subtopics.append(sub_block)

                                current_subtopic_title = text
                                current_buffer = []
                                subtopic_start_page = page_num + 1
                            else:
                                current_buffer.append(text)

                chapter_text += page.get_text("text") + "\n" # type: ignore

            if current_subtopic_title and current_buffer:
                sub_text = " ".join(current_buffer).strip()
                print(f"    -> Saving final subtopic: {current_subtopic_title} (pages {subtopic_start_page}-{chapter_end})")
                sub_block = SubtopicBlock(
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    subtopic_title=current_subtopic_title,
                    page_range=(subtopic_start_page, chapter_end),
                    text=sub_text
                )
                subtopic_chunks.append(sub_block)
                subtopics.append(sub_block)

            chapter_block = ChapterBlock(
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                page_range=(chapter_start, chapter_end),
                text=chapter_text.strip(),
                subtopics=subtopics
            )
            print(f"  -> Chapter '{chapter_title}' done. {len(subtopics)} subtopics found.")
            chapter_chunks.append(chapter_block)

        print("\nExtraction completed.")
        return subtopic_chunks, chapter_chunks

    def save_chunks(self, subtopics, chapters, filename):  # this function is just for testing purposes
        # Ensure the file is saved as .txt, not .pdf, to avoid PDF structure errors
        if filename.lower().endswith('.pdf'):
            filename = filename[:-4] + '.txt'
            print(f"Warning: Changed output file extension to .txt to avoid PDF corruption.")

        # Prepare data for JSON serialization
        data = {
            "subtopics": [sub.__dict__ for sub in subtopics],
            "chapters": [
            {
                **chap.__dict__,
                "subtopics": [s.__dict__ for s in chap.subtopics]
            }
            for chap in chapters
            ]
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"✅ Saved chunks to {filename}")

    def close(self):
        self.doc.close()