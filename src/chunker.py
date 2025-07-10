import fitz
from dataclasses import dataclass
import re
from typing import List, Tuple
import json

@dataclass
class SubtopicBlock:
    """
    A data structure representing a subtopic within a chapter of a PDF document.
    
    This class stores information about individual subtopics found within chapters,
    including their metadata, content, and location within the document.
    
    Attributes:
        chapter_id (str): The identifier of the parent chapter (e.g., "Chapter 1")
        chapter_title (str): The full title of the parent chapter
        subtopic_title (str): The title/heading of this specific subtopic
        page_range (Tuple[int, int]): A tuple containing (start_page, end_page) where this subtopic appears
        text (str): The complete text content of this subtopic
    
    Format Example:
        {
            "chapter_id": "Chapter 1",
            "chapter_title": "Introduction to HR Policies",
            "subtopic_title": "EMPLOYEE ONBOARDING",
            "page_range": (15, 18),
            "text": "The onboarding process includes orientation, documentation..."
        }
    """
    chapter_id: str
    chapter_title: str
    subtopic_title: str
    page_range: Tuple[int, int]
    text: str


@dataclass
class ChapterBlock:
    """
    A data structure representing a complete chapter of a PDF document.
    
    This class stores comprehensive information about an entire chapter,
    including all its subtopics and the complete chapter content.
    
    Attributes:
        chapter_id (str): The identifier of the chapter (e.g., "Chapter 1", "Chapter 2")
        chapter_title (str): The full title of the chapter
        page_range (Tuple[int, int]): A tuple containing (start_page, end_page) of the entire chapter
        text (str): The complete raw text content of the entire chapter
        subtopics (List[SubtopicBlock]): A list of all SubtopicBlock objects found within this chapter
    
    Format Example:
        {
            "chapter_id": "Chapter 1",
            "chapter_title": "Introduction to HR Policies",
            "page_range": (14, 25),
            "text": "Chapter 1: Introduction to HR Policies\n\nThis chapter covers...",
            "subtopics": [
                {
                    "chapter_id": "Chapter 1",
                    "chapter_title": "Introduction to HR Policies",
                    "subtopic_title": "EMPLOYEE ONBOARDING",
                    "page_range": (15, 18),
                    "text": "The onboarding process..."
                },
                {
                    "chapter_id": "Chapter 1",
                    "chapter_title": "Introduction to HR Policies",
                    "subtopic_title": "PERFORMANCE EVALUATION",
                    "page_range": (19, 23),
                    "text": "Performance evaluations..."
                }
            ]
        }
    """
    chapter_id: str
    chapter_title: str
    page_range: Tuple[int, int]
    text: str
    subtopics: List[SubtopicBlock]

class PDFChunker:
    """
    A comprehensive PDF document parser that extracts structured content from PDF files.
    
    This class is designed to parse PDF documents with a specific structure (chapters and subtopics)
    and extract them into organized chunks. It's particularly useful for processing policy documents,
    manuals, or any structured PDF content with hierarchical organization.
    
    The parser identifies:
    - Chapter structures from Table of Contents (TOC)
    - Subtopic headings within chapters using font and color analysis
    - Page ranges for each section
    - Complete text content for each chunk
    
    Attributes:
        HEADING_FONT (str): The font name used for main headings
        HEADING_COLOR (int): The color value used for main headings
        doc (fitz.Document): The PyMuPDF document object for the opened PDF
    """
    HEADING_FONT = "AGaramondPro-Bold"
    HEADING_COLOR = 31926  # Hex for detected heading color

    def __init__(self, pdf_path):
        """
        Initialize the PDFChunker with a PDF file.
        
        Args:
            pdf_path (str): The file path to the PDF document to be processed
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            fitz.FileDataError: If the file is not a valid PDF
        """
        print(f"Opening PDF: {pdf_path}")
        self.doc = fitz.open(pdf_path)

    def is_subtopic_heading(self, text, font_name, color_val):
        """
        Detect subtopic headings based on specific text, font, and color criteria.
        
        This method uses a combination of text formatting rules and visual properties
        to identify subtopic headings within the PDF content. It applies strict
        filtering to avoid false positives.
        
        Args:
            text (str): The text content to analyze
            font_name (str): The font name of the text span
            color_val (int): The color value of the text span
            
        Returns:
            bool: True if the text is identified as a subtopic heading, False otherwise
            
        Criteria for subtopic headings:
            - Text must be in uppercase
            - Length must be between 3-8 words
            - Must not be numeric or special characters
            - Must not start with 'CHAPTER '
            - Text length must be between 5-50 characters
            - Must use specific font: "Cambria-Bold"
            - Must use specific color: 32183
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
        """
        Parse the Table of Contents (TOC) to extract chapter structure and page mappings.
        
        This method analyzes the TOC pages to identify chapters, their titles, and
        corresponding page numbers. It builds a structured mapping of the document's
        organization.
        
        Args:
            start_page (int, optional): The starting page number of the TOC (0-indexed). Defaults to 4.
            end_page (int, optional): The ending page number of the TOC (0-indexed). Defaults to 10.
            offset (int, optional): Page number offset to adjust for differences between 
                                  TOC page numbers and actual PDF page numbers. Defaults to 10.
            last_page (int, optional): The last page number of the document. Defaults to 208.
            
        Returns:
            dict: A dictionary mapping chapter IDs to their information with the following structure:
                {
                    "Chapter 1": {
                        "name": "Chapter Title",
                        "page": {
                            "start": 15,
                            "end": 25
                        }
                    },
                    "Chapter 2": {
                        "name": "Another Chapter Title",
                        "page": {
                            "start": 26,
                            "end": 40
                        }
                    }
                }
                
        Process:
            1. Reads text blocks from TOC pages
            2. Identifies lines with chapter titles and page numbers using regex
            3. Extracts chapter IDs and titles
            4. Calculates page ranges for each chapter
            5. Returns structured chapter mapping
        """
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
        """
        Extract structured chunks from the PDF based on the TOC structure.
        
        This method processes each chapter identified in the TOC and extracts:
        - Individual subtopic chunks with their specific content
        - Complete chapter chunks containing all subtopics
        
        The extraction process analyzes font properties and text formatting to
        identify subtopic boundaries and creates organized data structures.
        
        Args:
            toc (dict): The table of contents structure returned by parse_toc_structure().
                       Must contain chapter IDs as keys with "name" and "page" information.
                       
        Returns:
            tuple: A tuple containing two lists:
                - subtopic_chunks (List[SubtopicBlock]): List of all extracted subtopic blocks
                - chapter_chunks (List[ChapterBlock]): List of all extracted chapter blocks
                
        Process for each chapter:
            1. Iterate through all pages in the chapter range
            2. Analyze text spans for font and color properties
            3. Identify subtopic headings using is_subtopic_heading()
            4. Collect text content between subtopic headings
            5. Create SubtopicBlock objects for each identified subtopic
            6. Create ChapterBlock object containing all chapter content and subtopics
            7. Return organized chunks for further processing
            
        Output Format:
            subtopic_chunks: [
                SubtopicBlock(
                    chapter_id="Chapter 1",
                    chapter_title="HR Policies Introduction",
                    subtopic_title="EMPLOYEE ONBOARDING",
                    page_range=(15, 18),
                    text="Complete subtopic content..."
                ),
                ...
            ]
            
            chapter_chunks: [
                ChapterBlock(
                    chapter_id="Chapter 1",
                    chapter_title="HR Policies Introduction",
                    page_range=(14, 25),
                    text="Complete chapter content...",
                    subtopics=[list of SubtopicBlock objects]
                ),
                ...
            ]
        """
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

                            # Use the specific subtopic heading detection
                            is_heading = self.is_subtopic_heading(text, font_name, color_val)

                            if is_heading:
                                if current_subtopic_title:
                                    sub_text = " ".join(current_buffer).strip()
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
            chapter_chunks.append(chapter_block)

        print("\nExtraction completed.")
        return subtopic_chunks, chapter_chunks

    def save_chunks(self, subtopics, chapters, filename):
        """
        Save extracted chunks to a JSON file for testing and analysis purposes.
        
        This method serializes the extracted subtopic and chapter chunks into a
        structured JSON format that can be easily loaded and analyzed later.
        It handles proper encoding and formatting for readability.
        
        Args:
            subtopics (List[SubtopicBlock]): List of extracted subtopic blocks
            chapters (List[ChapterBlock]): List of extracted chapter blocks  
            filename (str): The output filename for saving the chunks.
                          If the filename ends with '.pdf', it will be changed to '.txt'
                          to avoid file format confusion.
                          
        Returns:
            None: This method doesn't return anything but saves data to file
            
        Output JSON Structure:
            {
                "subtopics": [
                    {
                        "chapter_id": "Chapter 1",
                        "chapter_title": "HR Policies Introduction", 
                        "subtopic_title": "EMPLOYEE ONBOARDING",
                        "page_range": [15, 18],
                        "text": "Complete subtopic content..."
                    },
                    ...
                ],
                "chapters": [
                    {
                        "chapter_id": "Chapter 1",
                        "chapter_title": "HR Policies Introduction",
                        "page_range": [14, 25], 
                        "text": "Complete chapter content...",
                        "subtopics": [
                            {
                                "chapter_id": "Chapter 1",
                                "chapter_title": "HR Policies Introduction",
                                "subtopic_title": "EMPLOYEE ONBOARDING", 
                                "page_range": [15, 18],
                                "text": "Complete subtopic content..."
                            },
                            ...
                        ]
                    },
                    ...
                ]
            }
            
        Features:
            - Automatically converts .pdf extension to .txt
            - Uses UTF-8 encoding for proper character support
            - Formats JSON with indentation for readability
            - Handles tuple serialization using default=str
            - Prints confirmation message upon successful save
        """
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
        """
        Close the PDF document and free up resources.
        
        This method properly closes the PyMuPDF document object to prevent
        memory leaks and release file handles. Should be called when done
        processing the PDF document.
        
        Args:
            None
            
        Returns:
            None
            
        Note:
            Always call this method when finished with the PDFChunker to
            ensure proper cleanup of resources.
        """
        self.doc.close()