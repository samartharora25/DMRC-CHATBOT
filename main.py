from src.chunker import PDFChunker
import json

def main():
    try:
        pdf_chunker = PDFChunker("data/Sample HR Policy Manual.pdf")
        toc_structure = pdf_chunker.parse_toc_structure()
        print(json.dumps(toc_structure, indent=2))

        subtopics, chapters = pdf_chunker.extract_subtopic_and_chapter_chunks(toc_structure)
        print("Subtopics:")
        for sub in subtopics:
            print(f" - {sub.title} (Chapter: {sub.chapter}, Pages: {sub.page_range})")

        print("Chapters:")
        for chap in chapters:
            print(f" - {chap.title} (Pages: {chap.page_range})")

        # Close the PDF document
        pdf_chunker.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
