from src.chunker import PDFChunker
import json
# from src.embedder import Embedder

def main():
    try:
        # 1. Load PDF and initialize chunker
        chunker = PDFChunker("data/Sample HR Policy Manual.pdf")

        # 2. Parse the table of contents
        toc = chunker.parse_toc_structure()

        # 3. Extract chunks
        subtopic_chunks, chapter_chunks = chunker.extract_subtopic_and_chapter_chunks(toc)

        # # 4. Print subtopic chunk summary
        # print("\n--- Subtopic Chunks ---")
        # for sub in subtopic_chunks[:5]:  # limit to first 5
        #     print(f"Chapter: {sub.chapter_id} - {sub.chapter_title}")
        #     print(f"Subtopic: {sub.subtopic_title}")
        #     print(f"Pages: {sub.page_range}")
        #     print(f"Text preview: {sub.text[:200]}...\n")  # Preview only first 200 chars

        # # 5. Print chapter summary
        # print("\n--- Chapter Chunks ---")
        # for chap in chapter_chunks[:3]:  # limit to first 3
        #     print(f"Chapter: {chap.chapter_id} - {chap.chapter_title}")
        #     print(f"Pages: {chap.page_range}")
        #     print(f"Subtopics: {[s.subtopic_title for s in chap.subtopics]}")
        #     print(f"Text preview: {chap.text[:200]}...\n")

        # # 6. Save chunks to file just for testing purposes
        # chunker.save_chunks(subtopic_chunks, chapter_chunks, "chunks.json")

        # 6. start embedding

        # 7. Cleanup
        chunker.close()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
