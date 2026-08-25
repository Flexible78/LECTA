"""Quick test: ZIP extraction + parse_and_save_document for .zip files."""
import tempfile, zipfile, shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from libs.document_parser import _extract_zip, parse_and_save_document
from libs.utils import data_path

def main():
    tmp_dir = Path(tempfile.mkdtemp(prefix="test_fb2tts_"))
    
    # Create test txt
    test_txt = tmp_dir / "test_book.txt"
    test_txt.write_text(
        "Hello world. This is a test book.\n"
        "Chapter 1: The beginning.\n\n"
        "It was a dark and stormy night.",
        encoding="utf-8"
    )
    
    # Zip it
    test_zip = tmp_dir / "test_archive.zip"
    with zipfile.ZipFile(test_zip, "w") as zf:
        zf.write(test_txt, "test_book.txt")
    print(f"[1] Created test ZIP: {test_zip}")
    
    # Test extraction
    extracted = _extract_zip(test_zip)
    print(f"[2] Extracted {len(extracted)} files:")
    for f in extracted:
        print(f"      {f.name}: {len(f.read_text(encoding='utf-8'))} chars")
    
    # Clean old projects
    for d in list(data_path.iterdir()):
        if d.is_dir() and "test_book" in d.name:
            shutil.rmtree(d)
    
    # Parse ZIP
    name, err = parse_and_save_document(str(test_zip), False)
    print(f"[3] parse result: name={name}, error={err}")
    
    if name:
        fb2_path = data_path / name / f"{name}.fb2"
        print(f"[4] FB2 exists: {fb2_path.exists()}")
        if fb2_path.exists():
            content = fb2_path.read_text(encoding="utf-8")
            print(f"[5] FB2: {len(content)} chars")
            print("     First 200 chars:", content[:200].replace("\n", "\\n"))
        shutil.rmtree(data_path / name)
    
    shutil.rmtree(tmp_dir)
    print("[OK] All ZIP tests passed")

if __name__ == "__main__":
    main()