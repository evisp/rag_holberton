import os
import frontmatter
from pathlib import Path
from app.config import DOCS_PATH


def load_documents(docs_path: Path = DOCS_PATH) -> list[dict]:
    """
    Walk docs_path, parse every .md file.
    Returns a list of dicts with keys: filename, title, content, filepath
    """
    documents = []

    md_files = sorted(docs_path.glob("*.md"))

    if not md_files:
        print(f"Warning: no .md files found in {docs_path}")
        return documents

    for filepath in md_files:
        try:
            post = frontmatter.load(filepath)

            title = (
                post.metadata.get("title")
                or _extract_title_from_content(post.content)
                or filepath.stem.replace("-", " ").replace("_", " ").title()
            )

            content = post.content.strip()
            if not content:
                print(f"Skipping empty file: {filepath.name}")
                continue

            documents.append({
                "filename": filepath.name,
                "filepath": str(filepath),
                "title": title,
                "content": content,
            })

        except Exception as e:
            print(f"Error loading {filepath.name}: {e}")
            continue

    print(f"Loaded {len(documents)} documents from {docs_path}")
    return documents


def _extract_title_from_content(content: str) -> str | None:
    """Pull the first # heading from the markdown as the title."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return None