#!/usr/bin/env python3
"""将 PDF 按 200 页一份自动拆分，输出文件名格式：原文件名_part1.pdf"""

import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter

PAGES_PER_PART = 200


def split_pdf(pdf_path: str, pages_per_part: int = PAGES_PER_PART):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"文件不存在: {pdf_path}")
        sys.exit(1)

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    print(f"总页数: {total_pages}，每份 {pages_per_part} 页，共 {-(total_pages // -pages_per_part)} 份")

    stem = pdf_path.stem
    output_dir = pdf_path.parent

    for i in range(0, total_pages, pages_per_part):
        part_num = i // pages_per_part + 1
        end = min(i + pages_per_part, total_pages)

        writer = PdfWriter()
        for page_idx in range(i, end):
            writer.add_page(reader.pages[page_idx])

        output_path = output_dir / f"{stem}_part{part_num}.pdf"
        with open(output_path, "wb") as f:
            writer.write(f)
        print(f"  已生成: {output_path.name}  (第 {i + 1}-{end} 页)")

    print("拆分完成。")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <pdf文件路径> [每份页数]")
        sys.exit(1)

    path = sys.argv[1]
    per_part = int(sys.argv[2]) if len(sys.argv) > 2 else PAGES_PER_PART
    split_pdf(path, per_part)
