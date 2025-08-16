from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor, Inches
from datetime import datetime
from typing import Optional, List
from io import BytesIO
import pandas as pd

try:
    from pptx import Presentation
    from pptx.util import Inches as PPTInches
    PPTX_AVAILABLE = True
except Exception:
    PPTX_AVAILABLE = False

class DocumentTemplate:
    def apply_template(self, doc: Document, content: str, prompt: str, model: str) -> Document:
        for i in range(len(doc.paragraphs) - 1, -1, -1):
            p = doc.paragraphs[i]
            p._element.getparent().remove(p._element)

        header = doc.add_heading("AI Generated Document", level=1)
        header.alignment = WD_ALIGN_PARAGRAPH.CENTER

        doc.add_heading("About this document", level=2)
        metadata = doc.add_paragraph()
        metadata.add_run("Generated on: ").bold = True
        metadata.add_run(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        metadata.add_run("\nPrompt: ").bold = True
        prompt_run = metadata.add_run(prompt)
        prompt_run.italic = True
        metadata.add_run("\nModel: ").bold = True
        metadata.add_run(model)

        doc.add_paragraph("").paragraph_format.space_after = Pt(12)

        doc.add_heading("Generated Content", level=2)

        for paragraph_text in content.split('\n\n'):
            if not paragraph_text.strip():
                continue

            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(10)
            p.paragraph_format.first_line_indent = Inches(0.25)
            p.add_run(paragraph_text)

        doc.add_paragraph("").paragraph_format.space_before = Pt(20)
        footer = doc.add_paragraph(f"Generated using {model.split(':')[0].upper()} AI")
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer_run = footer.runs[0]
        footer_run.font.size = Pt(9)
        footer_run.font.color.rgb = RGBColor(100, 100, 100)

        return doc

    def add_image(self, doc: Document, image_bytes: bytes, width_inches: float = 6.0, caption: Optional[str] = None) -> Document:
        image_stream = BytesIO(image_bytes)
        paragraph = doc.add_paragraph()
        run = paragraph.add_run()
        run.add_picture(image_stream, width=Inches(width_inches))

        if caption:
            caption_p = doc.add_paragraph(caption)
            caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption_run = caption_p.runs[0]
            caption_run.font.size = Pt(9)
            caption_run.italic = True

        return doc

    def add_table(self, doc: Document, data: List[List[str]], headers: Optional[List[str]] = None) -> Document:
        if headers:
            table = doc.add_table(rows=1, cols=len(headers))
            hdr_cells = table.rows[0].cells
            for i, h in enumerate(headers):
                hdr_cells[i].text = str(h)
            for row in data:
                cells = table.add_row().cells
                for i, cell in enumerate(row):
                    cells[i].text = str(cell)
        else:
            table = doc.add_table(rows=0, cols=len(data[0]) if data else 0)
            for row in data:
                cells = table.add_row().cells
                for i, cell in enumerate(row):
                    cells[i].text = str(cell)

        return doc

    def create_presentation(self, title: str, slides: List[dict]) -> Optional[bytes]:
        if not PPTX_AVAILABLE:
            return None

        prs = Presentation()
        title_slide_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(title_slide_layout)
        slide.shapes.title.text = title

        for s in slides:
            layout = prs.slide_layouts[1] if len(s.get('content', '')) > 0 else prs.slide_layouts[5]
            slide = prs.slides.add_slide(layout)
            if 'title' in s:
                slide.shapes.title.text = s['title']
            if 'content' in s and s['content']:
                body = slide.shapes.placeholders[1].text = s['content']
            if 'image' in s and s['image']:
                img_stream = BytesIO(s['image'])
                slide.shapes.add_picture(img_stream, PPTInches(1), PPTInches(1), width=PPTInches(8))

        out = BytesIO()
        prs.save(out)
        return out.getvalue()