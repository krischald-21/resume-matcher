from docx import Document
import PyPDF2

# Class to read different file formats
class file_reader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_text_file(self):
        with open(self.file_path, 'r') as file:
            return file.read()
        
    def read_pdf_file(self):
        with open(self.file_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ''
            for page_num in range(reader.numPages):
                text += reader.getPage(page_num).extract_text()
            return text
        
    def read_docx_file(self):
        doc = Document(self.file_path)
        text = ''
        for para in doc.paragraphs:
            text += para.text + '\n'
        return text
    
