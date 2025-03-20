import time
import sys
import os
from google import genai
from docx import Document

class App:
    def __init__(self):
        self.API = ""  

    def document_gen(self, data):
        filename = f"document_{int(time.time())}.docx"
        filepath = os.path.join(os.getcwd(), filename)
        doc = Document()
        doc.add_paragraph(data)
        doc.save(filepath)
        print(f"Document saved: {filepath}")

    def generate(self, prompt):
        client = genai.Client(api_key=self.API)
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=str(prompt)
        )
        self.document_gen(response.text)

    def menu(self):
        print("v1.0.0")
        print("Type 'exit' to quit or enter a prompt to generate a document.")

        while True:
            x = input("Prompt: ")
            if x.lower() == "exit":
                sys.exit()
            self.generate(x)

try:
    AI = App()
    AI.menu()
finally:
    time.sleep(3)
    sys.exit()
