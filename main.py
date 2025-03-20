import time, sys
from google import genai

class app:
  def __init__(self):
    self.API=""

  def document_gen(self, data)
     pass
  
  def generate(self,promt):
    client = genai.Client(api_key=self.API)
    response = client.models.generate_content(
       model="gemini-2.0-flash", contents=str(prompt)
    )
    app.document_gen(self, response.text)
  
  def menu(self):
    print(
    x=input("Promt: ")
    if x!="exit":
      app.generate(self, x)
    else:
      sys.exit()

try:
  AI=app()
  AI.menu()
finally:
  time.sleep(3)
  sys.exit()
