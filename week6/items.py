from typing import Optional
from transformers import AutoTokenizer
import re

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
MAX_TOKENS = 160
MIN_TOKENS = 150
MIN_CHAR = 300
CEILING_CHAR = MAX_TOKENS*7

class Item:

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    removals = ['"Batteries Included?": "No"', '"Batteries Included?": "Yes"', '"Batteries Required?": "No"', 
                '"Batteries Required?": "Yes"', "By Manufacturer", "Item", "Date First", 
                "Package", ":", "Number of", "Best Sellers", "Number", "Product "]
    prefix = "Price is $"
    question = "How much does this cost to the nearest dollar?"

    title: str
    price:float
    category:str
    token_count:int = 0
    details:Optional[str]
    prompt:Optional[str]
    include = False

    def __init__(self, data, price):
        self.title = data["title"]
        self.price = price
        self.parse(data)

    def scrub_details(self):
        details = self.details
        for remove in self.removals:
            details = details.replace(remove, "")
        return details

    def scrub(self, stuff):
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        stuff = stuff.replace(" ,", ",").replace(",,,",",").replace(",,",",")
        words = stuff.split(" ")
        select = [word for word in words if len(word)<7 or not any(char.isdigit()) for char in word]
        return " ".join(select)
    
    def parse(self, data):
        contents = "\n".join(data["description"])
        if contents:
            contents += "\n"
        features =  "\n".join(data["features"])
        if features:
            contents += features 
        self.details = "\n".join(data["details"])
        if self.details:
            contents += self.scrub_details() + "\n"
        
        if len(contents)>MIN_CHAR:
            contents = contents[:CEILING_CHAR]
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            if len(tokens)> MIN_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                text = self.tokenizer.decode(tokens)
                self.make_prompt(text)
                self.include = True

    
    def make_prompt(self, text):
        self.prompt = f"{self.question}\n\n{text}\n\n"
        self.prompt += f"{self.prefix}{str(round(self.price))}.00"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    
    def test_prompt(self):
        return self.prompt.split(self.prefix)[0] + self.prefix
    
    def __repr__(self):
        return f"<{self.title} = ${self.price}>"