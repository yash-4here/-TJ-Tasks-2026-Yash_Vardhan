STOP_WORDS = ["the", "is", "at", "which", "and", "to", "a", "an", "for", "my", "i"]
technical = ["crash","bug","broken","error"]
billing = ["bill","charged","payment","subscription"]

def classify_message(sentence):
   if any(word in sentence for word in technical):
       print("Technical support")
   elif any(word in sentence for word in billing):   
       print("Billing support")
   else:
       print("General support")

def clean_and_tokenize(sentence):
    sentence=sentence.lower()
    new= " ".join(
        word for word in sentence.split()
        if word not in STOP_WORDS
)
    print(new)
    return new

sentence = str(input("Write a sentence to clean and classify it : "))
classify_message(clean_and_tokenize(sentence))
classify_message(clean_and_tokenize("the app crashes every time I open it"))
