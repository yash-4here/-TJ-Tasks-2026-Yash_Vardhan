Easy: Message Classifier

Basic NLP without any fancy libraries, just plain Python.

How it works:

Lowercase the sentence + split it into words
Toss out boring stop words like "the", "is", "my" — keep only the words that actually matter
Check what's left against 3 buckets:
Words like crash, bug, broken → Technical Support
Words like bill, payment, charged → Billing Support
Nothing matches → General Inquiry

No ML model here, just clean if-else logic. Simple but it works

Output:
![easy output](Users/yashvardhan/downloads/easy.png)
