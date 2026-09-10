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
![easy output]([Users/yashvardhan/downloads/easy.png](https://github.com/yash-4here/-TJ-Tasks-2026-Yash_Vardhan/blob/46eb1003e2e731f4ef278cb151ea6d76f7462c87/easy.png))
