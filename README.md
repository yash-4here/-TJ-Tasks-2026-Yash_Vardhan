𝐄𝐚𝐬𝐲: 𝐌𝐞𝐬𝐬𝐚𝐠𝐞 𝐂𝐥𝐚𝐬𝐬𝐢𝐟𝐢𝐞𝐫

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
![image alt](https://github.com/yash-4here/-TJ-Tasks-2026-Yash_Vardhan/blob/0ca84cf6ef2ce3f4408222ad81494149dfa6056a/easy.png)
