<html>
<head>

  <body>
    <h1>Easy:Message Classifier</h1>

  </body>
</head>
  
</html>

Basic NLP without any fancy libraries, just plain Python.

How it works:

Lowercase the sentence + split it into words
Toss out boring stop words like "the", "is", "my" — keep only the words that actually matter
Check what's left against 3 buckets:
Words like crash, bug, broken → Technical Support
Words like bill, payment, charged → Billing Support
Nothing matches → General Inquiry

No ML model here, just clean if-else logic. Simple but it works

OUTPUT:
![image alt](https://github.com/yash-4here/-TJ-Tasks-2026-Yash_Vardhan/blob/0ca84cf6ef2ce3f4408222ad81494149dfa6056a/easy.png)


<html>
<head>

  <body>
    <h1>Medium: Perceptron from Scratch</h1>

  </body>
</head>
  
</html>

Built a single neuron (perceptron) by hand to see how neural nets actually make decisions under the hood, instead of jumping straight to PyTorch.

How it works:

Two inputs: study_hours and attendance (both 0–1)
Hardcoded weights (w1=3.5, w2=2.0) + a bias (-3.0)
Multiply inputs by weights, add bias → get one number (the "weighted sum")
Run that number through a step function: >= 0 → Pass, else → Fail
Printed every single step so you can literally watch the math happen 🔍

Basically this is what one neuron does inside any real neural network, just done manually instead of learned.

OUTPUT:
![image alt](https://github.com/yash-4here/-TJ-Tasks-2026-Yash_Vardhan/blob/fb0313bc42cd4858b16ce659bb329a22fe5851aa/medium.png)


<html>
<head>

  <body>
    <h1>Hard: CNN on MNIST</h1>

  </body>
</head>
  
</html>

Built and trained an actual CNN in PyTorch to recognize handwritten digits (0-9).

How it works:

Load MNIST, normalize pixels to [-1, 1]
Architecture: Conv2D (16 filters) → ReLU → MaxPool → Flatten → Dense(10)
Trained with Cross-Entropy Loss + Adam optimizer for 3 epochs
Tested on 10k unseen images to check real accuracy

Got ~97-98% test accuracy — pretty solid for such a tiny model.

OUTPUT:
![image alt](https://github.com/yash-4here/-TJ-Tasks-2026-Yash_Vardhan/blob/fb0313bc42cd4858b16ce659bb329a22fe5851aa/hard.png)

