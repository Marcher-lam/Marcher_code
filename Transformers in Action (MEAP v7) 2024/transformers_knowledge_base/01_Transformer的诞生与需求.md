# Transformer 的诞生与需求

> 本章来源：Transformers in Action (MEAP v7) 2024 - Chapter 1

## 学习要点
- 理解 Transformer 为什么被发明——RNN/LSTM 的局限性
- 掌握 Attention 机制的核心思想
- 理解 Multi-Head Attention 的作用和优势
- 了解 Transformer 的使用场景和选择标准

---

# The need for transformers

# This chapter covers

How transformers revolutionized NLP   
Attention mechanism - the transformers key architectural component   
How to use transformers   
When and why you want to use transformers

The eld of natural language processing (NLP) has undergone a revolutionary change with the invention of a new class of neural networks called transformers. These models, famous for their capacity to understand and generate natural language, are the backbone of widely-used applications such as Google Translate or ChatGPT.

Transformers, along with their derivatives like large language models (LLMs), are distinctive due to their unique architectural approach. Unlike their predecessors, like Long Short Term Memory (LSTMs), transformers incorporate an innovative architectural component called the "attention mechanism." This mechanism empowers the model to concentrate on dierent segments of the input data to varying extents, thereby enhancing its ability to process and comprehend complex sequential data. This capability is not only relevant to LLMs processing natural language, but also applies to the broader use of transformers in processing audio streams, images, and even video.

Let’s start by comparing transformers with their predecessors, the LSTM models, and examine each component of a transformer in more detail.

# 1.1 The transformers breakthrough

Transformers are a type of deep learning model that use the attention mechanism to increase processing speed and improve performance in tasks involving sequential data. Introduced in the groundbreaking paper "Attention is all you need" (1), transformers marked a signicant shift from the previously popular recurrent neural network architectures, such as LSTMs. Before the invention of transformers, LSTMs, a particular type of Recurrent Neural Networks (RNNs), were the predominant choice for processing sequential data, including natural language. RNNs, as the name suggests, handle sequences by iterating through elements and maintaining a form of "memory" about the information processed so far. To illustrate this, suppose we want to translate the sentence "I don’t speak French" into "Je ne parle pas français". To translate the sentence, with an RNN, the architectural component of this neural network, the encoder, would process each word in the sentence "I don’t speak French" one word at the time, updating its state with each word. Producing then a so-called context vector of the English sentence. This context vector encapsulates the semantics of the input sentence and serves as the bridge to the next component, the decoder, of the RNN. The decoder then uses this context vector to generate the output sequence: "Je ne parle pas français", again one word at a time, using its internal (recurrent) state to remember the previously generated words. Throughout this process, the recurrent nature of the architecture allows each step to build upon the previous ones, thereby crafting a coherent translation. This sequence-based processing enables the RNN to capture temporal dependencies within the sentence but can also lead to challenges in handling long-distance relationships between words. This process is visualized in gure 1.1.

![](images/127cd71a73e50a5d2a02862b19467564350873cfbd1eb23fbf940abe7e5555b1.jpg)  
Figure 1.1 A high-level overview of the general flow of sequential data within an RNN. The encoder in an RNN processes the input sentence one word at a time, updating its internal state with each word. This culminates in a final context vector representing the entire sentence. The decoder then uses this context vector to generate the output sentence, again one word at a time, using its internal (recurrent) state to remember the previously generated words.

As we can see in gure 1.1, the "recurrent state" is updated for each item in the output sequence, capturing a contextual information about the parts of the sequence it has processed so far. However, RNNs come with some drawbacks, they struggle with long sequences, which makes them less ecient in capturing long-term dependencies in the data.

# HOW ARE TRANSFORMERS DIFFERENT?

Transformers, on the other hand, pioneered a radically dierent strategy. They are built around the "attention mechanism," a powerful concept that enables them to take into account multiple parts of the input sequence concurrently when processing each individual part. Instead of processing the sequence element by element and carrying a single recurrent state forward, the transformer model calculates an attention score for each element in the context of all other elements in the sequence.

This fundamental shift allows transformers to be much more parallelizable and thus more ecient. It also dramatically enhances the model’s capacity to understand intricate patterns and long-term dependencies in the data. Unlike traditional sequence-to-sequence models, transformers leverage this attention mechanism to create a more interconnected understanding of the entire sequence, which enhances the model’s ability to make accurate predictions.

Moreover, the transformer’s architecture enables the simultaneous processing of all elements in the sequence, signicantly reducing computation time. To illustrate this process, again, suppose we want to translate the sentence "I don’t speak French" into "Je ne parle pas français." Now, with the transformer model, the encoder extracts features from the input sentence, "I don’t speak French." These features are then processed by the transformer, with the help of the attention mechanism, to calculate attention scores for each element. These scores eectively capture the contextual relationship between each word and the rest of the sentence. The decoder, another component of the transformer model, uses these attention scores and the extracted features to generate the translated sentence, "Je ne parle pas français." This high-level functionality of the translation with a transformer model is illustrated in gure 1.2.

![](images/9d5cb5f60b80cdfbcc572f9ac5ab807d98b8c2d6c594f33d60f39e3f3d10186b.jpg)  
Figure 1.2 A high-level overview of the general flow of sequential data within a transformer architecture. Starting from the left, an input sentence is passed into the transformer, where it is first processed by the encoder along with the attention mechanism. The result, depicted as attention scores (also called contextual representations), are passed to the decoder with its own attention mechanism. The output is then the translated sentence.

As gure 1.2 demonstrates, the attention mechanism within the transformer model allows for the concurrent processing of the entire sequence, rather than updating a "recurrent state" for each item as in RNNs. This design enables the transformer to eectively capture the contextual relationship between all parts of the sequence, overcoming the challenges faced by traditional models like RNNs that can struggle with long sequences. The result is

a more ecient and powerful model, capable of handling intricate patterns and long-term dependencies in the data.

# 1.1.1 Unveiling the attention mechanism

The attention mechanism in transformers enables the model to weigh dierent positions of a sequence when computing a representation of that sequence. In essence, within the encoder-decoder architecture, the attention mechanism assesses the relevance of various input vectors and assigns higher weights to the most important ones. This stands in contrast to RNNs such as LSTM models, which process input sequences one item at a time.

To make this more clear, let’s revisit our previous translation example where we translated the sentence "I don’t speak French" into "Je ne parle pas français". An ideal translation model needs to comprehend each word in the context of the others in the sentence. In this instance, the translation of the word "speak" into "parle" is inuenced by its surrounding words "I", "don’t", and "French". This is illustrated in gure 1.3 where this connection is represented by the thickness of the edges, which represent the attention scores assigned by the model, demonstrating how the words relate to each other.

![](images/f5b27d29df4b0e68e5a59726db84761be16b854b91bdf7cfea94e7b61a0f0239.jpg)  
Figure 1.3 A sample translation made by a transformer model with attention. The thickness and color of the edges demonstrate the attention scores assigned by the model.

This attention mechanism is a key part of the transformer model, providing it the ability to process sequences with complex relationships between elements. However, our example sentence, while straightforward, demands the model to consider relationships between various words in the sequence through attention. In addition, as sequences grow more intricate and tasks more nuanced, the importance of advanced attention strategies, such as multi-head attention, becomes evident. Lets take a look at multi-head attention as we consider a more complicated NLP task: sentiment classication.

# 1.1.2 The power of multi-head attention

The transformer model introduces a powerful extension to the attention mechanism known as "multi-head attention." Multi-head attention enhances the models ability to capture multiple relationships within the sequence. This approach allows the model to focus on dierent positions of the input simultaneously, capturing various aspects of the information. This means the model can maintain multiple "perspectives" to better understand complex patterns in the data.

To illustrate the benets of multi-head attention, let us take the sentence, "The movie was not bad". In this case, the transformer’s multi-head attention allows the model to parse the interaction of "not" and "bad" simultaneously, thereby understanding that the

overall sentiment is positive. Figure 1.4 visualizes this interaction between the words in the sentence.

![](images/86dd1652572ec33f2b4f2c5d4dc70e8d4375b7c881729253b9dbabb5e92351d7.jpg)  
Figure 1.4 Example of different relations in a sentence.

In contrast, recurrent models like LSTMs often struggle with identifying such longterm dependencies within a sentence, leading to potential diculties in recognizing these subtle nuances. Transformers, with their attention-based architecture, bypass this problem, capturing complex interrelationships within textual data eciently.

In a nutshell, multi-head attention allows transformers to understand various aspects of the input simultaneously, marking a signicant leap over previous models like LSTMs. Using this technique, the transformer was able to reach a score of 41 for a metric called BLEU (BiLingual Evaluation Understudy). This is a very high score and indicates that the generated translation from the transformer model is very similar to the reference text. Even more fascinating is that the rst transformer model achieved this result after only 3.5 days of training. This was only a small fraction of the time it took for earlier state-of-the-art networks, like LSTMs, to achieve a similar high score for the same English-to-French translation task.

Moreover, in comparison to other RNN-based models like LSTMs, the success of the transformer and LLMs is more evident in terms of their revolutionary impact on the eld of NLP. Although LSTMs were signicant advancements for sequence processing, the speed at which LLMs and transformers conquered the eld and state-of-the-art models are being continuously developed is unparalleled. As a result of the transformers’ fast-paced and groundbreaking success, it was eventually possible to build such advanced neural networks such as ChatGPT just ve years after the introduction of the rst transformer model.

# 1.2 How to use transformers

Starting your journey with transformers is greatly simplied through the Transformers library of the machine learning platform Hugging Face. This library provides a vast amount of pre-trained models, facilitating tasks that range from language translation to text generation, text classication and sentiment analysis.

The term "pre-trained" refers to models that have been previously trained on extensive datasets, usually containing millions of documents, encompassing a broad spectrum of topics. This exposure allows the models to learn the nuanced patterns of language, including syntax, semantics, and context. By using these pre-trained models, you essentially gain the ability to leverage their learned knowledge, eliminating the need for training a model from scratch. This approach saves considerable computational resources and time, because

you now only have to train the model for your specic task at hand. For instance, using a labeled dataset consisting of pairs of sentences and the labels neutral, negative or positive, to perform a sentiment analysis.

Another factor to consider when using transformers is the processing power required. Transformers, given their complexity, can be computationally intensive. This is where GPUs come into play. GPUs are designed to carry out many operations concurrently. This makes them well-suited for the matrix calculations and parallel computations that are commonplace in training and using a transformer. Even if you don’t have a high-powered GPU at your disposal, many cloud services, such as Google Colab or Paperspace, oer access to GPUs, making the use of transformers accessible to a wide audience.

Hence, with pre-trained models available through Hugging Face’s Transformers library and the processing power oered by cloud services, you are well-equipped to harness the power of transformers for your own NLP tasks.

# 1.3 When and why you’d want to use transformers

Transformers have become an integral part of the modern machine learning landscape. They especially oer unparalleled capabilities in natural language processing, but their potential reaches even far beyond this eld. As transformers have shown promise in domains like computer vision and audio recognition, hinting at a future where they may become a more general-purpose machine learning architecture.

However, the real charm of transformers lies in their accessibility. The Hugging Face’s Transformers library is home to a diverse range of pre-trained models, making it easier for practitioners to get started. Unlike LSTM-based architectures, which often require training from scratch, transformers are via Hugging Face readily available in pre-trained forms, saving considerable time and computational resources.

This advantage is further magnied by the active community surrounding the Hugging Face’s Transformers library. Being open-source, it benets from constant contributions and improvements by machine learning enthusiasts worldwide. Consequently, it’s often a matter of a few hours to ne-tune a pre-trained transformer model for a task like sentiment analysis.

Furthermore, the rise of zero-shot and few-shot learning techniques has expanded the applicability of transformers. Zero-shot learning refers to a model’s ability to handle tasks it was never specically trained for. Essentially, it can understand and perform unseen tasks based on its broad training. Few-shot learning, on the other hand, implies that the model can quickly learn to perform new tasks after being trained on a very small amount of data related to that task. This technique takes advantage of the model’s pre-existing knowledge from its extensive training. These advanced techniques, combined with the resourcefulness of the community, make transformers an appealing choice for a broad array of tasks in the eld of NLP.

Nonetheless, as impressive as these diverse models are, they do come with some limitations. If we consider the so-called billion-parameter models, which refers to the scale and learnable parameters or "weights" of these transformer models. For instance, early

transformer models had 110 million parameters but as advancements in the eld continues, this has led to even larger models, possessing 175 billion parameters and more.

These "billion-parameter models," exhibit an impressive ability to generate coherent and contextually relevant responses in a conversation. However, despite their vast scale and impressive performance, these models do have their limitations. Their practical applicability might diminish in certain specialized domains like nance or healthcare, where domainspecic context is crucial. The computational and memory demands of these models can make their deployment challenging, especially in real-time systems that require quick and accurate results.

Nevertheless, the potential of these dierent variations of transformer models cannot be underestimated. They are pushing the boundaries of what’s possible in natural language understanding and generation, text classication, translation, and more. But as with any tool, the key lies in a balanced, problem-specic approach. As we move forward, we will explore this balance in more depth, focusing on how to leverage the strengths of these dierent types of models while navigating their specic limitations.

# 1.4 Summary

A transformer model employs attention and multi-head attention mechanisms. Using these tools, it expertly navigates through various parts of a sentence, shining a spotlight (signifying more attention) on the words that are pivotal in shaping the overall narrative and context comprehension.   
Attention allows the model to focus on key portions of the input and emphasizes the most important information.   
The Transformer model’s multi-head attention component helps it to identify numerous links between words in a sequence, leading to its popularity as one of the most extensively used NLP models.   
Transformers have excelled in NLP tasks because of their capacity to handle long-term dependencies in sequential data.   
They revolutionized the eld by making it possible to train a transduction model in a matter of days rather than weeks or months, yet outperforming state-of-the-art networks.   
Advanced techniques like zero-shot or few-shot learning allow large models to infer and generalize about new tasks based on their pre-existing training, without needing explicit retraining, enabling a more ecient use of resources and time.   
Despite their impressive capabilities, extremely large language models aren’t without limitations. A potential issue is the decreased eectiveness in certain specialized domains like nance or healthcare. Furthermore, their computational and memory demands can make deployment challenging. Thus, selecting the most suitable model requires careful consideration of the specic task at hand, balancing the trade-o between model complexity and practical applicability.

# Bibliography

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” CoRR, vol. abs/1706.03762, 2017. 3
