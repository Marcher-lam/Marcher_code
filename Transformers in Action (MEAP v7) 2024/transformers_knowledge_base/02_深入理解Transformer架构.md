# 深入理解 Transformer 架构

> 本章来源：Transformers in Action (MEAP v7) 2024 - Chapter 2

## 学习要点
- 从 Seq2Seq 模型到 Transformer 的演进
- Encoder-Decoder 架构详解
- Scaled Dot-Product Attention 和 Self-Attention 原理
- Multi-Head Attention 的实现
- Position-wise Feed-Forward Networks
- Positional Encoding（位置编码）
- 梯度消失/爆炸问题及 Transformer 的解决方案

---

# A deeper look into transformers

# This chapter covers

Sequence modeling before transformers   
Core components of a transformer model   
Attention mechanism and its variants   
How transformers can help stabilize gradient propagation

Now that we have a high-level understanding of the transformer model’s benets and that one of the primary goals for developing the transformer architecture was to reduce the sequential computation from previously used networks like LSTMs, we can delve deeper. This chapter aims to demystify the seemingly complex and daunting transformer model by breaking it down into its foundational elements. We’ll dissect the model, highlighting simple concepts such as matrix multiplication operations, which are pivotal to the attention mechanism and its variant, the multi-head attention. As we journey through the architecture of the transformer, we’ll unravel the importance of its building blocks, including feedforward networks and positional encoding, and their contribution to its eectiveness in natural language processing tasks. But before we delve deeper into transformers, let’s rst briey take a closer look at LSTMs and sequence-to-sequence models to better understand the unique advancements and benets of transformers in more detail.

RNN-based and sequence-to-sequence (seq-2-seq) models have been widely employed in NLP tasks, such as machine translation. These models have the ability to take in variable length sequences as input and produce variable length sequences as output. To illustrate this, let us use our sentence from chapter 1 : "I do not understand French", which we want to translate into: "Je ne comprends pas le français". We can see that our input sequence is shorter (5 words) than our output sequence (6 words). This examples makes it clear that we need to have a model architecture which can handle variable length input and output sequences. Let us examine in detail how these models are constructed and where the transformer comes in. Note: all code examples of this chapter can be found in the book’s repository, https://github.com/Nicolepcx/Transformers-in-Action, in folder CH02.

# 2.1 From seq-2-seq models to transformers

As you are aware, from chapter 1, the structure of most competitive neural sequence transduction models is encoder-decoder based. The RNN encoder-decoder architecture is used in machine translation tasks to read the source language sentence and construct a xed-length representation, z, of it, which is then passed to the decoder, as shown in gure 2.1.

![](images/770ac996ba3aa1876d580514438ad67284621dba2076e3bd923889b00cbbe5df.jpg)  
Figure 2.1 Illustration of Sequence to Sequence learning.

The decoder constructs the target language sentence by predicting one word at a time in an autoregressive mode, based on the previous words and the encoder’s xed-length representation. Sequence-to-sequence modeling is a process central to tasks where context capture is vital. The key advantage of this approach is its ability to comprehend the contextual semantics of a sentence, a crucial aspect in translation tasks. For example, the word "date" may refer to both a fruit and a social engagement, and I’m sure you don’t want these two to be mixed up in a translation.

# 2.1.1 The difficulty of training RNNs

In training RNN-based encoder and decoder architectures, we face signicant challenges. These issues can make improvements in BLEU scores dicult, when using LSTM-based encoder-decoder models. This holds true, especially when the model must translate between languages with vastly dierent syntax or vocabulary. To give you some more background on the BLEU score, it is a metric used in machine translation to quantitatively evaluate the quality of translated text by comparing it to a set of reference translations; a higher score indicates a closer match to the reference.

These challenges in training RNNs, particularly deep RNNs, are thoroughly analyzed in the paper "On the diculty of training Recurrent Neural Networks". As explained by the authors of the paper, diculties often arise due to the propagation of errors over extended

time sequences. To mitigate these issues, they suggest careful initialization and the use of non-saturating activation functions that avoid getting stuck in specic ranges. This allows for ecient gradient ow during backpropagation and batch normalization, contributing to improved network stability. Within LSTMs, "stability" refers to the model’s capability to accurately and reliably learn and represent data patterns and relationships, without becoming overly sensitive to minor changes or noise in the input.

Nevertheless, it is often necessary to employ a deep neural network design, so that the model is capable of modeling the data’s complex patterns and relationships and to eectively capture long-term dependencies in sequential data. This is usually accomplished, in the case of LSTMs, by stacking multiple recurrent layers on top of one another, enabling the network to learn increasingly complex representations of the input over time. However, as the number of layers in the network grows, it becomes more dicult to successfully propagate error signals through the network during training, which, in turn, results in the vanishing gradient problem and other stability issues. In the following section, section 2.1.2, we will look more closely at the vanishing gradient problem.

Hence, even though RNNs like LSTMs are designed to capture long-term dependencies in sequential data by employing a memory cell that can selectively remember or forget information from earlier time steps, they can still struggle to learn these dependencies eectively. One solution to this problem is the introduction of attention mechanisms for LSTMs. These mechanisms enable the models to account for long-term dependencies; however, the attention mechanism itself is dierent and less eective than the one used in transformers. Moreover, in attention-enhanced LSTMs, the input is still fed into the network sequentially, which makes training and inference slow.

In contrast, transformers employ an attention architecture that allows for more ecient processing of long-range dependencies, as shown in gure 2.2. Another advantage of the transformer architecture is the use of multi-head attention, which enables the model to capture dierent aspects of the input data in parallel, further enhancing its ability to process long-range dependencies eectively.

![](images/db116bcd4f1c8c790c434e296f048287e6a277f00fbbe0052b40ce6355486952.jpg)

![](images/172a2094047c01a45b7d8b508280fe405d3ea194fefaae8f70139dd9739bf07f.jpg)  
Figure 2.2 Comparing RNN (left), and self-attention (right) architectures. As illustrated, transformers, with their use of self-attention, can directly attend to all positions in the sequence, regardless of their position in time. Dashed lines represent the flow of information or dependencies. The variables $x _ { 1 }$ to $x _ { 5 }$ represent data input, and the circles above represent the state of the network at each timestep for the RNN. In contrast, for the self-attention mechanism, the circles above denote how each state considers information from all other states, not just the immediate predecessor.

As gure 2.2 demonstrates, transformers provide an alternative method for capturing long-term relationships in sequential data that does not require sequential processing or deep layer stacking.

Even with the rise of transformers, which have shown remarkable eectiveness in many tasks, it’s important to acknowledge the ongoing relevance of RNNs and their variants like LSTMs. These models are still useful in a variety of applications where specic sequential data characteristics and temporal dynamics are at play, such as in certain time-series predictions. As we explore the details of transformer models, understanding the strengths and limitations of RNNs illuminates why the newer techniques represent a signicant leap in the eld of deep learning.

# 2.1.2 Vanishing gradients: transformer to the rescue

The vanishing gradient problem, in particular, can make it dicult to propagate errors through the network and update the model parameters during backpropagation. This can make learning long-term dependencies and correctly modeling sequential data dicult, which is required for many sequence-to-sequence transduction tasks. And like I just pointed out in the previous section, RNN-based models suer from the vanishing gradient problem (gure 2.3).

![](images/75cb109748b05748d3da3ecc0f79b70dbdd9e0a33a7739b791848c4bec251338.jpg)  
Figure 2.3 Simplified example of the vanishing gradient problem, where the gradient contribution from earlier steps (left) become insignificant.

The vanishing gradient problem is a common problem in neural networks, referring to the problem that the gradients get very small, and, therefore, the weights of the network do not update eectively, leading to slow training and poor performance.

This is where the transformer architecture comes in, oering a solution to this problem by using the attention mechanism to capture dependencies between all positions in an input sequence, as shown in gure 2.2. Figure 2.4 compares the gradients for an LSTM and a transformer architecture based on the sentence: "The quick brown fox jumps over the lazy dog and runs through the elds, while the lazy dog barks loudly and then chases the quick brown fox through the forest", to illustrate how the gradients in an LSTM become very small when the sentence is lengthy. I invite you to run the code provided in the books repository to see how various sentence lengths aect the gradients of the two dierent architectures.

```csv
Gradients for RNN model:  
tensor([-1.1910e-08, -5.4378e-09, -1.7932e-08, -1.0601e-07, 4.1367e-08, -9.4355e-08, 3.2711e-07, 2.2510e-08, -2.8239e-07, -1.7337e-06, -3.7372e-06, -3.9611e-07, 3.1367e-06, -2.0029e-05, -1.4689e-05, 3.0543e-05, 6.7980e-05, 9.9472e-05, 4.7608e-06, -4.2451e-04, -6.8603e-06, 1.0076e-03, 6.5036e-04, 3.5185e-03, 8.8883e-03, 4.6364e-03, 5.6579e-03, 1.5272e-03, -1.9715e-02, -3.8232e-01]) 
```

```txt
Gradients for self-attention model:  
tensor([-0.0010, -0.0011, -0.0007, -0.0013, -0.0019, -0.0011, -0.0010, -0.0014, -0.0017, -0.0009, -0.0010, -0.0010, -0.0012, -0.0012, -0.0014, -0.0017, -0.0016, -0.0015, -0.0009, -0.0009, -0.0014, -0.0010, -0.0011, -0.0007, -0.0013, -0.0010, -0.0010, 0.0032])
```

Figure 2.4 Comparison of gradients for an LSTM and transformer architecture based on the sentence: ”The quick brown fox jumps over the lazy dog and runs through the fields, while the lazy dog barks loudly and then chases the quick brown fox through the forest”.

Now that we understand that the attention mechanism in the transformer helps to alleviate the vanishing gradient problem, let us look at its overall architecture in the following section. We’ll start by looking at the transformer’s two main components, the encoder and the decoder, and how they work together to transform an input sequence into an output sequence. Then we look more closely at the transformer architecture’s key innovation, the self-attention mechanism, and nally how positional encoding works.

Before concluding this section, it’s important to highlight the other side of the gradient problem, the exploding gradient problem. This is a situation where the gradients during backpropagation become excessively large, causing model weights to update too rapidly and potentially leading to unstable learning and poor performance. This issue often arises when dealing with long sequences or deep architectures, where the compounded eects of the calculations can cause the gradient values to explode. Similar to the vanishing gradients problem, the exploding gradient problem can also interfere with the model’s ability to learn long-term dependencies, but for the opposite reason. In section 2.3, after we’ve learned more about the transformer’s architecture, we’ll explore this problem in more depth and see how the modular design of transformers mitigates some of the most common problems that arise during training. This will provide us with a better understanding of how the transformer architecture addresses these issues, improving both network stability and performance.

# 2.2 Model architecture

This section distills the essence of the original transformer model as presented in the groundbreaking paper, "Attention Is All You Need". By using simplied terms and concepts, I aim to make this innovative architecture more accessible and easier to grasp. My focus is on the core components that make the transformer what it is - the encoder, the decoder, and the unique attention mechanism that underpins both.

By the end of this section, you should have gained a comprehensive and intuitive understanding of the transformer’s architecture. Therefore, this sections is not an academic exercise; it is your entry ticket into the fascinating world of transformer models. It equips

you with the knowledge necessary for exploring the many transformer variants discussed in the subsequent chapters.

With this understanding, you can join the ongoing conversation around transformerbased models and their applications in the eld of articial intelligence. The discussion surrounding these models is vibrant, growing, and fundamental to the future of AI.

It’s crucial to note that the transformer model, despite being a radical departure from traditional RNN, still adheres to the encoder-decoder framework at its core. This adherence is a testament to the robustness of the encoder-decoder paradigm, which continues to serve as a solid foundation for cutting-edge models.

However, the transformer model achieves its unique capabilities by deploying stacked attention and point-wise, fully connected layers for both the encoder and decoder. These architectural choices, as shown in the left and right portions of gure 2.5, result in a highly exible and scalable model that excels in a wide array of sequence-to-sequence prediction tasks. This scalability and exibility underline the true power of the transformer architecture, making it a cornerstone in the rapidly evolving landscape of articial intelligence.

![](images/46bf07b496bc713a524d49d93980c19e0909034196d5a242fce9f48b7cb628e1.jpg)  
Figure 2.5 The encoder of the transformer architecture is depicted on the left side of the figure, and the decoder is depicted on the right side.

As we have now a visual representation of the transformer architecture, let us look more deeply into each component. We will begin by discussing the encoder and its particular structure and functions before moving on to other important aspects of the transformer model.

# 2.2.1 Encoder and decoder stacks

The transformers model structure is split into an encoder and a decoder part, as shown in gure 2.5. Let us start with the encoder before moving on to other important parts of the model architecture.

# ENCODER PART OF THE TRANSFORMER

The encoder plays a vital role in processing the input sequence in the transformer architecture. It consists of a stack of layers, with each layer having two primary sub-layers: the multi-head self-attention mechanism and the fully connected feed-forward network. The multi-head self-attention mechanism processes the input sequence by determining the signicance or "attention" that should be allocated to dierent parts of the text. Following this, the feed-forward network applies a transformation to the processed sequence from the attention mechanism.

A notable feature in the encoder’s design is the use of residual connections. Instead of merely passing the output of one sub-layer to the next, the encoder also merges this output with its original input. This procedure is akin to adding information from the original text sequence to the processed sequence at every step. By doing this, the transformer ensures that the initial context of the input sequence is preserved and integrated throughout the encoding process. This mechanism helps maintain continuity in the processed sequence, ensuring that the model does not lose the inherent meaning and relationships present in the original text.

Let us delve deeper into the encoder’s architecture. The encoder is composed of $N = 6$ identical layers. Each of these layers contains the aforementioned sub-layers: the multihead self-attention mechanism and a position-wise fully connected feed-forward network, which we’ll look at in detail in this section.

Building upon the encoder’s integration of residual connections, these connections facilitate a direct ow of information from the input through dierent parts of the network. This ow aids in circumventing some transformations, ensuring that the model retains essential input details. Practically, each sub-layer’s output is formulated as LayerN orm $( x +$ Subl ayer $( x )$ ), with Subl ayer (x) representing the function executed by the sub-layer. Additionally, to enhance model robustness and curtail overtting, dropout is applied prior to nalizing these connections.

In a programming sense, this concept is illustrated in listing 2.1, which presents a simplied implementation of the encoder layer.

# Listing 2.1 Simplified encoder layer example

class EncoderLayer(nn.Module): def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):

```python
super().__init__(1)  
1 multi-head attention  
self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)  
self/feed_forward = nn.Sequential( ② feed-forward network nn.Linear(d_model, 2 * dim_feedforward), GLU(input_size=dim_feedforward, output_size=d_model), nnDDDropoutdropout)  
)  
self(norm1 = LayerNorm(d_model) ③ layer normalization (separate class)  
self(norm2 = LayerNorm(d_model)  
self.dropout1 = nnDDDropout Dropout) ④ dropout  
self.dropout2 = nnDDDropout Dropout)  
def forward(self, x, mask=None  
# self-attention layer ⑤ residual connection  
attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)  
x = x + self.dropout1(attn_output)  
x = self(norm1(x)  
# feed-forward layer ⑥ residual connection  
ff_output = self/feed_forward(x)  
x = x + self.dropout2(ff_output)  
x = self(norm2(x)  
return x 
```

All the sub-layers and the embedding layers in the model produce outputs of the same size, $d _ { m o d e l } = 5 1 2$ . This uniformity facilitates the seamless use of residual connections, enabling the information from the input to ow directly to the model’s output, bypassing individual layers. By adding the original input to the output of a network layer, the resulting output equals the sum of the initial input and the layer’s output. This enables the model to learn more eectively and provides a more stable training process, leading to improved performance.

# DECODER PART OF THE TRANSFORMER

As with the encoder, the decoder is another crucial component in the transformer architecture. But this part of the architecture is now responsible for generating the output sequence based on the information processed by the encoder. Each layer in the decoder has primary components similar to the encoder: the self-attention mechanism and the feed-forward network. However, the decoder also incorporates a third component: a multi-head attention mechanism that operates over the output of the encoder’s last layer. This added component allows the decoder to utilize information from the encoder, enabling it to focus on dierent parts of the input text while generating the output.

Residual connections, like in the encoder, are present in the decoder. These connections facilitate the merging of each sub-layer’s output with its input, ensuring that contextual information is preserved throughout the decoding process. Layer normalization further accompanies these residual connections, enhancing the stability of the model and assisting in training.

A distinguishing feature of the decoder is its masked self-attention mechanism. This masking ensures that while generating an output for a particular position in the sequence, the model is restricted to using only previously known outputs, thereby maintaining the order of sequence generation and ensuring causality in the model’s predictions.

The decoder comprises $N = 6$ identical layers. Unlike the encoder, which has two sub-layers, the decoder introduces a third sub-layer to attend over the encoder’s output. This unique attention mechanism enriches the decoder’s output by providing it with a broader context from the input sequence.

Figures 2.6 and 2.7 provide visual representations of the attention masking and the distinct components of the decoder, respectively.

![](images/0c6d677a5a9897a165ab6901018250f0b4b51d20f4c3221ff4b8f5a285a4356f.jpg)  
Figure 2.6 Illustration of multi-head attention masking.

![](images/24758da2c45916f1133b3aee2f1a9c3aefc7faf359e49a7183c1c66b9d27c4ed.jpg)  
Figure 2.7 Detailed structure of the decoder components.

Masking is fundamental for sequence-to-sequence tasks, where the model generates outputs token-by-token based on preceding tokens in the sequence. Dierent types of masking, such as padding masking or sequence masking, can be applied depending on the specic application or use case.

# 2.2.2 Attention

A pivotal element that sets transformer-based models apart is the attention mechanism, particularly its self-attention variant. This component has been instrumental in propelling advancements in natural language processing tasks. In this section, we will delve into the intricacies of self-attention and multi-head attention, aiming to demystify these concepts and demonstrate their role in the transformer architecture’s capability.

The term self-attention refers to the fact that the attention weights are computed within a single sequence. When an input sequence is passed through a multi-head self-attention layer, the attention weights are computed between dierent positions within the same sequence. This mechanism, named "self-attention", allows each element in the input sequence to relate to every other element, including itself, in the sequence. Therefore, the output generated is a weighted representation of the entire sequence.

# SCALED DOT-PRODUCT BASICS

If you are now wondering why I mentioned self-attention in the intro of this section selfattention and not scaled dot-product attention, let me rst clarify. Self-attention and scaled dot-product attention are two related concepts that are used in the transformer architecture to allow for ecient and eective learning of relationships between elements within a sequence.

To understand both, let us rst look at scaled dot-product attention by reducing it to its simplest components. To best understand this concept, we’ll start with a graphical illustration. Let’s consider a scenario where we have a sequence of ve inputs and ve outputs, as shown in gure 2.9. This visual representation will allow us to explore the mechanics of scaled dot-product attention and understand how it shapes the interactions within the transformer.

![](images/e997e83d451eb1ee727c04e990161dfa8c8735b3d53378b38dd72329de233658.jpg)  
Figure 2.8 Illustration of a sequence of five inputs and five outputs.

To compute $y _ { 3 }$ , we use the vector $x _ { 3 }$ to determine the associated weights. This is achieved by calculating the dot product of $x _ { 3 }$ with each vector in the sequence, starting from $x _ { 1 }$ and continuing through to $x _ { 5 }$ . As shown in gure 2.9.

![](images/f6ac2b27109d468d2e93a172b79fbbb16d52bfa0762a65d0397aa463d17e6bd3.jpg)  
Figure 2.9 Graphical explanation of taking the dot product to compute the weights for y3.

After computing these ve weights, we take the softmax so that it sums up to one. Then we multiply each input vector by the weights we just computed and sum them all up and this gives us then the vector y3. This process is shown in gure 2.10.

![](images/50741d8331e03d1456085fb7b93019b0d8f3b0de6ca07e1f06365a813bde89b1.jpg)  
Figure 2.10 To compute $y _ { 3 }$ , we take first the softmax function for all weights, so it sums up to 1 and then we multply each input vector by the weights and sum it up.

Moreover, we process every input vector $x _ { i }$ in three ways to fulll the three roles with matrix multiplication.

Compare to every other vector to compute attention weights for its own output $y _ { i }$ , which is the query.   
Compare to every other vector to compute attention weight $w _ { i j }$ for output , which $y _ { i }$ is the key.

□ Sum up with the other vectors to form the result of the attention weighted sum, which is the value.

To understand this in an intuitive way, using our text example from chapter 1, ”the movie was not bad”, this just means:

The query represents what we’re ”asking” about or looking for. It’s like a ”search term.” In our example, if we’re interested in understanding the importance relationship between ”not,” ”bad,” and ”movie,” we can imagine them as our queries.   
The key represents the ”features” or ”identifiers” for every word in the sentence. The keys determine how well each word in the sentence responds to the query. If our query is ”not,” then the keys for every word will determine how related or relevant each word in the sentence is to ”not.”   
The value contains the ”content” we want to retrieve or weigh. Once we’ve determined how relevant each word (via its key) is to the query, the values give us the actual content we’d retrieve or weigh. In many implementations, the initial value representations are just the input embeddings, but as layers of attention stack, they capture more nuanced contextual representations.

So, in our example with ”the movie was not bad”: If ”not” is the query, the attention mechanism might assign high importance (or weight) to ”bad” because ”not bad” is a common phrase. The values corresponding to both ”not” and ”bad” would then be summed up with their respective weights to produce the final output for the word ”not.”

However, in actual models, all words in a sequence simultaneously act as queries, keys, and values. The self-attention mechanism computes a weighted sum of values for each word in the sequence, based on the attention scores between its query representation and all key representations in the sequence. This allows every word to gather information from all other words, based on their relevancy, to produce new contextual embeddings.

Now, mathematically this is represented as follows:

Query: $\mathbf { q } _ { i } = \mathbf { W } _ { \mathrm { q } } \pmb { x } _ { i }$   
Key: $\mathbf { k } _ { i } = \mathbf { W } _ { \mathrm { k } } \pmb { x } _ { i }$   
Value: $\mathbf { v } _ { i } = \mathbf { W } _ { \mathrm { v } } \pmb { x } _ { i }$

Therefore, mathematically speaking, we can think of the processes as simple matrix multiplication. We then introduce a softmax function to make sure it sums to one, as shown in equation 2.1. So, to summarize all you have to do is to multiply each input vector by these three matrices and then combine the results in the aforementioned three dierent ways to produce the output $y$ of $i$ .

$$
w _ {i j} ^ {\prime} = \mathbf {q} _ {i} ^ {\top} k _ {j}
$$

$$
w _ {i j} = \operatorname {s o f t m a x} \left(w _ {i j} ^ {\prime}\right) _ {\square} \tag {2.1}
$$

$$
\mathbf {y} _ {i} = \sum_ {j} w _ {i j} v _ {j}
$$

From this explanation, it’s clear that the heart of the attention mechanism, or attention function, depends on matrix multiplication. However, it’s important to understand that it’s the specic combination of these matrix operations - the generation of queries, keys, and values - and their application through the dot product and softmax functions, that enables the model to capture complex relationships within sequences. This gives the transformer model its ability to perform remarkably in NLP tasks.

# SCALED DOT-PRODUCT ATTENTION

With a foundational understanding of the basics - query, key, and value matrices - of dot-product attention, we can now examine the more intricate aspect of this attention mechanism, the scaled dot-product attention. Figure 2.11 provides a graphical representation of the overall design of scaled dot-product attention.

![](images/cb60d86871aeb93dbfe2fe3b8aeaa886cd10b664613f7a43a70bd76f382551da.jpg)  
Figure 2.11 Illustration of scaled dot-product attention.

Scaled dot-product attention, a specic type of self-attention mechanism utilized in the transformer architecture, gets its name from the way attention weights are computed. In this mechanism, the dot product of every pair of query and key vectors is calculated and divided by the square root of the dimension of the key vectors, as demonstrated in equation 2.2. This normalization occurs before the application of the softmax function.

$$
\operatorname {A t t e n t i o n} (Q, K, V) = \operatorname {s o f t m a x} \left(\frac {Q K ^ {T}}{\sqrt {d _ {k}}}\right) V \tag {2.2}
$$

The importance of scaled dot-product attention lies in its ability to stabilize gradients during the backpropagation phase of training. By limiting the size of the attention scores via scaling the dot product by $\sqrt { d _ { k } }$ , softmax saturation can be prevented, and gradient explosion can be avoided. The eect of such scaling is illustrated in Figure 2.12.

![](images/2690d86efa53952be66bb9836814291f0cf2947ebd17068bf7312db2b0f1c465.jpg)  
Figure 2.12 This figure shows that the variance of the scaled dot product values is smaller than the variance of the unscaled dot product values, indicating that scaling helps to control the variance of the dot product values. This, in turn, helps prevent the gradients from exploding during backpropagation, improving the stability and convergence of the training process.

Hence, if the dot product values become excessively large, the resulting gradients may also become disproportionately large, resulting in unstable training due to the notorious exploding gradient problem. By controlling the variance of the dot product values through scaling, the magnitude of the gradients during backpropagation is inuenced. The scaling factor used in the transformer architecture is the square root of the key dimension (i.e.,√ $\sqrt { 5 1 2 }$ for a key dimension of 512), which has been found to yield good results in practice.

To illustrate the scaled dot-attention concept more clearly, I have prepared a simplied version of it in listing 2.2. For simplicity, we assume we have already learned the weight matrices during training, and we just assign the word embeddings. But in practice, these word embeddings would have been created by the encoder.

# Listing 2.2 Simplified scaled dot-attention implementation

0 Defining word embeddings

embed_1 = np.array([0, 1, 0])

embed_2 $=$ np.array([1, 0, 1])

embed_3 $=$ np.array([0, 1, 1])

embed_4 = np.array([1, 1, 0])

2 stacking the word embeddings into a single array

embeddings $=$ np.array([embed_1, embed_2, embed_3, embed_4])

Initialize weight matrices

$\widetilde { W } \mathbf { q } \ =$ rand(3, 3)

$\begin{array} { r l } { \mathbb { W } \mathbb { k } } & { { } = } \end{array}$ rand(3, 3)

```txt
Wv = rand(3, 3) 
```

$\textcircled{9}$ compute queries, keys and values

${ \textsc { Q } } =$ embeddings.dot(Wq)

$\mathrm { ~ \texttt ~ { ~ K ~ } ~ } =$ embeddings.dot(Wk)

${ \begin{array} { r l } { \nabla } & { = } \end{array} }$ embeddings.dot(Wv)

$\oplus$ Compute scaled attention scores

attention_scores $=$ softmax(Q.dot(K.T) / sqrt(K.shape[1]), axis $^ { = 1 }$ )

$\bullet$ Compute weighted sum of values

attention_output $=$ attention_scores.dot(V)

To recap, this scaling is crucial for model stability and eectiveness, particularly when the key vectors’ dimensions are large. In such scenarios, the dot product can grow large in magnitude, leading to very large pre-softmax values. This can lead to two potential problems: saturation of the softmax function, where very large inputs are mapped to the endpoint of the function, resulting in a loss of the original values’ information, and large gradients during the backpropagation phase, leading to instability in learning, known as the exploding gradient problem.

By reducing the dot product’s magnitude by the square root of the dimensionality, the magnitude of the values entering the softmax function is eectively controlled, mitigating these issues. Therefore, scaled dot-product attention not only aids in stabilizing the gradients during the training phase but also prevents saturation of the softmax function, preserving the original relationships between the inputs.

# SELF ATTENTION

Self-attention is a type of attention mechanism that allows a sequence-to-sequence model to focus on dierent parts of the input sequence when generating an output sequence. Figure 2.13 shows the computed weights of a self-attention matrix using our sentence from chapter 1, "the movie was not bad".

![](images/78348c4b0ca780735ae80603273f51b3d3c47789b421ac0aa70b3682fbe9e115.jpg)  
Figure 2.13 Weight distribution of self-attention.

Self-attention operates by allowing a sequence-to-sequence model to weigh dierent portions of an input sequence dierently when producing an output sequence. Instead of applying distinct rules or weights to each individual element in the sequence as in other attention mechanisms, self-attention evaluates the importance of each element based on its relation to every other element in the sequence.

Specically, in the transformer architecture, each element (or word) in the input sequence can attend to all positions in the sequence, enabling the model to determine which positions are crucial for a given context. This is achieved using the same set of parameters, making the process consistent across dierent positions. Thus, self-attention provides the model with the capability to focus on segments of the input sequence that are most relevant to the current processing step, ensuring that relevant contextual information is retained and emphasized.

The self-attention mechanism in the transformer architecture uses a singular matrix to calculate the queries, keys, and values. As explained in the original paper "Attention is All You Need" (1), this singular matrix is obtained by concatenating the weight matrices of the linear transformations applied to the queries, keys, and values. The resulting matrix is then divided into multiple heads, with the attention mechanism applied to each. Concatenating the attention results from each head and passing them through a linear layer yields the output.

Furthermore, self-attention is more numerically eective because it eliminates the need to compute a distinct matrix for each point in the input sequence, as in other mechanisms. Second, in addition to its numerical eciency, self-attention also provides greater freedom to model long-term relationships by allowing each part in the input sequence to respond to any other position. This valuable attention mechanism will be further explored in the next section, where we will discuss its relevance to the network’s stability.

# MULTI-HEAD ATTENTION

Instead of just using one attention function, the authors of the original transformer paper, "Attention Is All You Need", introduced multi-head attention to project queries, keys, and values $h$ -times with dierent learned linear projections into:

$$
d _ {k} = d _ {v} = d _ {\text {m o d e l}} / h = 6 4, \tag {2.3}
$$

where $h = 8$ , and $d _ { k }$ refers to the dimension of the keys and $d _ { v }$ refers to the dimension of the values and $d _ { \mathrm { m o d e l } }$ to the model’s dimension.

These values are then combined and projected again to obtain the nal values. This approach is called multi-head attention because it allows the model to look at information from dierent representations or "views" (referred as "subspaces" in the original transformer paper) of the input simultaneously.

One of the key reasons for adopting multi-head attention is that the dierent heads can learn to recognize dierent types of relationships in the data. Each head could potentially focus on a dierent type of interaction, for example, syntactic versus semantic, or shortterm versus long-term dependencies. This diversied perspective enables the model to

capture a richer set of information compared to a single head, which would have a limited, averaged view of the input.

![](images/885fb5ccabb74999ce5453a417ba4fb45899a310ec5751aed3d55b3442cd2251.jpg)  
Figure 2.14 Illustration of multi-head attention.

Figure 2.14 and equation 2.4 make it clear that, with multi-head attention, we are splitting the queries, keys, and values into multiple heads and computing their attention separately.

$$
\operatorname {M u l t i H e a d} (Q, K, V) = \operatorname {C o n c a t} \left(\operatorname {h e a d} _ {1}, \dots , \operatorname {h e a d} _ {\mathrm {h}}\right) W ^ {O}
$$

$$
\text {w h e r e} = \text {A t t e n t i o n} \left(Q W _ {i} ^ {Q}, K W _ {i} ^ {K}, V W _ {i} ^ {V}\right) \tag {2.4}
$$

Where the projections are parameter matrices $W _ { i } ^ { Q } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { k } } , W _ { i } ^ { K } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { k } } , W _ { i } ^ { V } \in$ $W _ { i } ^ { Q } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { k } }$ $\mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { v } }$ and $W ^ { O } \in \mathbb R ^ { h d _ { v } \times d _ { \mathrm { m o d e l } } }$

Let us transfer this to actual code, to make this concept more explicit. For this, we can use the functional API from PyTorch, as shown in listing 2.3

# Listing 2.3 Simplified multi-head attention implementation

```python
class MultiHeadAttention(nnModule): def __init__(self, d_model, num_heads): super(MultiHeadAttention, self).__init__(assert d_model % num_heads == 0 self.d_model = d_model self.num_heads = num_heads self.d_k = d_model // num_heads self.W_q = nn.Linear(d_model, d_model) self.W_k = nn.Linear(d_model, d_model) self.W_v = nn.Linear(d_model, d_model) self.W_o = nn.Linear(d_model, d_model) def forward(self, query, key, value, mask=None): batch_size = query.size(0) query = self.W_q(query).view(batch_size, -1, 
```

```prolog
self.num_heads, self.d_k).transpose(1, 2)  
key = self.W_k(key).view(batch_size, -1,  
self(num_heads, self.d_k).transpose(1, 2)  
value = self.W_v(value).view(batch_size, -1,  
self(num_heads, self.d_v).transpose(1, 2)  
scores = torch/matmul(query, key.transpose(-2, -1)) /  
math.sqrt(self.d_k)  
if mask is not None:  
    scores = scoresmasked_fill(mask == 0, -1e9)  
attention = F softmax(scores, dim=-1)  
output = torch/matmul(attention, value).transpose(1,  
2).contiguous().view(batch_size, -1, self.d_model)  
output = self.W_o(output)  
return output 
```

To explain the code in more detail, rst we dene a MultiHeadAttention class, which is a subclass of nn.Module, PyTorch’s base class for all neural network modules. This class is initialized with the dimensions of the model and the number of heads. The assert statement ensures that the model’s dimension is evenly divisible by the number of heads.

In the __init__ method, we dene the linear transformation matrices for the queries, keys, values, and output. Each of these matrices is implemented as a fully connected (nn.Linear) layer, which performs a linear transformation of the input data.

The forward method is where the actual computation happens. This method accepts the query, key, and value matrices as input (and optionally a mask), and returns the output of the multi-head attention mechanism.

First, the query, key, and value matrices are independently transformed using the respective weight matrices (self.W_q, self.W_k, self.W_v). They are then reshaped and transposed to have the shape (batch_size, num_heads, sequence_length, depth), to accommodate for the multiple heads.

Next, the scaled dot-product attention is computed. The attention scores are calculated by taking the dot product of the query and key matrices, then dividing by the square root of the depth of the key to scale the scores. If a mask is provided, it is applied to the scores. The scores are then passed through a softmax function to obtain the attention weights.

The output is computed by taking the dot product of the attention weights and the value matrix. This output is then reshaped and passed through the output weight matrix $( s e l f . W _ { - } o )$ ).

The implementation also includes a usage example, which illustrates how to instantiate the MultiHeadAttention class and use it to compute the multi-head attention of some random input data as shown in listing 2.4

Listing 2.4 Simplified multi-head attention implementation   
Assuming we have some data   
batch_size $= 32$ sequence_length $= 100$ d_model $= 512$ Embedding dimension   
num_heads $= 8$ InstANTIate the model   
multi_head_attn $=$ MultiHeadAttention(d_model, num_heads)   
Create some random data for input   
input_data $=$ torch RAND(batch_size, sequence_length, d_model)   
We use the same input for query, key and value for self-attention   
output $=$ multi_head_attn(input_data, input_data, input_data)

If we would print the shape of the output, we would get 32, 100, 512, that is the batch_size, sequence_length and d_model, respectively. This result demonstrates that despite the seemingly complex processing of information during multi-head attention, the output retains the original sequence structure and the model’s dimensionality.

To summarize this section, with multi-head attention, the model can attend to dierent aspects of the input and process them in parallel, thus enabling it to capture more complex relationships in the data. When we compare this with using a single attention head, a single attention head would average the information and would not be able to take advantage of the various elements in the input. This would restrict the model’s ability to understand and reect complicated patterns in the data. Therefore, multi-head attention is an essential component of the transformer architecture, allowing the model to accomplish cutting-edge success in a broad variety of NLP tasks.

# 2.2.3 Position-wise feed-forward networks

FFNs (position-wise feed-forward networks) are a type of neural network frequently employed in NLP tasks, specically designed to transform xed-length vectors. This transformation process is useful because it can convert input data (like words or sentences represented as xed-length vectors) into more abstract representations. These abstract representations can capture complex patterns in the input data, like the semantics of a sentence or the context of a word. This conversion is achieved through two fully connected layers separated by a nonlinear activation function.

Each element of the vector is individually fed through the network and transformed to a higher dimensional space. This transformation, aimed at increasing capacity, allows for more intricate interactions between the vector’s components. The output is then reverted back to its initial dimension using another fully connected layer, ensuring the output maintains a consistent shape with the original input, which is benecial when stacking multiple layers or components together in a neural network.

In the context of machine learning, a fully connected layer is a type of neural network layer in which each neuron is connected to every neuron in the previous layer, as illus-

trated in gure 2.15. By transforming input vectors in this manner, FFNs are able to extract and leverage higher-level features from the input data, thus improving the model’s understanding of the underlying patterns in the data.

![](images/fb0f676640c5c054e65f7632e00f823a2b10dde4f9de8f70e3f2fa2827c62165.jpg)  
Figure 2.15 Fully connected neural network with two hidden layer.

An implementation of a position-wise FFNs is illustrated in listing 2.5.

# Listing 2.5 Simple position-wise feed-forward network

class PositionwiseFeedforward(nnModule): def __init__(self, input_dim, hidden_dim, dropout=0.1): super(PositionwiseFeedforward, self).__init_(self.output_dim = input_dim self.hidden_dim = hidden_dim self_dropout = nn.Linear Dropoutdropout) self.fc1 = nn.Linear(input_dim, hidden_dim) self.fc2 = nn.Linear(hidden_dim, input_dim) def forward(self, x): $\mathbf{x} =$ self_dropout(torch.ReLU(self.fc1(x))) $\mathbf{x} =$ self.fc2(x) return x

NOTE The given code example presents a basic position-wise feedforward network with a single output layer. However, the complete transformer architecture, including multi-head attention and other components, handles sequences of inputs and outputs, not just single ones.

The word "position-wise" relates to the fact that the same transformation is done independently to each element in the vector, regardless of its location within the sequence. This

method is used because it enables the network to understand more complex patterns in the input and makes training on large datasets simpler. Additionally, position-wise FFNs are numerically eective since each member of the array is handled separately in parallel.

# 2.2.4 Positional encoding

The transformer lacks recurrence, which is a commonly used mechanism to understand the order of tokens in a sequence. Because of this absence, the transformer incorporates positional encoding, a technique for determining the relative or absolute position of tokens within a sequence. Without positional encoding, the transformer wouldn’t be able to dierentiate the order of tokens, leading to potential misinterpretations in the sequence. Recognizing the position of each token in the sequence allows the model to accurately infer relationships and meaning between tokens. To achieve this, the positional encoding is added to the sequence’s input embeddings by encoding each dimension of the position using a sinusoidal function. This encoding enables the model to attend to the relative positions of the words in the input sequence, using a linear function of the position index pos and the dimension index i. The mathematical formula for computing these positional encodings is shown in equation 2.5.

$$
P E _ {(p o s, 2 i)} = \sin \left(p o s / 1 0 0 0 0 ^ {2 i / d _ {\mathrm {m o d e l}}}\right) \tag {2.5}
$$

$$
P E _ {(p o s, 2 i + 1)} = \cos \left(p o s / 1 0 0 0 0 ^ {2 i / d _ {\mathrm {m o d e l}}}\right)
$$

The model can then use this information to better identify the context and meaning of each token in the sequence by summing the positional encoding with the input embeddings. Listing 2.6 shows an example implementation for positional encoding, using the formula from equation 2.5.

# Listing 2.6 Example implementation for positional encoding

```python
class PositionalEncoding(nnModule):
    "" "Positional encoding class."
    def __init__(self, num_hiddens, max_len=1000):
        super().__init()
        self.dropout = nn.Dropout(0.1)
        # Create a positional embedding matrix
        position = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        div_term = torch.exp(
            torch.arange(0, num_hiddens, 2, dtype=torch.float32) *
            - (math.log(10000.0) / num_hiddens))
        pe = torch.zeros((1, max_len, num_hiddens))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, X):
        X = X + self.pe[:, :X.shape[1], :] to(X_device)
        return self.dropout(X) 
```

The positional encoding and how it adds a sine wave depending on its position is shown in gure 2.16. Note, for each dimension, the wave’s frequency and oset are dierent.

![](images/9e755b03dbaa40de7cf6ad346009f61bbc1ea892207dc6a7feb5324f70c80ff9.jpg)  
Figure 2.16 Positional encoding example

If you want to get a better grasp of positional encoding, I recommend using the PositionalEncoding class, which is also available in the book’s repository, https://github. com/Nicolepcx/Transformers-in-Action, and changing the model’s inputs to see how this changes the plots output.

In this section, we explored the transformer architecture’s encoder and decoder components and their collaborative process in transforming input sequences into output sequences. We delved into the key innovation, the self-attention mechanism, and discussed positionwise feed-forward networks (FFNs) used for processing self-attention outputs. Lastly, we examined the role of positional encoding in conveying input sequence order, providing a comprehensive understanding of the transformer architecture and its contributions to cutting-edge performance in natural language processing tasks.

# 2.3 Building on the basics: a world of possibilities awaits!

With a better understanding of the transformer architecture and the role of self-attention, we can now delve deeper into how this type of architecture overcomes some of the issues that arise in RNNs. By comparing the two architectures more closely, we can gain valuable insights into the strengths and weaknesses of each approach.

# 2.3.1 Methods to stabilize the training of RNNs

In this subsection, we’ll delve into the challenges of training recurrent neural networks and discuss some widely recognized solutions. A notable reference is "On the diculty of training Recurrent Neural Networks", which oers a comprehensive view of these challenges and proposes practical solutions. Additionally, other strategies for increasing RNN stability exist, including equal kernel and recurrent kernel initializers, along with orthogonal matrix initializer for recurrence weights. These methods, as described in various studies, are

designed to alleviate the vanishing and exploding gradient problem in RNNs by carefully initializing the weights. The following are key techniques developed to enhance RNN stability.

Gradient clipping: this method limits the gradient values during backpropagation to prevent them from becoming too large.   
Initialization strategies: careful initialization of the network parameters can also help to prevent the exploding gradient problem.   
Layer normalization: this technique involves normalizing the activations of each layer in the network, which can help improve the stability and convergence of the training process.   
Skip and residual connections: these connections enable the gradient signal to propagate directly through the network without being reduced by activation functions or other transformations. This can prevent gradients from becoming too large or too small, because the network can better propagate the gradient during backpropagation.   
Adaptive learning rate algorithms: optimizers such as Adam, which use adaptive learning rates for each parameter, can help stabilize the learning process and mitigate the vanishing and exploding gradient problem by adjusting the learning rate based on the magnitude of the gradient. Moreover, by using a moving average of past gradient information, Adam reduces oscillations and improves the stability of the learning process.

# 2.3.2 The transformer architecture: a paradigm shift in neural network stability

In contrast to the RNN based networks, the transformer architecture, with its self-attention mechanism and the way the residual connections and layer normalization are set up, is constructed to naturally alleviate the vanishing and exploding gradient problem. Moreover, because of these design choices and the fact that it uses more ecient feed-forward layers, it is less sensitive to a specic initialization of weights and tuning of hyperparameters than RNNs. In the following are the key aspects on which the transformer builds and how self-attention and the overall architecture design help to stabilize the training.

Self-attention: the self-attention mechanism allows the model to directly capture dependencies between all positions in an input sequence, regardless of their position in time. This helps the model to better learn long-term dependencies, which can be dicult for traditional RNNs.   
1 Layer normalization: unlike RNNs that typically use batch normalization, the transformer uses layer normalization, which can help to improve the stability and convergence of the training process and prevent the exploding gradient problem.   
Residual connections: the use of residual connections in the transformer architecture helps to improve the ow of information through the model and prevent the vanishing gradient problem. Note, unlike LSTMs, which add residual connections to the hidden states of each recurrent layer, the transformer adds the input of each sub-layer to

its output using residual connections. This enables the input signal to easily pass through the layer and helps to avoid the vanishing gradient problem, even in deeper architectures. Furthermore, this design supports parallel computation of the sublayers, which can speed up the training process.

【 Positional encoding: the use of positional encoding allows the transformer to take into account the relative positions of elements in an input sequence, which can help to better capture temporal relationships and prevent the vanishing gradient problem.

Overall, the transformer architecture is addresses many of the limitations of traditional RNNs, and these improvements help to make the training process more stable and the models more eective in capturing complex dependencies in sequential data.

But it’s important to remember that hyperparameters like the learning rate and the number of layers can still aect how well the transformer works, which we’ll look at in more detail in chapter 9. However, unlike LSTMs, the transformer does not require a deep network structure to capture long-term dependencies, which further simplies the gradient computation and results in a simpler computational graph of the transformer than for an LSTM. The simpler computational graph of the transformer can make it easier to optimize hyperparameters and tune the model for better performance. To make this more evident, I provide you some classes and functions in the books repository in folder CH02 that let you plot and compare the computational graphs of both networks, the transformer and the LSTM, respectively.

I hope that I could convince you with this chapter that even though transformers look complex and dicult at rst sight, they can be broken down into simple building blocks that use basic mathematical operations. Remember, the attention mechanism, which is the key ingredient of the transformer, is basically a simple matrix multiplication operation. The same concept drives multi-head attention, a variant of the attention mechanism used in transformers. These models are a powerful yet adaptable design for natural language processing and other tasks due to the clever mix of these fundamental matrix multiplication operations, as well as other components such as feed-forward networks and positional encoding. This knowledge is a great foundation for the more advanced variants of the transformer model in this book.

# 2.4 Summary

□ The transformer model is divided into an encoder and decoder. The encoder processes the input sequence into a context or memory, which the decoder then uses to generate the output sequence.   
The pivotal part of the transformer is the attention mechanism. Where the query matrix, in a simplied way, just acts as a way of "retrieving" the key (matrix) with the value (matrix).   
The transformer model focuses on various portions of the input sequence using a self-attention mechanism and a feedforward neural network to transform the attention

outputs. These components are stacked in multiple layers, with residual connections and layer normalization.

Self-attention is computationally ecient because it allows for parallel processing of the input sequence. Unlike RNNs, which require a distinct set of learnable parameters for each point in the sequence, self-attention is exible and eective for various NLP tasks.   
Positional encoding is a way of guiding the transformer to dierentiate the order of tokens (words) within a sequence.   
1 While various methods have been developed to improve RNN stability during training, the transformer architecture, with its self-attention mechanism, layer normalization, and residual connections, naturally mitigates both vanishing and exploding gradient problems, making it easier to train and optimize than RNNs.

# Bibliography

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. CoRR, abs/1706.03762, 2017. URL: http://arxiv.org/abs/1706.03762 25
