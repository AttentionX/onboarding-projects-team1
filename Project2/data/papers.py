retro = '''
Improving language models by retrieving
from trillions of tokens
Sebastian Borgeaud † , Arthur Mensch † , Jordan Hoffmann † , Trevor Cai, Eliza Rutherford, Katie Millican,
George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas,
Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones,
Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero,
Karen Simonyan, Jack W. Rae ‡ , Erich Elsen ‡ and Laurent Sifre †,‡
All authors from DeepMind, † Equal contributions, ‡ Equal senior authorship
Abstract
We enhance auto-regressive language models by conditioning on document chunks retrieved from a
large corpus, based on local similarity with preceding tokens. With a 2 trillion token database, our
Retrieval-Enhanced Transformer (Re t ro) obtains comparable performance to GPT-3 and Jurassic-1
on the Pile, despite using 25× fewer parameters. After fine-tuning, Re t ro performance translates to
downstream knowledge-intensive tasks such as question answering. Re t ro combines a frozen Be rt
retriever, a differentiable encoder and a chunked cross-attention mechanism to predict tokens based on
an order of magnitude more data than what is typically consumed during training. We typically train
Re t ro from scratch, yet can also rapidly Re t rofit pre-trained transformers with retrieval and still
achieve good performance. Our work opens up new avenues for improving language models through
explicit memory at unprecedented scale.
1. Introduction
Language modelling (LM) is an unsupervised task that consists of modelling the probability of text,
Î
usually by factorising it into conditional next-token predictions 𝑝 ( 𝑥 1 , . . . , 𝑥 𝑛 ) = 𝑖 𝑝 ( 𝑥 𝑖 | 𝑥 <𝑖 ). Neural
networks have proven to be powerful language models, first in the form of recurrent architectures
(Graves, 2013; Jozefowicz et al., 2016; Mikolov et al., 2010) and more recently in the form of
Transformers (Vaswani et al., 2017), that use attention to contextualise the past. Large performance
improvements have come from increasing the amount of data, training compute, or model parameters.
Transformers have been scaled from 100 million parameter models in seminal work to over hundred
billion parameters (Brown et al., 2020; Radford et al., 2019) in the last two years which has led to
models that do very well on a wide array of tasks in a zero or few-shot formulation. Increasing model
size predictably improves performance on a wide range of downstream tasks (Kaplan et al., 2020).
The benefits of increasing the number of parameters come from two factors: additional computations
at training and inference time, and increased memorization of the training data.
In this work, we endeavor to decouple these, by exploring efficient means of augmenting language
models with a massive-scale memory without significantly increasing computations. Specifically, we
suggest retrieval from a large text database as a complementary path to scaling language models.
Instead of increasing the size of the model and training on more data, we equip models with the
ability to directly access a large database to perform predictions—a semi-parametric approach. At
a high level, our Retrieval Transformer (Re t ro) model splits the input sequence into chunks and
retrieves text similar to the previous chunk to improve the predictions in the current chunk. Existing
retrieval for language modelling work only considers small transformers (100 millions parameters)
and databases of limited size (up to billions of tokens) (Guu et al., 2020; Khandelwal et al., 2020;
Lewis et al., 2020; Yogatama et al., 2021). To our knowledge, our work is the first to show the benefits
of scaling the retrieval database to trillions of tokens for large parametric language models. Our main
Corresponding authors: {sborgeaud|amensch|jordanhoffmann|sifre}@deepmind.comImproving language models by retrieving from trillions of tokens
172M
425M
1.5B
7.5B
Baseline
RETRO [OFF]
1.0
1.0 0.9 0.9 0.9
0.8 0.8 0.8
0.7 0.7 0.7
1.0
200 400 800 1600
7500
Number of Non-Embedding Params (M)
0 1
10
100 1000 10000
Retrieval dataset (B Tokens)
0 1
RETRO [ON]
3 5 10
30 50 100
Number of neighbors
Figure 1 | Scaling of Re t ro. The performance gain of our retrieval models remains constant with
model scale (left), and is comparable to multiplying the parameteric model size by ∼ 10×. The gain
increases with the size of the retrieval database (middle) and the number of retrieved neighbours
(right) on the C4 validation set, when using up to 40 neighbours. Past this, performance begins to
degrade, perhaps due to the reduced quality. At evaluation Re t ro can be used without retrieval
data (Retro[OFF]), bringing limited performance degradation compared to baseline transformers.
contributions are the following.
• We introduce Re t ro, a retrieval-enhanced autoregressive language model (§2.2). We use a
chunked cross-attention module to incorporate the retrieved text (§2.4), with time complexity
linear in the amount of retrieved data. We show that retrieving based on a pre-trained frozen
Bert model (§2.3) works at scale, removing the need for training and updating a retriever
network.
• We show that our method scales well with model size and database size (Fig. 1): Ret ro
provides a constant gain for models ranging from 150M to 7B parameters, and Re t ro can be
improved at evaluation time by increasing the database size and the number of retrieved neigh-
bours. Our largest model obtains state-of-the-art results on a range of downstream evaluation
datasets including Wikitext103 (Merity et al., 2017) and the Pile (Gao et al., 2020) (§4). We
show that Retro can be fine-tuned to achieve competitive performance on downstream tasks
such as question answering (§4.3).
• We propose an evaluation aware of proximity of test documents with the training set (§2.6),
addressing the problem of test set leakage (Lee et al., 2021). This is relevant for all language
models, and especially for retrieval-enhanced models since they have direct access to the training
dataset during evaluation. Using this methodology, we show that the performance of Ret ro
comes from both explicit neighbour copying and general knowledge extraction (§4.4).
2. Method
We design our retrieval-enhanced architecture to be capable of retrieving from a database with trillions
of tokens. For this purpose, we retrieve at the level of contiguous token chunks instead of individual
tokens which reduces storage and computation requirements by a large linear factor. Our method first
constructs a key-value database, where values store raw chunks of text tokens and keys are frozen
Bert embedddings (Devlin et al., 2019). We use a frozen model to avoid having to periodically
re-compute embeddings over the entire database during training. Each training sequence is then split
into chunks, which are augmented with their 𝑘 -nearest neighbour retrieved from the database. An
encoder-decoder architecture integrates retrieval chunks into the model’s predictions. We summarize
the Retro architecture in Fig. 2, and detail it in this section. We end the section by introducing
2Improving language models by retrieving from trillions of tokens
Encoded neighbours
Retrieval
dataset
Neighbours
BERT
BERT
Encoded neighbours
E 2
Frozen kNN Retriever
E 1
Attend
Condition
Input
tokens
Chunked cross-attention (CCA)
E 1
K
E 2
V
H 1
C 1
EMB
C 2
ATTN
Q
H 1+
H 2
CCA
FFW
CA
CA(H 1+ , E 1 )
READ
H 2+
H 3
C 3
H
Attending chunks
CA
CA(H 2+ , E 2 )
CCA(H, E)
RETRO block (x L)
X
Figure 2 | Ret ro architecture. Left: simplified version where a sequence of length 𝑛 = 12 is split
into 𝑙 = 3 chunks of size 𝑚 = 4. For each chunk, we retrieve 𝑘 = 2 neighbours of 𝑟 = 5 tokens each. The
retrieval pathway is shown on top. Right: Details of the interactions in the Cca operator. Causality is
maintained as neighbours of the first chunk only affect the last token of the first chunk and tokens
from the second chunk.
a new methodology to evaluate language models when an evaluation set is partially present in the
training set.
2.1. Training dataset
We use a multi-lingual version of MassiveText (Rae et al., 2021) for both training and retrieval data.
The dataset consists of text documents from multiple sources and multiple languages totalling over
5 trillion tokens (detailed in Table 1). Sequences are sampled from subsets of the training data,
with sampling weights given in the right-most column of Table 1. We tokenize the dataset using
SentencePiece (Kudo and Richardson, 2018) with a vocabulary of 128,000 tokens. During training
(unless otherwise specified), we retrieve from 600B tokens from the training data. The training
retrieval database is made of the same subsets as the training data, in proportion that matches
the training sampling frequencies. During evaluation the retrieval database consists in the full
union of these datasets, with the exception of books for which we use a sub-sample of 4%. The
evaluation retrieval database thus contains 1.75T tokens. To limit test set leakage, we compute the
13-gram Jaccard similarity between train and test documents using the MinHash scheme and remove
all training documents with high similarity (0.8 or higher) to a validation or test set document.
Additionally, we remove all validation and test articles from Wikitext103 (Merity et al., 2017) from
our Wikipedia training data.
2.2. Retrieval-enhanced autoregressive token models
Our approach uses retrieval as a way to augment input examples at the granularity of small chunks
of tokens. Formally, we consider sequences of integer tokens in 𝕍 = [1 , 𝑣 ], obtained using a text
tokenizer 1 . We split each 𝑛 -token-long example 𝑋 = ( 𝑥 1 , . . . , 𝑥 𝑛 ) into a sequence of 𝑙 chunks ( 𝐶 1 , . . . , 𝐶 𝑙 )
of size 𝑚 = 𝑛𝑙 , i.e. 𝐶 1 , ( 𝑥 1 , . . . , 𝑥 𝑚 ) , . . . , 𝐶 𝑙 ,( 𝑥 𝑛 − 𝑚 +1 , . . . , 𝑥 𝑛 ) ∈ 𝕍 𝑚 . We use 𝑛 = 2048 and 𝑚 = 64.
We augment each chunk 𝐶 𝑢 with a set Re t D ( 𝐶 𝑢 ) of 𝑘 neighbours from the database D. Re t D (or
1 We
use the notation [1 , 𝑣 ] , {1 , . . . , 𝑣 } throughout the text.
3Improving language models by retrieving from trillions of tokens
Ret for brevity) is a non-trainable operator specified in §2.3. Token likelihoods are provided by a
model, parameterized by 𝜃 , that takes as input both previous tokens and their retrieved neighbours.
This defines the following retrieval-enhanced sequence log-likelihood:
𝐿 ( 𝑋 | 𝜃, D) ,
𝑙 ∑︁
𝑚
∑︁
 𝜃 𝑥 ( 𝑢 −1) 𝑚 + 𝑖 |( 𝑥 𝑗 ) 𝑗< ( 𝑢 −1) 𝑚 + 𝑖 , (Re t D ( 𝐶 𝑢 0 )) 𝑢 0 <𝑢 .

(1)
𝑢 =1 𝑖 =1
We set Re t( 𝐶 1 ) = ∅, namely the likelihood of tokens from the first chunk does not depend on
any retrieval data. This likelihood definition preserves autoregressivity: the probability of the 𝑖 -th
token of the 𝑢 -th chunk, 𝑥 ( 𝑢 −1) 𝑚 + 𝑖 , only depends on previously seen tokens ( 𝑥 𝑗 ) 16 𝑗< ( 𝑢 −1) 𝑚 + 𝑖 and on the
data retrieved from the previous chunks (Re t( 𝐶 𝑢 0 )) 𝑢 0 <𝑢 . We can therefore directly sample with log-
probability  , where sampling within the chunk 𝐶 𝑢 is conditioned on the neighbours (Re t( 𝐶 𝑢 0 )) 𝑢 0 <𝑢 .
This makes retrieval-enhanced models directly comparable with the largest language models that are
evaluated by sampling.
2.3. Nearest neighbour retrieval
Retrieval neighbours. Our database consists of a key-value memory. Each value consists of two
contiguous chunks of tokens which we denote [ 𝑁, 𝐹 ] where 𝑁 is the neighbour chunk which is used
to compute the key, and 𝐹 is its continuation in the original document. The corresponding key is
the Be rt embedding of 𝑁 , averaged over time, that we denote Be rt ( 𝑁 ). For each chunk 𝐶 , we
retrieve its approximate 𝑘 -nearest neighbours from our key-value database using the 𝐿 2 distance
on BERT embeddings 𝑑 ( 𝐶, 𝑁 ) = ||Be rt ( 𝐶 ) − Be rt ( 𝑁 )|| 22 . The model receives the corresponding
values Ret( 𝐶 ) , ( [ 𝑁 1 , 𝐹 1 ] , . . . , [ 𝑁 𝑘 , 𝐹 𝑘 ]). Both neighbour chunks and their continuations provide
meaningful improvements, as illustrated in our ablation study (Appendix D). We use a length 64 for
both 𝑁 𝑗 and 𝐹 𝑗 , thus Re t( 𝐶 ) has a shape of 𝑘 × 𝑟 with 𝑟 = 128. To avoid retrieving the chunk 𝐶 𝑢 +1
in the retrieval set Re t( 𝐶 𝑢 ), which would break causality during training, we filter out neighbours
originating from the same document as the training sequence 𝑋 .
For a database of 𝑇 elements, we can query the approximate nearest neighbours in O (log 𝑇 ) time.
We use the SCaNN library (Guo et al., 2020) to achieve this. This means that we can query our
2 trillion token database in 10 ms whilst evaluating or sampling from the model; this expense is
amortized over a chunk length. Performing retrieval on-the-fly is too slow to keep up with the training
calculations—we leverage the frozen aspect of the embedding operator Be rt to precompute all
approximate nearest neighbours and save the results as part of the data. In Fig. 9 in the Appendix, we
show results where we only retrieve neighbours within Wikipedia. We find that neighbours tend to
come from 2-3 links away from a given article whereas random articles are more than 5 links apart.
Table 1 | MassiveText. The last column indicates the sampling weight during training. The multilingual
subsets include documents in 10 languages. The full breakdown is given in §A.1.
Source
Web
Books
News
Wikipedia
GitHub
Token count (M)
977,563
3,423,740
236,918
13,288
374,952
Documents (M) Multilingual Sampling frequency
1,208
20
398
23
143 Yes
No
No
Yes
No 55%
25%
10%
5%
5%
4Improving language models by retrieving from trillions of tokens
2.4. Retro model architecture
Our model relies on an encoder-decoder transformer architecture, integrating the retrieved data
through a cross-attention mechanism as introduced in Vaswani et al. (2017). First, the retrieved
tokens Ret( 𝐶 ) are fed into an encoder Transformer, which computes the encoded neighbours set 𝐸 .
Denoting the intermediate activations by 𝐻 , our transformer decoder then interleaves Re t ro-blocks
Retro( 𝐻, 𝐸 ) and standard Transformer blocks L M( 𝐻 ) (the hyperparameter 𝑃 ⊆ [1 , 𝐿 ] determines at
which layers we use a Re t ro-block). These blocks are built from three different residual operators
with signature ℝ 𝑛 × 𝑑 → ℝ 𝑛 × 𝑑 : a fully-connected layer Ff w, the standard sequence-level self-attention
layer Attn, and a chunked cross-attention layer Cca (· , 𝐸 ) that incorporates information from the
retrieval encoder:
Ret ro ( 𝐻, 𝐸 ) , Ff w (Cca (At t n ( 𝐻 ) , 𝐸 )) ,
and
L m( 𝐻 ) , Ff w(At t n( 𝐻 ))
(2)
Since Ffw, At t n and Cca are all autoregressive operators whose output at position 𝑖 only
depends on ( ℎ 𝑗 ) 𝑗 6 𝑖 , any succession of Re t ro and l m layers, followed by a token classification
head defines an autoregressive log-likelihood (1). An overview of the model architecture is given in
Algorithm 1 and in Fig. 2. We next describe the retrieval encoder and the chunked cross-attention
layer in more detail, and explain how to sample from Re t ro.
Encoding retrieval neighbours. For each chunk 𝐶 𝑢 , the 𝑘 retrieval neighbours Re t( 𝐶 𝑢 ) are fed into
0
a bi-directional transformer E ncode r, yielding the outputs 𝐸 𝑢 𝑗 , E ncode r(Re t( 𝐶 𝑢 ) 𝑗 , 𝐻 𝑢 ) ∈ ℝ 𝑟 × 𝑑 ,
where 𝑗 ∈ [1 , 𝑘 ] indexes each neighbour. The retrieval encoder is a non-causal transformer. It
is conditioned on 𝐻 𝑢 , the activations of chunk 𝐶 𝑢 , through cross-attention layers; this allows the
representations of the retrieval encoder to be modulated by the retrieving chunk in a differentiable
way. More precisely, the encoding of the 𝑗 th neighbour of the 𝑢 th chunk, Re t ( 𝐶 𝑢 ) 𝑗 , depends on the
attended activation 𝐻 𝑢 , ( ℎ ( 𝑢 −1) 𝑚 + 𝑖 ) 𝑖 ∈ [1 ,𝑚 ] ∈ ℝ 𝑚 × 𝑑 of chunk 𝐶 𝑢 at layer min( 𝑃 ). All neighbours for
0
all chunks are encoded in parallel, yielding a full encoded set 𝐸 , ( 𝐸 𝑢 𝑗 ) 𝑢 ∈ [1 ,𝑙 ] , 𝑗 ∈ [1 ,𝑘 ] ∈ ℝ 𝑙 × 𝑘 × 𝑟 × 𝑑 . We
0
denote 𝐸 𝑢 ∈ ℝ 𝑘 × 𝑟 × 𝑑 as the encoded neighbours for chunk 𝑢 ∈ [1 , 𝑙 ].
Chunked cross-attention. To perform the
 a given intermediate acti-
 Cca operation, we first split
𝑛 × 𝑑
+
𝑚 × 𝑑
vation 𝐻 ∈ ℝ
into 𝑙 −1 attending chunks 𝐻 𝑢 , ( ℎ 𝑢 𝑚 + 𝑖 −1 ) 𝑖 ∈ [1 ,𝑚 ] ∈ ℝ
, as depicted on the
𝑢 ∈ [1 ,𝑙 −1]
right of Fig. 2. 𝐻 𝑢 + holds the intermediary embeddings of the last token in chunk 𝐶 𝑢 and of the first
𝑚 − 1 tokens in 𝐶 𝑢 +1 2 . We compute the cross-attention between 𝐻 𝑢 + and 𝐸 𝑢 —the encoded retrieval
set obtained from chunk 𝐶 𝑢 . Attention is computed across time and across neighbours simultaneously,
as we merge the neighbour and time dimensions of 𝐸 𝑢 before applying cross-attention. Since there
is a notion of alignment between data chunks and retrieval neighbours, we use relative positional
encodings as described in §B.1.2.
We concatenate the 𝑙 −1 outputs of the per-chunk cross-attentions (each of shape 𝑚 × 𝑑 ) across
time, and properly pad the result; we thus form the output activation Cca( 𝐻, 𝐸 ) ∈ ℝ 𝑛 × 𝑑 . Formally,
for each chunk 𝐶 𝑢 and for each token 𝑖 ∈ [1 , 𝑚 ] we set
Cca( 𝐻, 𝐸 ) 𝑢 𝑚 + 𝑖 −1 , C a( ℎ 𝑢 𝑚 + 𝑖 −1 , 𝐸 𝑢 ) ,
(3)
2 The
last token of chunk 𝐶 𝑢 is the first to be able to access the
 retrieved  content 𝐸 𝑢 while maintaining autoregressivity
in (1). Hence, there is a one token overlap between chunk 𝐶 𝑢 = 𝑥 ( 𝑢 −1) 𝑚 + 𝑖
and the corresponding attending chunk
𝐶 𝑢 + , ( 𝑥 𝑢 𝑚 + 𝑖 −1 ) 𝑖 ∈ [1 ,𝑚 ] .
𝑖 ∈ [1 ,𝑚 ]
5Improving language models by retrieving from trillions of tokens
Algorithm 1: Overview of Re t ro model architecture.
Hyperparam: 𝑃 and 𝑃 enc , indices of layers with cross-attention in the decoder and encoder
respectively
Hyperparam: 𝐿 and 𝐿 enc , number of decoder layers and number of encoder layers.
Input: 𝑋 ∈ 𝕍 𝑛 : sequence of tokens. (Re t ( 𝐶 𝑢 )) 16 𝑢 6 𝑙 : the retrieved neighbours
Output: 𝑂 ∈ ℝ 𝑛 × |𝕍 | : the output logits
def Encode r (Re t( 𝐶 𝑢 ) 16 𝑢 6 𝑙 , 𝐻 ):
( 𝐻 𝑢 ) 𝑢 ∈ [1 ,𝑙 ] ← Sp l i t( 𝐻 )
for 𝑗 ∈ [1 , 𝑘 ] , 𝑢 ∈ [1 , 𝑙 ] do // Encoder shared across neighbours and chunks
𝑗
𝐸 𝑢 = E m b enc (Re t( 𝐶 𝑢 ) 𝑗 ) // May be shared with the decoder E M B
for 𝑝 0 ∈ [1 , 𝐿 enc ] do
𝑗
𝑗
𝐸 𝑢 ← At t n enc ( 𝐸 𝑢 ) // Bi-directional attention
0
if 𝑝 ∈ 𝑃 enc then
𝑗
𝑗
𝐸 𝑢 ← C a enc ( 𝐸 𝑢 , 𝐻 𝑢 )
𝑗
𝑗
𝐸 𝑢 ← Ff w enc ( 𝐸 𝑢 )
return 𝐸
𝐻 ← Emb( 𝑋 )
for 𝑝 ∈ [1 , 𝐿 ] do
𝐻 ← At t n( 𝐻 ) // Causal attention
if 𝑝 = min( 𝑃 ) then
// The neighbour E N C O D E R is conditioned with the decoder activations of
the last layer before the first cross-attention
𝐸 = E ncode r(Re t( 𝐶 𝑢 ) 16 𝑢 6 𝑙 , 𝐻 )
if 𝑝 ∈ 𝑃 then
𝐻 ← Cca ( 𝐻, 𝐸 )
𝐻 ← Ff w( 𝐻 )
𝑂 ← Read ( 𝐻 )
where Ca is the cross-attention residual operator over time-concatenated encoded neighbours. We
recall that this operator is defined in its simplest version by three parameter matrices 𝐾 ∈ ℝ 𝑑 × 𝑐 , 𝑄 ∈
ℝ 𝑑 × 𝑐 and 𝑉 ∈ ℝ 𝑑 × 𝑑 . For all ℎ ∈ ℝ 𝑑 and 𝑌 ∈ ℝ 𝑇 × 𝑑 , we define
C a( ℎ, 𝑌 ) , softmax( 𝑌 𝐾𝑄 𝑇 ℎ ) 𝑌𝑉,
(4)
where the softmax is performed on the second dimension and all products are matrix products. We
use multi-head cross-attention, and add positional encodings to the softmax(see §B.1.2).
The first 𝑚 − 1 tokens cannot attend to any neighbour of a previous chunk; at these positions, we
define Cca as the identity, setting Cca( 𝐻, 𝐸 ) 𝑗 , ℎ 𝑗 for all tokens 𝑗 ∈ [1 , 𝑚 − 1]. Finally, the last token
ℎ 𝑙𝑚 attends to the last retrieval set 𝐸 𝑙 and we set ℎ 𝑙 𝑚 , C a( ℎ 𝑙 𝑚 , 𝐸 𝑙 ) (not shown in Fig. 2). Listing 1
contains a simplified implementation of Cca. Note that chunked cross-attention is autoregressive:
the output of Cca at position 𝑖 depends on the sequence from tokens from 0 to 𝑖 that is input to Cca.
With Retro models, even though each Cca cross-attention attends only to the neighbours of
the preceding chunk Re t( 𝐶 𝑢 −1 ), the dependencies over previous neighbours are propagated via the
self-attention operations. The activations of the 𝑖 th token in the 𝑢 th chunk therefore potentially depend
upon the set of all previous neighbours Re t( 𝐶 𝑢 0 ) 𝑢 0 <𝑢 , without incurring the quadratic cost of cross
attending to that set.
6Improving language models by retrieving from trillions of tokens
Sampling. When sampling, at the end of a chunk 𝐶 𝑢 , we use SCaNN to retrieve neighbours Re t( 𝐶 𝑢 ),
based on the embedding Be rt ( 𝐶 𝑢 ). The encoded neighbours 𝐸 𝑢 = E ncode r(Re t( 𝐶 𝑢 )) are then
used to condition the generation of the next chunk 𝐶 𝑢 +1 , which we do incrementally: overall the
cost of sampling is thus quadratic in the size of the sampled sequence, as when sampling from
regular Transformers; the added cost of retrieval is linear in the number of chunks 𝑙 , and is negligible
compared to the token sampling cost in practice.
2.5. Baseline Transformer architecture
We use a transformer (Vaswani et al., 2017) similar to the one described in (Radford et al., 2019),
with some minimal changes: we replace LayerNorm with RMSNorm (Zhang and Sennrich, 2019) and
use relative position encodings (Dai et al., 2019). As baselines, we train retrieval-free transformers
with 132M, 368M, 1.3B and 7.0B parameters (embedding matrices are excluded from parameter
counts). The hyperparameters we used are detailed in Table 2. All retrieval models use the same
size encoder for the retrieval data, with 𝑑 0 = 896 and 2 layers, which roughly adds 19 𝑀 parameters.
The encoder uses relative positional encodings. The retrieval models contain one Re t ro-block every
3 blocks, starting from layer 6. For our smallest model, Cca is applied in layers 6, 9 and 12 of the
main pathway and also once for query conditioning in the encoder, which adds an additional 12 𝑀
parameters. The relative number of extra parameters reduces as we increase the baseline model size.
All models are implemented using JAX (Bradbury et al., 2018) and Haiku (Hennigan et al., 2020).
2.6. Quantifying dataset leakage exploitation
Re t ro models may arguably benefit more easily from evaluation dataset leakage, i.e. the fact that
we evaluate on data that were also present in the training set. To better understand how retrieval
improves language modelling performance, we therefore quantify evaluation likelihood as a function
of the overlap between the evaluation and training datasets.
The following approach can be used with any language model, and depends only on the frozen
retriever system presented in §2.3. We split the evaluation sequences ( 𝑋 𝑖 ) 𝑖 into chunks of length
𝑚 ≤ 64, and we see the training data as a set of chunks C. For each evaluation chunk 𝐶 ∈ C, we
retrieve the 10 closest neighbours (of length up to 128) in the training data. We then compute the
longest token substring common to both the evaluation chunk and its neighbours. This gives a number
𝑠 ∈ [0 , 𝑚 ]. The value 𝑟 ( 𝐶 ) = 𝑚 𝑠 , ranging from 0 (chunk never seen) to 1 (chunk entirely seen), gives a
reliable indication of how much overlap there is between the evaluation chunk and the training data.
For a given model, we then obtain the log-likelihood  ( 𝐶 ) of each chunk 𝐶 , and the number of bytes
𝑁 ( 𝐶 ) it encodes. We then consider the filtered bits-per-bytes of the model:
Í
𝐶 ∈ C 𝛼  ( 𝐶 )
∀ 𝛼 ∈ [0 , 1] , C 𝛼 , { 𝐶 ∈ C , 𝑟 ( 𝐶 ) 6 𝛼 } , bpb( 𝛼 ) , Í
,
(5)
𝐶 ∈ C 𝛼 𝑁 ( 𝐶 )
Table 2 | Number of parameters for our baseline and Re t ro models, excluding embeddings, along
with the corresponding hyperparameters.
Baseline parameters Re t ro 𝑑 𝑑 ffw
132M
368M
1,309M
6,982M 172M (+30%)
425M (+15%)
1,451M (+11%)
7,532M (+8%) 896
1,536
2,048
4,096 3,584
6,144
8,192
16,384
# heads Head size # layers
16
12
16
32 64
128
128
128 12
12
24
32
7Improving language models by retrieving from trillions of tokens
which correspond to the bits-per-bytes on the set of chunks that overlap less than 𝛼 % with the training
chunks. Note that the full evaluation bit-per-bytes performance is recovered by bpb(1). The function
bpb(·) allows us to evaluate the impact of evaluation leakage over predictive performance: for low 𝛼 ,
bpb( 𝛼 ) gives an indication on how the model performs on chunks that are entirely new; the slope of
bpb(·) shows how much the model exploits evaluation leakage.
3. Related Work
We first review existing work on using retrieval for language modelling, and compare Retro to these
works (see Table 3). As we train Re t ro models on a large dataset containing a substantial section
of the internet, our work raises potential privacy, safety, and fairness issues that we then review.
3.1. Retrieval for language modelling
Brants et al. (2007) show that scaling the training data to trillions of tokens improves the machine
translation performance of 𝑛 -gram models. More recently, GPT-2 (Radford et al., 2019), GPT-3 (Brown
et al., 2020), and Jurassic-1 (Lieber et al., 2021) show that scaling up language models leads to
massive improvements on many downstream tasks. At the same time, Carlini et al. (2021) demonstrate
that large-scale language models can perfectly memorise parts of their training data, suggesting that
enhancing models with retrieval may lead to further improvements. However, significant leakage
between train and test datasets (Lee et al., 2021; Lewis et al., 2021) makes comparing and evaluating
large models trained on large datasets difficult, especially once retrieval capabilities over the training
dataset are added.
Historically, information retrieval for text relies on inverted index matching such as TF-IDF and
BM25 (Robertson and Zaragoza, 2009). Foundational work use latent topic modelling approaches
like LDA (Blei et al., 2003) to identify relevant neighbours (Wei and Croft, 2006). Work in machine
translation such as Zhang et al. (2018) and Gu et al. (2018) retrieve translation pairs based on edit
distance between source sentences and guide the translation output using the closest retrieved target
sentences. The retrieval database may also be structured — for example, Ahn et al. (2016) use a
symbolic knowledge graph to improve an RNN language model.
With the success of deep learning, retrieving systems have partly switched to dense learned
representations based on a neural network’s activations. Continuous cache (Grave et al., 2017)
adds probability mass to tokens for which previous activations resemble the current activation
vector, extending the model’s context to the local history. 𝑘 NN-LM (Khandelwal et al., 2020) applies
this idea to transformers and extends the retrieval database to English Wikipedia, resulting in
Table 3 | Comparison of Re t ro with existing retrieval approaches.
Continuous Cache
𝑘 NN-LM
Spal m
Dp r
Realm
R AG
Fi D
Em dr 2
Ret ro (ours)
# Retrieval tokens

O 10 3 
O 10 9 
O 10 9 
O 10 9 
O 10 9 
O 10 9 
O 10 9 
O 10 9 
O 10 12
Granularity Retriever training Retrieval integration
Token
Token
Token
Prompt
Prompt
Prompt
Prompt
Prompt
Chunk Frozen (L ST M)
Frozen (Transformer)
Frozen (Transformer)
Contrastive proxy
End-to-End
Fine-tuned Dp r
Frozen Dp r
End-to-End (EM)
Frozen (Be rt) Add to probs
Add to probs
Gated logits
Extractive QA
Prepend to prompt
Cross-attention
Cross-attention
Cross-attention
Chunked cross-attention
8Improving language models by retrieving from trillions of tokens
substantial improvements on Wikitext103 evaluation. Continuous cache and 𝑘 NN-LM do not modify
the underlying neural-network models, but interpolate at inference between the language model’s
output and distributions computed from retrieved tokens. These methods can therefore be plugged
into any model without additional training, although this limits the model’s ability to reason about
the retrieved text. Spal m (Yogatama et al., 2021) addresses this limitation by adding an extra gating
network to post-process the retrieved data; yet most of the network is unaffected by the retrieval
during inference.
The retrieval representations may be trained directly instead of relying on a pre-trained model—
retriever systems have been developed for this purpose, primarily on open-domain question answering.
For example, Dp r (Karpukhin et al., 2020) trains two Be rt models (for queries and keys respectively)
using a contrastive loss to align the representations of a question and of its answers. Lee et al. (2019)
use an inverse cloze task to find semantic representations of passages for retrieval. These works differs
from continuous cache and 𝑘 NN-LM in that they embeds passages (or chunks) of text together, as
opposed to each token individually. The retriever network is trained in isolation of the downstream
task that uses the retrieval data. This potential issue is specifically addressed by Re al m (Guu et al.,
2020), which trains the retrieval system end-to-end to maximize the final training cross-entropy. This
comes with the extra complexity of searching the database during training and periodically updating
the embedding table, severely limiting the scale at which it can operate. R AG (Lewis et al., 2020)
and FiD (Izacard and Grave, 2021) build upon Dp r to set the state of the art on question answering
benchmarks by training encoder-decoder transformer models. More recently, E m dr 2 (Sachan et al.,
2021) extends F i D by using an expectation-maximization algorithm to train the retriever end-to-end
and achieves state of the art results compared to similarly sized models.
In the open-domain dialogue setting, BlenderBot 2.0 (Komeili et al., 2021) learns to issue textual
internet queries, outperforming dense retrieval methods when evaluated on a task measuring how
close model responses are to those of humans. This involves collecting a dataset of human dialogues
with associated search queries, which limits the scalability of this approach. Hashemi et al. (2020)
introduce the Guided Transformer, a modified Transformer similar to Retro, for document retrieval
and clarifying question selection. Although effective on question answering and other tasks with
strong conditioning, none of these methods are designed to model arbitrary text sequences, in contrast
with Retro.
Retro shares components with 𝑘 NN-LM and Dp r in that it uses frozen retrieval representations.
Retro models longer sequences than QA examples; this requires to reason at a sub-sequence level,
and to retrieve different documents for the different chunks of a sequence. Similar to F i D, Ret ro
processes the retrieved neighbours separately in the encoder, and assemble them in the chunked
cross-attention. This differs from e.g. Re al m, that prepends retrieved documents to the prompt.
Using chunks allows for repeated retrieval whilst generating a sequence as opposed to retrieving
only once based on the prompt alone. Furthermore, retrieval is done during the whole pre-training
process in Retro, and is not simply plugged-in to solve a certain downstream task. Finally, previous
methods based on dense query vectors use small models and retrieval datasets with less than 3B
tokens (English Wikipedia). Table 3 summarizes the difference of Retro with existing approaches.
3.2. Privacy, safety and fairness
Bender et al. (2021); Weidinger et al. (2021) highlight several dangers of large language models.
Those stem from their ability to memorise training data, their high training cost, the static nature
of their training data (Lazaridou et al., 2021), their tendency of amplifying inherent biases in the
training data, and their ability to generate toxic language (Gehman et al., 2020). In this section we
inspect these dangers, focusing on how retrieval augmented language models may exacerbate or
9Improving language models by retrieving from trillions of tokens
mitigate them.
Large language models can perfectly memorise parts of their training data (Carlini et al., 2021).
When coupled with large training datasets gathered from the web or other sources, this has clear
privacy and safety implications. Retrieval models such as Re t ro that have access to the entire training
dataset during inference exacerbate these privacy issues by being able to directly copy training data.
However, retrieval systems offer a path towards mitigating these concerns via obliteration of the
retrievable data at inference time. In addition, differential privacy training (Abadi et al., 2016) of
retrieval models could guarantee that no private information is stored in the model weights, while
individualisation on private data could be made by updating the retrieval database at inference time.
Due to their high training cost, re-training large language model regularly to incorporate new
data, languages, and norms is prohibitively expensive. To keep retrieval models up-to-date, it may be
sufficient to update the retrieval database, which is orders of magnitude cheaper than re-training
a model from scratch. In addition to the benefits of updating models in terms of fairness and bias,
simply training large language models has a significant energy cost (Schwartz et al., 2020; Strubell
et al., 2019). Retrieval mechanisms offer a path to reducing the compute requirements needed to
train and update language models that reach a certain performance.
Large language models are prone to generating toxic outputs, as shown in Gehman et al. (2020).
Bender et al. (2021); Jo and Gebru (2020) advocate for the importance of better training data curation
and documentation. Additionally, if portions of the training data are found to be eliciting biased or
toxic outputs after training, retrieval allows for some correction, as the offending retrieval data can
be retroactively filtered. However, it is also the case that without careful analysis and intervention,
retrieval models may exacerbate biases that are present in the training data. Retrieval models can
also add a further source of bias through the selection mechanism for retrieval documents. Further
work in this area is required to better understand how retrieval affects the bias and toxicity of the
model outputs.
Finally, samples from large models are difficult to interpret, making mitigating these issues all the
more challenging (Belinkov et al., 2020; Jain and Wallace, 2019). Retrieval provides more insights in
to the outputs of a model, as one can directly visualise or modify the neighbours that are being used.
The examples in Table 6, 7, 20 and 21 illustrate how retrieval makes language models more factual
and interpretable by providing more transparent outputs.
4. Results
We first report results on language modelling benchmarks. Second, we show how to Re t rofit
pre-trained Transformer language models into retrieval models with few additional FLOPs. Next,
we report Ret ro results on question answering. Finally, we report evaluation metrics with leakage
filtering, to better understand the source of the gains with retrieval.
4.1. Language modelling
Datasets. We evaluate our models on C4 (Raffel et al., 2020), Wikitext103 (Merity et al., 2017),
Curation Corpus (Curation, 2020), Lambada (Paperno et al., 2016) and the Pile (Gao et al., 2020).
We also evaluate on a set of manually selected Wikipedia articles that were added or heavily edited in
September 2021, months after our pre-training and retrieval dataset was collected (details are given
in §A.2). We construct the dataset with articles from the “future” and manually remove new articles
that strongly overlap documents in our training data. This guarantees that the evaluation documents
are not leaked in our training data.
10Improving language models by retrieving from trillions of tokens
172M
425M
a) LAMBADA Accuracy
0.70
0.65
0.60
0.55
0.50
0.45
200 400 800 1600
7500
Non-Embedding Params (M)
0.70
1.5B
7.5B
Baseline
b) Curation Corpus bpb
20
0.65 10
0.60 5
0.55 3
0.50
200 400 800 1600
7500
Non-Embedding Params (M)
2
RETRO [OFF]
RETRO [ON]
c) Wikitext103 Perplexity
d) Wikipedia Sept 21 bpb
0.85
0.80
0.75
0.70
0.65
0.60
200 400 800 1600
7500
Non-Embedding Params (M)
200 400 800 1600
7500
Non-Embedding Params (M)
Figure 3 | Scaling with respect to model size. (a) LAMBADA top-1 accuracy. (b) Evaluation loss on
curation corpus. (c) Perplexity on Wikitext103 valid. (d) Bits-per-byte on selected Wikipedia articles
from September 2021.
For C4, Wikitext103, the Pile, and our Wikipedia dataset we evaluate the language modelling
performance on entire documents and measure the bits-per-byte (bpb). We favour bits-per-byte over
loss as it is tokenizer agnostic. We evaluate with a sequence length of 2048 tokens but use a stride of
1024 within documents to mitigate boundary effects. On Curation Corpus we concatenate the article,
the “TL;DR:” string, and the summary, but only evaluate the bpb on the summary. For Lambada we
evaluate the accuracy on the last word, using greedy generation.
Model scaling. In Fig. 1(left) and Fig. 3 we show the language modelling performance as we scale
models from 150 million to 7 billion (non-embedding) parameters. We see that on all datasets,
Retro outperforms the baseline at all model sizes. Furthermore, we observe that improvements do
not diminish as we scale the models. The performance is dataset dependent, with the largest gains on
Wikitext103 and C4. Wikipedia articles and other web pages are similar to Wikitext103 documents,
even if not exact copies (§4.4), we thus obtain dramatic improvements on Wikitext103 as our retrieval
model is able to directly exploit these overlaps. The smallest gains are for Curation Corpus, where
Retro only slightly outperforms the baseline. This is expected as Curation Corpus summaries are
designed to only contain information from the source article and are not included in our retrieval
database. On our “future” Wikipedia September 2021 dataset, we also observe consistent gains for
all model sizes.
Data scaling. Fig. 1 (middle) shows how scaling the retrieval database at evaluation improves the
language modelling performance. We observe dramatic gains as the retrieval data is increased from
Wikipedia (4 billion tokens) to all of Massive text (1.7T tokens). Fig. 1(right) shows how performance
scales as we increase the number of retrieved chunks. Despite being only trained with 2 neighbours,
we see consistent improvements for all models when the number of neighbours is increased from 1 to
10. Furthermore, we observe that larger models are able to better utilise more neighbours: the 172M
model improves with up to 10 neighbours, whereas the 7B model improves with up to 40 neighbours.
The Pile. We evaluate our 7B models on the Pile test sets 3 and compare against the 178B parameter
Jurrasic-1 (Lieber et al., 2021) model and the 280B parameter Gopher (Rae et al., 2021) model. We
do not compare against GPT-3 as it is outperformed by Jurassic-1 and Gopher on almost all subsets.
Fig. 4 shows the relative improvements in bits-per-byte over our 7B transformer baseline for our
3 Due
to legal and ethical concerns relating to their use, we exclude the Enron Emails and the Youtube Subtitles datasets.
11Relative bits-per-byte improvement over our 7B baseline without retrieval
uspto_backgrounds
arxiv
Jurassic-1 (178B)
Gopher (280B)
RETRO (7.5B)
100
80
60
40
20
0
20
Improving language models by retrieving from trillions of tokens
Figure 4 | The Pile: Comparison of our 7B baseline against Jurassic-1, Gopher, and Re t ro. We
observe that the retrieval model outperforms the baseline on all test sets and outperforms Jurassic-1
on a majority of them, despite being over an order of magnitude smaller.
7.5B Re t ro model, Jurassic-1 and Gopher. Jurassic-1 outperforms the baseline on all datasets
except for books, likely due to the inclusion of books in our training data. Gopher and Re t ro
outperform the baseline on all test sets. Overall, Retro 7.5B outperforms Jurassic-1 and Gopher on
a majority of the test sets. On the dm_mathematics and ubuntu_irc subsets, our Retro model
does not outperform our 7B baseline and underperforms Jurassic-1. We hypothesise that the retrieved
neighbours on these datasets are not helpful, due to a combination of what is in our retrieval dataset
and the efficacy of the nearest-neighbour search.
Wikitext103. To validate our approach in a controlled setting, we compare our method with 𝑘 NN-LM
(Khandelwal et al., 2020) on the Wikitext103 dataset in Table 4. We train a baseline transformer
on the training set of Wikitext103. This transformer has 24 layers, 1024 hidden units, 16 heads
and a key size of 64, as in Baevski and Auli (2019). Our baseline does not have adaptive input, and
our tokenizer has an open vocabulary, unlike Baevski and Auli (2019), which makes our baseline
Table 4 | Perplexities on Wikitext103. When using the Wikpedia dataset for retrieval, Ret ro
performs similarly to our implementation of 𝑘 NN-LM. As we scale the retrieval dataset, Ret ro
performs much better. The perplexities for retrieving from full MassiveText are quite low, which is
partly due to partial overlap with Wikitext103 not caught by our deduplication.
Model Retrieval Set
Adaptive Inputs (Baevski and Auli, 2019)
Spal m (Yogatama et al., 2021)
𝑘 NN-LM (Khandelwal et al., 2020)
Megatron (Shoeybi et al., 2019) -
Wikipedia
Wikipedia
-
Baseline transformer (ours)
𝑘 NN-LM (ours)
Re t ro
Re t ro
Re t ro
Re t ro
Re t ro -
Wikipedia
Wikipedia
C4
MassiveText (1%)
MassiveText (10%)
MassiveText (100%)
#Database tokens #Database keys Valid Test
-
3B
3B
- -
3B
3B
- 17.96
17.20
16.06
- 18.65
17.60
16.12
10.81
-
4B
4B
174B
18B
179B
1792B -
4B
0.06B
2.9B
0.8B
4B
28B 21.53
18.52
18.46
12.87
18.92
13.54
3.21 22.96
19.54
18.97
10.23
20.33
14.95
3.92
12Improving language models by retrieving from trillions of tokens
perplexities a bit higher. The full experiment details and hyperparameters are given in §C.2 and
Table 11.
We re-implement 𝑘 NN-LM with our tokenizer and baseline transformer to produce embeddings of
size 1024 for every token in Wikitext103. 𝑘 NN-LM has probabilities 𝑝 𝑘 NN-LM = 𝜆 𝑝 𝑘 NN + (1 − 𝜆 ) 𝑝 L m
with 𝑝 𝑘 NN ( 𝑛 𝑘 ) ∝ exp (− 𝛼𝑑 𝑘 ). We tune 𝜆 = 0 . 118 and 𝛼 = 0 . 00785 on the validation set (Fig. 7) and
report performance for these hyperparameters on both the validation and test set.
We fine-tune our baseline transformer into a Re t ro model (Fig. 7), using the Wikitext103
training data and retrieving from Wikipedia with 2 neighbours. We only train the new weights, as
explained in §4.2, and share the embedding weights between the encoder and the main pathway.
This is necessary for Wikitext103 which is quite small, as training Re t ro from scratch in this setting
leads to over-fitting.
We evaluate the fine-tuned Re t ro model with different retrieval sets. We use 10 neighbours at
evaluation for both Re t ro and 𝑘 NN-LM. When retrieving from Wikipedia, we obtain results com-
parable to our 𝑘 NN-LM implementation. Furthermore, scaling the retrieval database to MassiveText
yields dramatic improvements, though this is partly due to leakage (see §4.4). For reproducibility,
we also include results when retrieving from C4, which are close to previous state-of-the-art and
comparable to using 10 % of MassiveText.
It is worth noting that 𝑘 NN-LM requires 1024 floats for every token in the retrieval dataset,
totalling 15 terabytes (Tb) for the 4 billion tokens in Wikipedia. 𝑘 NN-LM and other token-level
retrieval approaches therefore don’t scale to retrieval databases with trillions of tokens such as
MassiveText. In comparison, Re t ro only requires 215Gb to index our Wikipedia dataset, and 93Tb
for MassiveText. Inspecting the number of retrieval database entries in Table 4 makes it clear why
retrieving at the chunk level is necessary when scaling to datasets with trillions of tokens.
4.2. Retro-fitting baseline models
We extend baseline models into Re t ro models by freezing the pre-trained weights and training
only chunked cross-attention and neighbour encoder parameters (less than 10% of weights for the
7B model) in Fig. 5. This offers an efficient alternative path to enhance transformers with retrieval,
requiring only 6 million sequences (3% of the pre-training sequences that we used). Additionally,
by only training the new weights we ensure that when evaluated without retrieval, the original
model performance is exactly maintained. Retrofitting models quickly surpasses the performance of
baseline models and even achieves performance close to that of Re t ro models trained from scratch.
The experiment hyperparameters are given in §C.3.
4.3. Question answering
We fine-tune our retrieval models on the Natural Questions (Kwiatkowski et al., 2019) dataset
to demonstrate that our retrieval pathway can be used to inject information from arbitrary data
sources. We use the version 4 provided by Izacard and Grave (2021) which is augmented with the
retrieved passages from Dp r (Karpukhin et al., 2020). We fine-tune all the weights of our 7.5B
pre-trained Re t ro model for 25,000 steps using the top 20 retrieved passages. We format the
data as “question: {question} \n answer: {answer}” and left pad the data such that
“answer:” coincides with the end of the first chunk of 64 tokens and thus aligns with the first
retrieving chunk. The model has access to the question via the previous tokens in the sequence as well
as the top 20 DPR Wikipedia passages and their titles via the chunked cross-attention mechanism.
4 https://github.com/facebookresearch/FiD
13Improving language models by retrieving from trillions of tokens
Figure 5 | Ret ro-fitting a baseline transformer. Any transformer can be fine-tuned into a retrieval-
enhanced transformer by randomly initializing and training only the chunked cross-attention and
retrieval encoder weights. Fine-tuning in this way quickly recovers and surpasses the non-retrieval
performance, and almost achieves the same performance as training a retrieval model from scratch
(shown by the arrow on the right hand side of each plot). We find good performance Re t ro-fitting
our models training on only 3% the number of tokens seen during pre-training.
The exact match scores are shown in Table 5 and the full fine-tuning details are given in §C.4. Our
method is competitive with previous approaches such as Re al m, R AG and Dp r, but underperforms
the more recent F i D. In contrast with this work, we find that increasing the number of neighbours
past 20 does not improve Re t ro performance on this task. We hypothesise that the encoder-decoder
structure of T5—the base model in F i D— and the T5 pre-training objective leads to a model that
relies more on the encoder output than Re t ro, which is important in the QA setting. To compete
with T5-finetuned models, future work should consider ways of forcing Retro to rely further on the
retrieval encoder output when producing tokens.
4.4. Relating retrieval performance to dataset leakage.
We report the filtered eval losses as detailed in §2.6 on C4, Curation Corpus and Wikitext103 in Fig. 6.
On C4 and Wikitext103, for which there is leakage into the training set, the slope is negative for both
baseline models and Re t ro models. Re t ro models exploit leakage more strongly than baseline
models, as indicated by the more negative slope. This is due to its explicit ability to copy-paste existing
training chunks to predict leaked evaluation chunks (see a qualitative example of this model behavior
Table 5 | Question answering results. Exact match accuracy on Natural Questions.
Model
Test Accuracy
Re al m (Guu et al., 2020)
Dp r (Karpukhin et al., 2020)
R AG (Lewis et al., 2020)
E m dr 2 (Sachan et al., 2021)
F i D (Izacard and Grave, 2021)
F i D + Distill. (Izacard et al., 2020) 40.4
41.5
44.5
52.5
51.4
54.7
Baseline 7B (closed book)
Re t ro 7.5B (DPR retrieval) 30.4
45.5
14Improving language models by retrieving from trillions of tokens
172M
C4
0.65
0.9 0.60
0.8 0.55
0.7 0.50
1.0
12.5%
50%
425M
1.5B
7.5B
Curation Corpus
100% 12.5%
Baseline
0.6
0.4
0.2
100% 12.5%
Wikipedia Sept 2021
0.85
0.80
0.75
0.70
0.65
0.60
0.8
50%
RETRO [ON]
Wikitext103
50%
Max eval/train chunk overlap when filtering
100% 12.5%
50%
100%
Figure 6 | Performance vs. longest common retrieval substring. Evaluation loss as a function of
allowed longest common substring between evaluation data chunks and their nearest neighbours.
Retrieval still helps when considering chunks with no more than 8 contiguous tokens overlapping
with training dataset chunks.
on a Wikitext103 article in Table 19). On Curation Corpus, retrieval provides a constant offset, which
is expected as there is by design no leakage between Curation Corpus and the training dataset.
On the other hand, Retro outperforms baseline models at all leakage levels, down to 𝛼 = 12 . 5%.
At this level, the loss is computed on chunks with less than 8 contiguous tokens shared with the
closest matching chunk in the training dataset—this is a reasonable level of overlap at which we
consider that there is no local leakage. Retrieval thus improves predictions on both chunks that are
syntactically similar to chunks in the training set, and on chunks that are syntactically different from
all training chunks. This points toward a non trivial Re t ro capacity of generalizing based on both
model parameters and retrieval database. Similar results are found on the Pile dataset (see Fig. 12,
§F.3).
4.5. Using Re t ro for sampling
We show examples of samples obtained using the 7.5B Re t ro model in Table 6, Table 7 and
Appendix E. For each chunk (the first one being the prompt), we juxtapose sampled chunks 𝐶 𝑢 with
retrieved neighbours Re t( 𝐶 𝑢 ). To give an indication of local overlap, we colour each sampled token
in chunk 𝐶 𝑢 based on the length of the longest common prefix (LCP) found in the retrieved chunks
Ret( 𝐶 𝑢 −1 ). Similarly, we colour the retrieved chunks based on the LCP in the sampled chunk. For the
sample in Table 6, for which we chose the prompt, we observe that the retrieved chunks influence the
sample as there are overlaps between the sampled tokens and neighbour tokens. Overall, retrieval
reduces hallucinations (in line with the findings of Shuster et al. (2021)) and makes the model more
knowledgeable, when comparing with samples produced with retrieval disabled. In the sample in
Table 7, the model recognises that the prompt is the beginning of the first scene of Hamlet and
leverages retrieval data to continue it with only a few mistakes. We provide further examples in
Appendix E, including examples from the evaluation sets, as well as the detailed procedure used for
colouring the tables.
5. Conclusion
We present Retrieval-Enhanced Transformers (Re t ro), a method for modelling arbitrary text se-
quences whilst retrieving from databases with trillions of tokens—scaling the data available to models
by an order of magnitude compared to what is typically consumed during training. Re t ro models
15Improving language models by retrieving from trillions of tokens
gains do not diminish for models with up to at least 7B parameters, and correspond to non-retrieval
models with 10× more parameters on certain datasets. On Wikitext103 and the Pile, Retro outper-
forms previous models trained on large scale datasets. We also show that Re t ro is competitive on
retrieval-intensive downstream tasks such as question answering.
Retro models are flexible and can be used without retrieval at evaluation and still achieve
comparable performance to baseline models. Conversely, baseline models can be rapidly fine-tuned
into Retro models to obtain nearly the same performance as if trained from scratch. Careful analysis
shows that only a modest fraction of the gains obtained by Re t ro are due to test set leakage. In
general, we caution for such leakage in large-scale language datasets and suggest further work in
better understanding the role of test set leakage in the performance of large-scale language models.
Overall, our work demonstrates at an unprecedented scale that semi-parametric approaches can
provide an orthogonal, more efficient approach than raw parameter scaling as we seek to build more
powerful language models.
Acknowledgements
We would like to thank Nikolai Grigorev, Marc’aurelio Ranzato, Cyprien de Masson d’Autume, Po-Sen
Huang, Johannes Welbl, Lisa Anne Hendricks, Ethan Perez, Jeff Stanway, Eric Noland, Gregory Wayne,
John Jumper, Julian Schrittwieser, Lorrayne Bennett, Devang Agrawal, Dani Yogatama, Susannah
Young, Nando de Freitas, Demis Hassabis, and Koray Kavukcuoglu for their help, advice and reviews.
Additionally, we would like to thank Zonglin Li, David Simcha, and the ScaNN developers for their
help.
16Improving language models by retrieving from trillions of tokens
Table 6 | Sample - Beavers are interesting animals. The Re t ro [Of f ] sample quickly diverges to other
animals while the Re t ro [On ] sample tends to stay focused on the beaver topic due to neighbour conditioning.
[ 𝑁 𝑢 1 , 𝐹 𝑢 1 ] colored by LCP with 𝐶 𝑢 +1 [ 𝑁 𝑢 2 , 𝐹 𝑢 2 ] colored by LCP with 𝐶 𝑢 +1
LCP = 0 , 1 , 2 , 3 , 4 , > 5 LCP = 0 , 1 , 2 , 3 , 4 , > 5
.Beavers build their lodges in pon
ds they have created in wooded areas
.Like many things in nature, there
is a connection between creatures i
n the wild.Beaver ponds cause tree
s to drown, but the dead trees attra
ct the great blue heron, which often
return year after year. Over time,
a beaver pond can attract more than
50 nests in a colony, called a rooke
ry.An example of this can be found
in the large pond off Bradford Road
at Carter Fields near the Boxford l
ine.Chris Leahy, an expert with th
e Massachusetts Audubon Society who
wrote naw them into smaller sections and d
rag them into the water.Engineers
of the Pond Beavers are interesting
animals because they change the hab
itat in which they live. Beavers do
this by blocking up streams to creat
e ponds. Then they build their homes
, called lodges, in these ponds. Bea
vers’ bodies make them well-suited f
or underwater building Special muscl
es close off their noses, ears, and
throats to keep the water out. Beave
rs’ broad tails act like rudders for
steering. Their two very large, ora
nge front teeth are used to gnaw dow
n trees. They begin building their d
am
their houses called beaver dams in
the riverbeds. They also live on lan
d.Beavers use their strong teeth an
d strong jaws to cut down trees and
branches to build their homes. They
also use their teeth and jaws to che
w up food. Beavers use their big, fl
at tails to swim. They use , then they mean that you are very b
usy. Beavers swim easily in streams,
picking up rocks and sticks to buil
d their dams. They gnaw at trees wit
h their big front teeth to cut them
down. Then they use parts of the tre
es to build their houses.Beavers ar
e clever builders. They know exactly
what they need to build their beave
r dams. They use mud from the stream
to make their dams stay together. T
hey use their tails to pat down the
mud.Beavers put a snug room at the
top of their dams for their babies.
They store their food underwater. Be
avers eat the bark from the ar-like tail, and two protruding tee
th that are strong enough to gnaw do
wn trees. The beaver uses trees, bra
nches, and mud to build dams across
rivers and streams. These dams creat
e a deep pool of water in which the
beaver builds its home. A beaver hom
e is called a lodge. A baby beaver o
r “kit” remains in the family lodge
until the age of two.Beaver fur, kn
own as pelt, was once highly popular
as a trim for hats and coats. How m
ight the popularity of beaver fur co
ntributed to the colonization of New
York?wwwWw
cold areas. A kangaroo is an anima
l that is very large. It can jump ve
ry high. They can run very fast. A
meerkat is a small animal that lives
in Africa. They live in colonies of
up to a hundred of them. They can c
limb trees very easily. their sharp teeth to gnaw at rocks
and cut trees. They use their flat t
ails to steer when they are swimming
.In the winter, beavers have to bui
ld a house called a beaver lodge. Th
ey build a house that is very strong
. The walls are made of twigs. The r
oof is made , then they mean that you are very b
usy. Beavers swim easily in streams,
picking up rocks and sticks to buil
d their dams. They gnaw at trees wit
h their big front teeth to cut them
down. Then they use parts of the tre
es to build their houses.Beavers ar
e clever builders. They know exactly
what they need to build their beave
r dams. They use mud from the stream
to make their dams stay together. T
hey use their tails to pat down the
mud.Beavers put a snug room at the
top of their dams for their babies.
They store their food underwater. Be
avers eat the bark from the w at trees with their big front teet
h to cut them down. Then they use pa
rts of the trees to build their hous
es.Beavers are clever builders. The
y know exactly what they need to bui
ld their beaver dams. They use mud f
rom the stream to make their dams st
ay together. They use their tails to
pat down the mud.Beavers put a snu
g room at the top of their dams for
their babies. They store their food
underwater. Beavers eat the bark fro
m the trees that they cut down!1. W
hat is the main idea of the first pa
ragraph?.2. What is the main idea o
f the second paragraph?
A mouse is a small mammal that lives
on land. It is a very good climber
and it can run very fast. Penguins
are birds that live on Antarctica. T
hey have a thick coat to keep them w
arm. Rabbits are small animals that
live in the ground. They of branches and other tree parts. T
hey also use their strong jaws to cu
t trees. They bring them to their ho
use. They also use their sharp teeth
to chew up the tree parts. They use
their flat tails to swim to the top
of their house. Then they use their
teeth and jaws to chew up the tree Prompt and sample of Retro[Off] Prompt and sample of Retro[On]
colored by LCP with Ret ( 𝐶 𝑢 −1)
LCP = 0 , 1 , 2 , 3 , 4 , > 5
Beavers are interesting animals that
live near rivers. They build Beavers are interesting animals that
live near rivers. They build
dams to create ponds. Frogs are am
phibians, so they can live in both l
and and water. They have great camou
flage to hide from predators. The G
olden Retriever, or Golden, is a ver
y big dog. They are very strong, and
have a thick coat to help them live
in
17Improving language models by retrieving from trillions of tokens
Table 7 | Sample - Hamlet, Act 1, Scene 1. The Re t ro [Of f ] sample has correct syntax but is hallucinated,
and ends with repetition of one character (FRANCISCO Approach me not). The Re t ro [On ] sample is the
correct continuation of the original text, and is robust to formatting differences between our prompt and the
retrieved data.
[ 𝑁 𝑢 1 , 𝐹 𝑢 1 ] colored by LCP with 𝐶 𝑢 +1 [ 𝑁 𝑢 2 , 𝐹 𝑢 2 ] colored by LCP with 𝐶 𝑢 +1
LCP = 0 , 1 , 2 , 3 , 4 , > 5 LCP = 0 , 1 , 2 , 3 , 4 , > 5
ACT I SCENE I. Elsinore. A pla
tform before the castle. (FRAN
CISCO at his post. Enter to hi
m BERNARDO) BERNARDO Who’s the
re? FRANCISCO Nay, answer me:
stand, and unfold yourself. BE
RNARDO Long live the king! FRA
NCISCO Bernardo? BERNARDO He.
FRANCISCO You come most carefu
lly upon your hour. BERNARDO ’
Tis now struck twelve; get the
e to bed, Francisco. FRANCISCO
For this relief much thanks:
’tis bitter cold, And I am sic
k at heart. BERNARDO Have you ><TEXT>ACT I SCENE I. Elsin
ore. A platform before the cas
tle. FRANCISCO at his post. E
nter to him BERNARDO BERNARDO
Who’s there? FRANCISCO Nay, an
swer me: stand, and unfold you
rself. BERNARDO Long live the
king! FRANCISCO Bernardo? BERN
ARDO He. FRANCISCO You come mo
st carefully upon your hour. B
ERNARDO ’Tis now struck twelve
; get thee to bed, Francisco.
FRANCISCO For this relief much
thanks: ’tis bitter cold, And
I am sick at heart.
Long live the king! FRANCISCO
Bernardo? BERNARDO He. FRANCI
SCO You come most carefully up
on your hour. BERNARDO ’Tis no
w struck twelve; get thee to b
ed, Francisco. FRANCISCO For t
his relief much thanks: ’tis b
itter cold, And I am sick at h
eart. B Long live the king! FRANCISCO
Bernardo? BERNARDO He. FRANCI
SCO You come most carefully up
on your hour. BERNARDO ’Tis no
w struck twelve; get thee to b
ed, Francisco. FRANCISCO For t
his relief much thanks: ’tis b
itter cold, And I am sick at h
eart.</TEXT></DOC><DOC><DO
CNO>romeo</DOCNO><TEXT>ACT I
PROLOGUE Two households, bo
th alike in dignity, In fair V
erona, where we lay our scene,
From ancient grudge break to
new mutiny, live the king! FRANCISCO Bern
ardo? BERNARDO He. FRANCISCO Y
ou come most carefully upon yo
ur hour. BERNARDO ’Tis now str
uck twelve: get thee to bed, F
rancisco. FRANCISCO For this r
elief much thanks: ’tis bitter
cold, And I am sick at heart.
BERNARDO Have you had quiet g
uard? FRANCISCO Not a mouse st
irring. BERNARDO Well, good ni
ght. Ifyou do meet Horatio and
Marcellus, The rivals2 of my
watch, bid them make haste. FR
ANCISCO I think I hear them.—
Stand, ho! who is there? EN
Francisco, I would speak with
you. FRANCISCO Approach me not
, but speak. BERNARDO Your han
d, your voice FRANCISCO I will
not hear thee speak. BERNARDO
Francisco, your hand, I entre
at thee. FRANCISCO Approach me
not. BERNARDO Francisco FRANC ERNARDO Have you had quiet gua
rd? FRANCISCO Not a mouse stir
ring. BERNARDO Well, good nigh
t. If you do meet Horatio and
Marcellus, The rivals of my wa
tch, bid them make haste. FRAN
CISCO I think I hear them. Sta
nd, ho! who is there? Enter had quiet guard? FRANCISCO No
t a mouse stirring. BERNARDO W
ell, good night. If you do mee
t Horatio and Marcellus, The r
ivals of my watch, bid them ma
ke haste. FRANCISCO I think I
hear them. Stand, ho! Who’s th
ere? (Enter HORATIO and MARCEL
LUS) HORATIO Friends to this g
round. MARCELLUS And liegemen
to the Dane. FRANCISCO Give yo
u good night. MARCELLUS O, far
ewell, honest soldier: Who hat
h relieved you? FRANCISCO Bern
ardo has my place. Give you go
od night. (Exit ARDO Have you had quiet guard?
FRANCISCO Not a mouse stirrin
g. BERNARDO Well, good night.
Ifyou do meet Horatio and Marc
ellus, The rivals2 of my watch
, bid them make haste. FRANCIS
CO I think I hear them.— Stand
, ho! who is there? ENTER HORA
TIO AND MARCELLUS. HORATIO Fri
ends to this ground. MARCELLUS
And liegemen to the Dane.3 FR
ANCISCO Give you good night. M
ARCELLUS O, farewell, honest s
oldier: Who hath relieved you?
FRANCISCO Bernardo hath my pl
ace. Give you good night
ISCO Approach me not. BERNARDO
I have a letter FRANCISCO App
roach me not. BERNARDO For the
king. FRANCISCO Approach me n
ot. BERNARDO There’s no treaso
n in’t. FRANCISCO Approach me
not. BERNARDO I will HORATIO and MARCELLUS HORATIO
Friends to this ground. MARCE
LLUS And liegemen to the Dane.
FRANCISCO Give you good night
. MARCELLUS O, farewell, hones
t soldier: Who hath relieved y
ou? FRANCISCO Bernardo hath my
place. Give you good night. Prompt and sample of Retro[Off] Prompt and sample of Retro[On]
colored by LCP with Ret ( 𝐶 𝑢 −1)
LCP = 0 , 1 , 2 , 3 , 4 , > 5
ACT I SCENE I. Elsinore. A pla
tform before the castle. FRANC
ISCO at his post. Enter to him
BERNARDO BERNARDO Who’s there
? FRANCISCO Nay, answer me: st
and, and unfold yourself. BERN
ARDO ACT I SCENE I. Elsinore. A pla
tform before the castle. FRANC
ISCO at his post. Enter to him
BERNARDO BERNARDO Who’s there
? FRANCISCO Nay, answer me: st
and, and unfold yourself. BERN
ARDO
Who calls ? FRANCISCO I am th
e lord here; I, Francisco, tha
t am sick of grief. [ Aside. B
ERNARDO The king ! FRANCISCO I
am sick of that also. BERNARD
O My lord ? FRANCISCO Do not a
pproach me. BERNARDO
18Improving language models by retrieving from trillions of tokens
References
M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang. Deep learning
with differential privacy. In ACM SIGSAC Conference on Computer and Communications Security,
2016.
S. Ahn, H. Choi, T. Pärnamaa, and Y. Bengio. A neural knowledge language model. arXiv preprint
arXiv:1608.00318, 2016.
A. Baevski and M. Auli. Adaptive input representations for neural language modeling. In International
Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=
ByxZX20qFQ.
Y. Belinkov, S. Gehrmann, and E. Pavlick. Interpretability and analysis in neural NLP. In Proceedings
of the 58th Annual Meeting of the Association for Computational Linguistics: Tutorial Abstracts,
pages 1–5, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.
acl-tutorials.1. URL https://aclanthology.org/2020.acl-tutorials.1.
E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell. On the dangers of stochastic parrots:
Can language models be too big? In ACM Conference on Fairness, Accountability, and Transparency,
2021.
D. M. Blei, A. Y. Ng, and M. I. Jordan. Latent Dirichlet Allocation. Journal of Machine Learn-
ing Research, 3(Jan):993–1022, 2003. URL https://jmlr.csail.mit.edu/papers/v3/
blei03a.html.
J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. V.
der Plas, S. Wanderman-Milne, and Q. Zhang. JAX: composable transformations of Python+NumPy
programs, 2018. URL http://github.com/google/jax.
T. Brants, A. C. Popat, P. Xu, F. J. Och, and J. Dean. Large Language models in machine translation.
In Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural
Language Learning, pages 858–867, 2007.
T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry,
A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu,
C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish,
A. Radford, I. Sutskever, and D. Amodei. Language models are few-shot learners. In Advances
in Neural Information Processing Systems, 2020. URL https://proceedings.neurips.cc/
paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.
N. Carlini, F. Tramer, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. Brown, D. Song,
U. Erlingsson, A. Oprea, and C. Raffel. Extracting training data from large language models.
Preprint, 2021.
C. Consonni, D. Laniado, and A. Montresor. Wikilinkgraphs: a complete, longitudinal and multi-
language dataset of the wikipedia link networks. In AAAI International Conference on Web and
Social Media, volume 13, 2019.
Curation. Curation corpus base, 2020.
Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. Le, and R. Salakhutdinov. Transformer-XL: Attentive language
models beyond a fixed-length context. In Annual Meeting of the Association for Computational
Linguistics, July 2019. URL https://aclanthology.org/P19-1285.
19Improving language models by retrieving from trillions of tokens
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of deep bidirectional transformers
for language understanding. In Conference of the North American Chapter of the Association for
Computational Linguistics, June 2019. URL https://aclanthology.org/N19-1423.
L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima,
S. Presser, and C. Leahy. The Pile: An 800GB dataset of diverse text for language modeling. arXiv
preprint arXiv:2101.00027, 2020.
S. Gehman, S. Gururangan, M. Sap, Y. Choi, and N. A. Smith. RealToxicityPrompts: Evaluating neural
toxic degeneration in language models. In Conference on Empirical Methods in Natural Language
Processing, Nov. 2020. URL https://aclanthology.org/2020.findings-emnlp.301.
E. Grave, A. Joulin, and N. Usunier. Improving neural language models with a continuous cache. In
International Conference on Learning Representations, 2017. URL https://openreview.net/
forum?id=B184E5qee.
A. Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850,
2013.
J. Gu, Y. Wang, K. Cho, and V. O. Li. Search engine guided neural machine translation. In AAAI
Conference on Artificial Intelligence, 2018.
R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, F. Chern, and S. Kumar. Accelerating large-scale
inference with anisotropic vector quantization. In International Conference on Machine Learning,
2020. URL https://arxiv.org/abs/1908.10396.
K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang. Retrieval augmented language model pre-training.
In International Conference on Machine Learning, 2020.
H. Hashemi, H. Zamani, and W. B. Croft. Guided transformer: Leveraging multiple external sources
for representation learning in conversational search. In Proceedings of the 43rd International ACM
SIGIR Conference on Research and Development in Information Retrieval, pages 1131–1140, 2020.
T. Hennigan, T. Cai, T. Norman, and I. Babuschkin. Haiku: Sonnet for JAX, 2020. URL http:
//github.com/deepmind/dm-haiku.
G. Izacard and E. Grave. Leveraging passage retrieval with generative models for open domain
question answering. In Conference of the European Chapter of the Association for Computational
Linguistics, Apr. 2021. URL https://aclanthology.org/2021.eacl-main.74.
G. Izacard, F. Petroni, L. Hosseini, N. De Cao, S. Riedel, and E. Grave. A memory efficient baseline for
open domain question answering. arXiv preprint arXiv:2012.15156, 2020.
S. Jain and B. C. Wallace. Attention is not Explanation. In Proceedings of the 2019 Conference of
the North American Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers), pages 3543–3556, Minneapolis, Minnesota, June
2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1357. URL https:
//aclanthology.org/N19-1357.
E. S. Jo and T. Gebru. Lessons from archives: Strategies for collecting sociocultural data in machine
learning. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, pages
306–316, 2020.
R. Jozefowicz, O. Vinyals, M. Schuster, N. Shazeer, and Y. Wu. Exploring the limits of language
modeling. arXiv preprint arXiv:1602.02410, 2016.
20Improving language models by retrieving from trillions of tokens
J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu,
and D. Amodei. Scaling laws for neural language models. CoRR, 2020. URL https://arxiv.
org/abs/2001.08361.
V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih. Dense passage re-
trieval for open-domain question answering. In Conference on Empirical Methods in Natural Language
Processing, Nov. 2020. URL https://aclanthology.org/2020.emnlp-main.550.
U. Khandelwal, O. Levy, D. Jurafsky, L. Zettlemoyer, and M. Lewis. Generalization through memoriza-
tion: Nearest neighbor language models. In International Conference on Learning Representations,
2020. URL https://openreview.net/forum?id=HklBjCEKvH.
M. Komeili, K. Shuster, and J. Weston. Internet-augmented dialogue generation. arXiv preprint
arXiv:2107.07566, 2021.
T. Kudo and J. Richardson. Sentencepiece: A simple and language independent subword tokenizer
and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226, 2018.
T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin,
M. Kelcey, J. Devlin, K. Lee, K. N. Toutanova, L. Jones, M.-W. Chang, A. Dai, J. Uszkoreit, Q. Le, and
S. Petrov. Natural Questions: a benchmark for question answering research. Transactions of the
Association of Computational Linguistics, 7:452–466, Mar. 2019. URL https://aclanthology.
org/Q19-1026.
A. Lazaridou, A. Kuncoro, E. Gribovskaya, D. Agrawal, A. Liska, T. Terzi, M. Gimenez, C. de Mas-
son d’Autume, S. Ruder, D. Yogatama, K. Cao, T. Kociský, S. Young, and P. Blunsom. Pitfalls of static
language modelling. CoRR, 2021. URL https://arxiv.org/abs/2102.01951.
K. Lee, M.-W. Chang, and K. Toutanova. Latent Retrieval for Weakly Supervised Open Domain
Question Answering. In Annual Meeting of the Association for Computational Linguistic, June 2019.
URL http://arxiv.org/abs/1906.00300.
K. Lee, D. Ippolito, A. Nystrom, C. Zhang, D. Eck, C. Callison-Burch, and N. Carlini. Deduplicating
training data makes language models better. arXiv preprint arXiv:2107.06499, 2021.
P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih,
T. Rocktäschel, S. Riedel, and D. Kiela. Retrieval-augmented generation for knowledge-intensive NLP
tasks. In Advances in Neural Information Processing Systems, 2020. URL https://proceedings.
neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf.
P. Lewis, P. Stenetorp, and S. Riedel. Question and answer test-train overlap in open-domain question
answering datasets. In Conference of the European Chapter of the Association for Computational
Linguistics, Apr. 2021. URL https://aclanthology.org/2021.eacl-main.86.
O. Lieber, O. Sharir, B. Lenz, and Y. Shoham. Jurassic-1: Technical details and evaluation. White Paper.
AI21 Labs, 2021.
I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on
Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.
S. Merity, C. Xiong, J. Bradbury, and R. Socher. Pointer sentinel mixture models. In International
Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=
Byj72udxe.
21Improving language models by retrieving from trillions of tokens
T. Mikolov, M. Karafiát, L. Burget, J. Cernockỳ, and S. Khudanpur. Recurrent neural network based
language model. Interspeech, 2(3):1045–1048, 2010.
D. Paperno, G. Kruszewski, A. Lazaridou, N. Q. Pham, R. Bernardi, S. Pezzelle, M. Baroni, G. Boleda,
and R. Fernández. The LAMBADA dataset: Word prediction requiring a broad discourse context.
In Annual Meeting of the Association for Computational Linguistics, Aug. 2016. URL https://
aclanthology.org/P16-1144.
A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised
multitask learners. Preprint, 2019.
J. Rae, S. Borgeaud, T. Cai, K. Millican, J. Hoffmann, F. Song, J. Aslanides, S. Henderson, R. Ring,
S. Young, E. Rutherford, T. Hennigan, J. Menick, A. Cassirer, R. Powell, G. van den Driessche, L. A.
Hendricks, M. Rauh, P.-S. Huang, A. Glaese, J. Welbl, S. Dathathri, S. Huang, J. Uesato, J. Mellor,
I. Higgins, A. Creswell, N. McAleese, A. Wu, E. Elsen, S. Jayakumar, E. Buchatskaya, D. Budden,
E. Sutherland, K. Simonyan, M. Paganini, L. Sifre, L. Martens, X. L. Li, A. Kuncoro, A. Nematzadeh,
E. Gribovskaya, D. Donato, A. Lazaridou, A. Mensch, J.-B. Lespiau, M. Tsimpoukelli, N. Grigorev,
D. Fritz, T. Sottiaux, M. Pajarskas, T. Pohlen, Z. Gong, D. Toyama, C. de Masson d’Autume, Y. Li,
T. Terzi, V. Mikulik, I. Babuschkin, A. Clark, D. de Las Casas, A. Guy, J. Bradbury, M. Johnson,
B. Hechtman, L. Weidinger, I. Gabriel, W. Isaac, E. Lockhart, S. Osindero, L. Rimell, C. Dyer,
O. Vinyals, K. Ayoub, J. Stanway, L. Bennett, D. Hassabis, K. Kavukcuoglu, and G. Irving. Scaling
language models: Methods, analysis & insights from training Gopher. arXiv submission, 2021.
C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring
the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning
Research, 21(140):1–67, 2020. URL http://jmlr.org/papers/v21/20-074.html.
S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He. Zero: Memory optimizations toward training trillion
parameter models. In IEEE International Conference for High Performance Computing, Networking,
Storage and Analysis, 2020.
S. Robertson and H. Zaragoza. The probabilistic relevance framework: BM25 and beyond. Foundations
and Trends in Information Retrieval, 3:333–389, Jan 2009.
D. S. Sachan, S. Reddy, W. Hamilton, C. Dyer, and D. Yogatama. End-to-end training of multi-document
reader and retriever for open-domain question answering. arXiv preprint arXiv:2106.05346, 2021.
R. Schwartz, J. Dodge, N. A. Smith, and O. Etzioni. Green AI. Communications of the Association for
Computing Machinery, 63(12):54–63, Nov. 2020.
M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro. Megatron-LM: Training
multi-billion parameter language models using model parallelism. CoRR, 2019. URL http:
//arxiv.org/abs/1909.08053.
K. Shuster, S. Poff, M. Chen, D. Kiela, and J. Weston. Retrieval augmentation reduces hallucination in
conversation. arXiv:2104.07567 [cs], Apr. 2021. URL http://arxiv.org/abs/2104.07567.
E. Strubell, A. Ganesh, and A. McCallum. Energy and policy considerations for deep learning in NLP.
In Association for Computational Linguistics, July 2019. URL https://aclanthology.org/
P19-1355.
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser,
and I. Polosukhin. Attention is all you need. In Advances in Neural Information Pro-
cessing Systems, 2017. URL https://proceedings.neurips.cc/paper/2017/file/
3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.
22Improving language models by retrieving from trillions of tokens
X. Wei and W. B. Croft. LDA-based document models for ad-hoc retrieval. In ACM SIGIR International
Conference on Research and Development in Information Retrieval, 2006. URL http://portal.
acm.org/citation.cfm?doid=1148170.1148204.
L. Weidinger, I. Gabriel, C. Griffin, M. Rauh, J. Uesato, J. Mellor, W. Isaac, P.-S. Huang, L. A. Hendricks,
M. Cheng, B. Balle, J. Haas, C. Biles, L. Rimell, W. Hawkins, M. Glaese, A. Kasirzadeh, Z. Kenton,
S. Brown, A. Birhane, T. Stepleton, G. Irving, and S. Legassick. Ethical and social risks of harm
from language models. arXiv submission, 2021.
D. Yogatama, C. de Masson d’Autume, and L. Kong. Adaptive semiparametric language models.
Transactions of the Association for Computational Linguistics, 9:362–373, 2021.
B. Zhang and R. Sennrich. Root mean square layer normalization. In Advances in Neural Information
Processing Systems, 2019. URL https://proceedings.neurips.cc/paper/2019/file/
1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf.
J. Zhang, M. Utiyama, E. Sumita, G. Neubig, and S. Nakamura. Guiding neural machine translation
with retrieved translation pieces. In Conference of the North American Chapter of the Association for
Computational Linguistics, 2018.
23Improving language models by retrieving from trillions of tokens
A. Datasets
We provide a full description of MassiveText and of our extract of recent Wikipedia articles.
A.1. Full description of MassiveText
The full break down of MassiveText by source and languages is given in Table 8. For a full description
and analysis of MassiveText, see Rae et al. (2021).
Source Language Token count (M)
Documents Sampling weight
Web En
Ru
Es
Zh
Fr
De
Pt
It
Sw
Ur 483,002
103,954
95,762
95,152
59,450
57,546
44,561
35,255
2,246
631 604,938,816
93,004,882
126,893,286
121,813,451
76,612,205
77,242,640
62,524,362
42,565,093
1,971,234
455,429 0.314
0.033
0.033
0.033
0.033
0.033
0.033
0.033
0.0044
0.0011
Books En 3,423,740 20,472,632 0.25
News En 236,918 397,852,713 0.1
Wikipedia En
De
Fr
Ru
Es
It
Zh
Pt
Ur
Sw 3,977
2,155
1,783
1,411
1,270
1,071
927
614
61
15 6,267,214
3,307,818
2,310,040
2,767,039
2,885,013
2,014,291
1,654,772
1,423,335
344,811
58,090 0.0285
0.003
0.003
0.003
0.003
0.003
0.003
0.003
0.0001
0.0004
Github - 374,952 142,881,832 0.05
Total - 5,026,463 1,792,260,998 1
Table 8 | MassiveText dataset. The final column indicates the sampling weight for each dataset
during training. For the retrieval database, the entire dataset is used, with the exception of books for
which we use a sub-sample of 4%.
A.2. Wikipedia September 2021
We create an evaluation dataset consisting of 23 Wikipedia articles that were added or heavily edited
in September 2021, after we collected our training dataset. In addition, we filter out articles that rely
too heavily on templated content, using the method detailed in §2.6 to identify articles with chunks
that have a high overlap with their neighbours. Fig. 10 show that little overlap remains between our
test dataset and the retrieved neighbours from the training dataset. The full list of included articles is
given in Table 9.
24Improving language models by retrieving from trillions of tokens
Table 9 | Full set of articles included in our Wikipedia Sept. 2021 evaluation dataset.
Megan Rohrer
Emma Raducanu
Ambra Sabatini
WhyDonate
The Juggernaut (company)
Angela Diaz
2020 Summer Paralympics
2021 Afghan protests
Rexh Xhakli
Julia Laskin
Cuijk
Ghoubet Wind Power Station
Aakashavaani
Junior Eurovision Song Contest 2021
Pavilion Bukit Jalil
Blake Desjarlais
2021 All-Ireland Senior Football Championship Final
Drift-barrier hypothesis
Venomics
Great Circle (novel)
Hurricane Ida
2021 Montenegrin episcopal enthronement protests
At War With the Silverfish
We first parse articles using mwparserfromhell 5 . We then remove sections with the following
titles: “references”, “external links”, “sources”, “further reading”, “see also”, “citations”, and “note”. In
the remaining sections, we remove Wikilinks and remove the following templates: “reflist”, “notelist”,
“notelist-ua”, “notelist-lr”, “notelist-ur”, and “notelist-lg”. We also exclude objects with the “ref ” or
“table” tag and clean the remaining text with the strip_code function. Finally, we concatenate the
title and all the sections and use \n\n to delimitate them.
B. Details on the retrieval architecture
We give details on the Retro architecture, and on the fine-tuning procedure we use for Retrofitting
existing language models.
B.1. Retro architecture and implementation
B.1.1. Feed-forward architecture
As mentioned in the main text, the overall encoder-decoder architecture is fully feed-forward. We start
with a sequence 𝑋 ∈ 𝕍 𝑛 = ( 𝐶 𝑢 ) 16 𝑢 6 𝑙 , and its pre-computed neighbours (Re t( 𝐶 𝑢 )) 16 𝑢 6 𝑙 and returns
logits in ℝ 𝑛 × |𝕍 | . Along with At t n, Ff w, Cca and C a operators introduced in the main text, we
define the decoder embedding layer E mb : 𝕍 𝑛 → ℝ 𝑛 × 𝑑 , the Sp l i t operator that extracts chunked
intermediary embeddings Sp l i t( 𝐻 ) , ( 𝐻 𝑢 ) 16 𝑢 6 𝑙 ∈ ℝ 𝑙 × 𝑚 × 𝑑 and the read-out layer Re ad : ℝ 𝑛 × 𝑑 →
ℝ 𝑛 × |𝕍 | . We then describe the forward pass in Algorithm 1. In addition to the usual Transformer ones,
Retro architecture hyperparameters involves the layer indices 𝑃 enc and 𝑃 , at which the encoder and
the decoder perform cross-attention.
B.1.2. Relative positional encoding in the chunked cross-attention layer
The Ca operator uses relative positional logits, that are computed from a specific relative distance
separating data tokens from retrieval tokens. Indeed, we expect any retrieval neighbour Re t ( 𝐶 𝑢 ) 𝑗 and
the chunk 𝐶 𝑢 to be relatively well aligned, and assume that they start at the same position. Therefore,
when computing C a( 𝐻 𝑢 + , 𝐸 𝑢 ), we set the distance between the data token 𝑖 ∈ [1 , 𝑙 ] of chunk 𝐶 𝑢 + and
5 https://github.com/earwig/mwparserfromhell
25Improving language models by retrieving from trillions of tokens
the retrieval token 𝑖 0 ∈ [1 , 2 𝑙 ] of Re t( 𝐶 𝑢 ) 𝑗 to be
𝑑 ( 𝑖, 𝑖 0 ) , 𝑖 − 𝑖 0 + 𝑙 − 1 .
(6)
When computing the encoder cross-attentions C a(Re t( 𝐶 𝑢 ) 𝑗 , 𝐻 𝑢 ), we set the distance between the
retrieval token 𝑖 0 ∈ [1 , 2 𝑙 ] and the data token 𝑖 ∈ [1 , 𝑙 ] to be
𝑑 enc ( 𝑖 0 , 𝑖 ) , 𝑖 0 − 𝑖.
(7)
Positional logits are obtained as a linear transform of a cosine vector computed from ( 𝑑 ( 𝑖, 𝑖 0 )) 𝑖,𝑖 0 , and
are added to content logits, as in a regular self-attention block.
B.1.3. Chunked cross-attention implementation
Our implementation of the Cca operator, shown in Listing 1, is based on a vectorized application of
a cross-attention layer. For simplicity, we omit the multi-head attention logic and use the simplest
Q,K,V attention. We omit relative positional logits computation, described above.
B.1.4. Optional sharing of embedding matrices
We use disjoint embeddings for the encoder and decoder by default, which allows us to use a different
dimensionality for the encoder (typically kept at 𝑑 E nc = 896) and for the decoder (that we scale up
to 𝑑 = 8192). It is possible to share the embeddings, with little difference in training, as we show in
the ablation section.
B.2. Baseline to Re t ro model fine-tuning
As shown in Fig. 5, we found that we were able to take a pre-trained baseline transformer and add
Retro through fine-tuning. In all cases, we froze all weights from pre-training and freshly initialised
the retrieval encoder and cross-attention weights. In all cases, the cross-attention is added every third
layer starting at layer six. The learning rate for the three smaller models was set to 2 × 10 −4 and
half that for the larger model. We experimented with allowing the entire model to resume training
during fine-tuning but consistently found that the best approach was to freeze the pre-trained model.
This kept the retrieval-off performance frozen whereas when all weights were tuned the retrieval off
performance would degrade.
C. Training details and hyperparameters
We provide the hyperparameters used in the various experiments of §4.
C.1. Language model pre-training
In Table 10, we show the hyperparameters of the different models we train. In all cases, we train for
419,430,400,000 training tokens. The three smaller models are trained with a batch size of 256 and
the largest model is trained with a batch size of 1024. The minimum learning rate is set to 0.1 times
the maximum learning rate, which is shown in Table 10. The learning rate is decayed using a cosine
cycle length that matches the total number of training tokens. All models are trained using AdamW
(Loshchilov and Hutter, 2019) with a weight decay parameter of 0.1. The learning rate linearly
increases from 10 −7 to the maximum learning rate over the first 750 steps of training. All models use
ZeRO to shard the optimiser state (Rajbhandari et al., 2020). Additional infrastructure details can be
found in Rae et al. (2021).
26Improving language models by retrieving from trillions of tokens
Listing 1 | Jax implementation of the chunked cross attention, simplified.
n
m
r
k
d
l =
=
=
=
=
=
128 # Sequence length
16 # Chunk length
32 # Retrieval length
4 # Number of neighbours
16 # Embedding size
n // m # Number of chunks
#
Q
K
V Parameters
= jnp.zeros((d, d))
= jnp.zeros((d, d))
= jnp.zeros((d, d))
def relative_positional_encodings(attending_length, attended_length):
# Classical relative positional encodings
...
def cross_attention(chunk, neighbour):
m, d = chunk.shape
r, d = neighbour.shape
queries = chunk @ Q
keys = neighbour @ K
logits = queries @ keys.T
values = neighbour @ V
return logits, values
def multi_neighbour_cross_attention(chunk, neighbours):
m, d = chunk.shape
k, r, d = neighbours.shape
logits, values = jnp.vectorize(cross_attention,
signature=’(m,d),(r,d)->(m,r),(r,d)’)(
chunk, neighbours)
assert logits.shape == (k, m, r)
assert values.shape == (k, r, d)
logits += relative_positional_encodings(m, r)[None, :, :]
logits = jnp.moveaxis(logits, 0, -1).reshape((m, r * k))
values = jnp.moveaxis(values, 0, 1).reshape((r * k, d))
return jax.nn.softmax(logits) @ values
def multi_chunk_cross_attention(observation, neighbours):
attending_chunks = jnp.pad(observation[m-1:],
((0, m - 1), (0, 0)),
mode=’constant’).reshape(l, m, d)
chunked_output = jnp.vectorize(multi_neighbour_cross_attention,
signature=’(m,d),(k,r,d)->(m,d)’)(
attending_chunks, neighbours)
assert chunked_output.shape == (l, m, d)
output = jnp.pad(chunked_output.reshape(n, d),
((m - 1, 0), (0, 0)),
mode=’constant’)[:n]
return output
observation = jnp.zeros((n, d)) # Input
neighbours = jnp.zeros((l, k, r, d))
h = multi_chunk_cross_attention(observation, neighbours)
assert h.shape == (n, d) # Output
27Improving language models by retrieving from trillions of tokens
Table 10 | Ret ro model hyperparameters, along with the size of the decoder.
Baseline 𝑑 𝑚𝑜𝑑𝑒𝑙 𝑑 𝑓 𝑓𝑤 # heads Head size # layers 𝑃 𝑃 Enc Max LR
247M
564M
1,574M
7,505M 896
1536
2048
4096 3584
6144
8192
16384 16
12
16
32 64
128
128
128 12
12
24
32 [6 , 9 , 12]
[6 , 9 , 12]
[9 , 12 , . . . , 24]
[9 , 12 , . . . , 32] [1]
[1]
[1]
[1] 2×10 −4
2×10 −4
2×10 −4
1×10 −4
Table 11 | Hyperparameters for the Wikitext103 experiments presented in Table 4. We use the same
learning rate schedule for the baseline and the Re t ro-fitting. For Re t ro-fitting, we reset the
schedule i.e. the schedule starts from step 0, not from step 35,000.
Model
Number of layers
𝑑
𝑑 Ff w
Key size
Value size
Number of heads
Training data Dataset
Sequence length
Batch size
Tokenizer vocabulary size
Optimisation optimiser
Adam’s 𝛽 1
Adam’s 𝛽 2
Adam’s 𝜀
Dropout rate
Learning rate start
Learning rate max
Learning rate min
Warmup steps
Cosine cycle steps
Overlapping proportion
Schedule
Evaluation
18
1024
4096
64
64
16
Wikitext103train
3072
128
128,000
Adam
0.9
0.95
1e-8
0.25
1e-7
2.5e-4
2e-5
4,000
100,000
87.5 %
C.2. Wikitext103 comparison
We provide more details on our Wikitext103 results presented in §4.1 and Table 4. We train a baseline
transformer on the Wikitext103 training set with the hyperparameters presented in Table 11. The
learning rate ramps linearly from 1 × 10 −7 to 2 . 5 × 10 −4 in the first 4,000 steps, then decays to
2 × 10 −5 at 100,000 steps using a cosine schedule. The baseline checkpoint at step 35,000 has the
lowest perplexity on Wikitext103 valid, of 21 . 58, for overlapping proportion of 75% (sliding window
evaluation that only uses probabilities for tokens that have at least 75% of the sequence length of
context, when available). We use this checkpoint for all our baseline and 𝑘 NN-LM numbers reported
in Table 4, except that Table 4 reports for an overlapping proportion of 87.5 %, which slightly lowers
the perplexity of our baseline to 21.53 on Wikitext103 valid.
We also use the 35,000 step baseline checkpoint as initialization for a Re t rofit, which otherwise
uses the same optimiser and schedule hyperparameters but only trains the new retrieval weights, as
explained in §4.2. Our best Re t rofit checkpoint has a Wikitext103 valid perplexity 18 . 46, when
retrieving from Wikipedia. We use this Re t ro checkpoint in Table 4 for all other retrieval sets. The
evaluation curves for our baseline and Re t rofit is shown if Fig. 7 (left). In this particular case,
28Improving language models by retrieving from trillions of tokens
because Wikitext103 is quite small, training a Re t ro model from scratch led to weaker results than
the baseline, at least when retrieving from Wikipedia, as we couldn’t find an effective way to mitigate
the increased over-fitting due to the additional weights of Re t ro.
We also re-implement 𝑘 NN-LM using the same tokenizer and dataset that we use for our base-
line and Retrofitting experiments. 𝑘 NN-LM has probabilities 𝑝 𝑘 NN-LM = 𝜆 𝑝 𝐿𝑀 + (1 − 𝜆 ) 𝑝 𝑘𝑁 𝑁 with
𝑝 𝑘𝑁 𝑁 ( 𝑛 𝑘 ) ∝ exp(− 𝛼𝑑 𝑘 ). To tune 𝜆 and 𝛼 , we begin with 𝛼 = 0 . 0012, which corresponds to the inverse
of the standard deviation of the norm of the embeddings that we use as keys and queries for 𝑘 NN-LM.
We find the best 𝜆 = 0 . 118. We then find the best 𝛼 = 0 . 00785 for that value of 𝜆 . Fig. 7 center and
right respectively show the perplexity of 𝑘 NN-LM as a function of 𝜆 and 𝛼 .
22 22 22
20 20 20
24 Baseline
24
18
0
20
40
60
1,000 steps
80
18
0.0
RETROfit
0.2
lambda
kNN-LM
24
18 10
0.4
4
10
3
10
alpha
2
10
1
Figure 7 | Wikitext103valid perplexities. Left: Baseline and Re t rofit (initialized from baseline’s
checkpoint at 35,000 steps) perplexities as a function of training steps. Center and right: 𝑘 NN-LM
perplexity as a function of 𝜆 (for 𝛼 = 0 . 0012) and 𝛼 (for 𝜆 = 0 . 12) respectively.
C.3. Retrofitting baseline models experiments
In Table 12, we give the hyperparameters used for Re t rofitting the models on Massive Text.
Table 12 | Hyperparameters for the Re t rofitting experiments
Model Layers with Re t ro-block ( 𝑃 )
172M
425M
1.5B
7.5B 3 rd
Every
Every 3 rd
Every 3 rd
Every 3 rd
from 6
from 6
from 6
from 6
Learning rate
10 −4
2 ×
2 × 10 −4
2 × 10 −4
1 × 10 −4
10 −5
→ 2 ×
→ 2 × 10 −5
→ 2 × 10 −5
→ 1 × 10 −5
Batch size
256
256
256
256
C.4. Question answering experiments
We fine-tune our 7.5B Re t ro model for 25,000 steps, using a batch size of 128, a learning rate
cosine scheduled from 10 −6 to 10 −7 , with a linear ramp of 750 steps. We use dropout in the decoder
only, as it performs better than using dropout in both the encoder and the decoder. Each neighbour
is formatted as title: {title}, source: {source}. We use the top 20 neighbours from
Dpr when training and evaluating.
29Improving language models by retrieving from trillions of tokens
Table 13 | Performance of Re t ro for different variants. Model performance on C4 evaluation set,
measured in bytes-per-bits, for a 247M parameter model trained with a 157 billion token schedule.
Ablation group Ablation
C4 eval bpb
Model Re t ro
No query conditioning
No CA positional encodings
Shared embeddings
6-layer encoder 0.822
0.829
0.826
0.823
0.821
Retrieval values Neighbours N
Continuations F
No retrieval 0.950
0.895
0.987
Training neighbours 1 training neighbours
4 training neighbours 0.858
0.847
Cross attention position CA top layer (1/12)
CA mid layer (6/12)
CA top layer (12/12)
CA all layers
CA every 3 from 1 0.827
0.823
0.831
0.860
0.823
D. Model ablations
We validate important design choices by evaluating what happens when we do not include them. We
use the 247M parameter model for all experiments and we train on a compressed 157 billion token
schedule for all ablation experiments. We describe results relative to the default settings presented in
the main text and recalled here. We report C4 evaluation loss at the end of the training process, and
also compares how the evaluation loss decrease versus the training time, measured relatively to the
baseline training time. Results are reported in Fig. 8 and Table 13.
Using relative encodings in cross-attention. Using relative encodings in cross-attention, as de-
scribed in §B.1.2, provides a pure improvement both in the number of steps to reach a given perfor-
mance and computational efficiency.
Conditioning the encoder on the previous chunk. Conditioning the encoder on the previous
chunk’s intermediate embeddings, as described in §B.1.1, provides a pure improvement both in term
of number of steps and computational efficiency.
Sharing embeddings. Sharing embeddings across the encoder and the decoder does not affect
performance. This motivates us using separate embeddings, as it allows to have a narrower encoder
than decoder as we scale up the decoder size.
Attending neighbours and their continuation. Re t ro models are trained by attending, for a
given chunk, to both the neighbours of the preceding chunk and their continuation in time. We
measure how training and evaluating Re t ro models on neighbours only and their continuation
only affects performance. Overall, attending to neighbours only provides 22% of the performance
improvement due to retrieval in Retro, while attending the future of the neighbours gives 56% of
30Improving language models by retrieving from trillions of tokens
0.88
0.86
RETRO
No CA positional encodings
0.88
0.86
RETRO
No query conditioning
0.88
0.86
0.84 0.84 0.84
0.82 0.82 0.82
0.88 1.2
0.88
0.86
RETRO: 2 layer encoder
6 layer encoder
1.1
0.86
0.84 0.84
0.82
0.0 0.2 0.4 0.6 0.8 1.0 1.2
Training time (relative to baseline) 0.82
RETRO: 2 training nei.
1 training nei.
4 training nei.
0.840
0.835
0.830
0.825
1.0
RETRO: distinct embeddings
Shared embeddings
RETRO: retrieve [N,F]
Neighbours N
Continuations F
No retrieval
0.9
0.8
0.0 0.2 0.4 0.6 0.8 1.0 1.2
Training time (relative to baseline)
RETRO: CA every 3 from 6
CA top layer (1/12)
CA mid layer (6/12)
CA top layer (12/12)
CA all layers
CA every 3 from 1
0.820
0.0 0.2 0.4 0.6 0.8 1.0 1.2
Training time (relative to baseline)
Figure 8 | Computational efficiency for different variants. We report the training curves plotting
C4 evaluation bytes per bits against time, relative to the time taken to train the baseline Re t ro
model. Overall, our design choices are optimal in term of computational efficiency.
the performance. Attending to both neighbours and their continuation is the most efficient choice
both in term of final performance and training efficiency.
Training a deeper encoder. All models in the text use a relatively small Re t ro encoder. We
experimented with a 3× deeper encoder. We found that this resulted in a tiny decrease in loss– 0.15%
at the cost of a larger training time (+20%). Overall, using a shallow encoder is the best choice in
term of training efficiency.
Training with multiple neighbours. We measure the effect of training on a single retrieved neigh-
bour, as well as training on 4 neighbours (Re t ro uses 2 neighbours in training). Training on a
single neighbour results in a large decrease in performance, while training on 4 neighbours does not
give substantial performance improvement at the end of training, but induces a large computational
overhead. Overall, we find that using 2 neighbours is the best choice in term of training efficiency.
Furthermore, evaluation can be done with additional neighbours.
Frequency of cross-attention. We measure how the frequency of cross-attention in the decoder
affects performance. Overall, attending only once at the top or the bottom layer is a bad choice, while
attending once on a mid-depth layer is relatively sound. We choose to have cross-attention every 3
layer as this provides a good trade-off between performance and run-time.
31Improving language models by retrieving from trillions of tokens
E. Qualitative experiments
We illustrate the usage of Retro models by looking at the perplexity of evaluation samples and by
producing samples autoregressively.
E.1. Inspecting neighbours and perplexities on evaluation data
To build an intuition of what kind of information is leveraged by Re t ro models, we suggest to
have a closer look at a few evaluation documents and the corresponding retrieved data in Tables
16, 17, 18 and 19. In these tables, the 4 rows corresponds to the first 4 chunks of the documents.
The left-most column shows the chunk 𝐶 𝑢 from the document being evaluated, where each token is
coloured by the negative cross entropy loss difference 𝐿 Re t ro [Of f ] − 𝐿 Re t ro , a positive value, coloured
in yellow, indicates that Re t ro performs better when it has access to neighbours data. The second
columns also shows the evaluated chunk 𝐶 𝑢 but where each token 𝑖 is coloured by the length of the
longest common prefix (LCP) with the preceding neighbours, i.e. the largest integer 𝑗 such that
the prefix ( 𝑥 𝑖 − 𝑗 −1 , . . . , 𝑥 𝑖 ) also appears in Re t( 𝐶 𝑢 −1 ). Conversely, columns three and four show the
first two neighbours and their continuation, respectively [ 𝑁 𝑢 1 , 𝐹 𝑢 1 ] and [ 𝑁 𝑢 2 , 𝐹 𝑢 2 ] coloured by LCP with
subsequent chunk 𝐶 𝑢 +1 . LCP colouring helps to visually identify where the evaluated document
overlaps the retrieved data. Note that the first chunk, 𝐶 1 , in the second column is not coloured as
it does not have any preceding neighbours to compute LCP with. Similarly, we do not show the
neighbours of the fourth chunk, as these are not used to condition any of the first four chunks.
Our qualitative analysis exhibits two major behaviors.
Firstly, we observe that sometimes, specific facts in 𝐶 𝑢 can be extracted from the preceding
neighbours Re t( 𝐶 𝑢 −1 ) and that this can correspond to significant reduction in loss from the Ret ro
model for the corresponding tokens. Some examples of such behavior include the journal name
Publishers Weekly in Table 16, the football team name Tyrone in Table 17 or the event dates 25 August
to 6 September 2020 in Table 18. In these three examples, the evaluated data consists of recent
Wikipedia articles written in September 2021, after we built our retrieval dataset (see section §A.2).
Yet, relevant information to predict this new data was available in the pre-existing retrieval data and
the Retro model seems to be able to correctly leverage it.
On the other hand, we also observe that some of the evaluation data can partially leak in our
training and retrieval data, despite the use of deduplication. Re t ro can dramatically exploit such
leakage. Table 19 illustrates this behavior, where the chunks 𝐶 2 and 𝐶 3 largely overlaps Re t( 𝐶 1 ) and
Ret( 𝐶 2 ) respectively, up to small formatting differences, which leads to much lower Re t ro loss for
all the corresponding tokens. Fig. 6 shows that it is possible to quantify how much of the Re t ro loss
reduction is due to each of these two behaviors, by filtering out evaluation chunks that overlaps with
the retrieval set.
E.2. Inspecting samples
We can follow the same procedure as above on samples generated using Retro models, in order to
better understand where retrieval data had an influence on sampling. We show examples of samples
obtained using the 7.5B Re t ro model in Table 6, 7, 20 and 21.
E.3. Neighbour quantification
To quantify a notion of distance between the source document and the retrieved chunks, we can ask
the distance between source articles when retrieving only from Wikipedia. Consonni et al. (2019)
32Improving language models by retrieving from trillions of tokens
Figure 9 | Wikipedia link-distance between retrieved articles. For each sequences, chunk combina-
tion we compute the link distance between the target and the top-5 neighbours using only Wikipedia.
The rank shows the relative neighbour distance, where rank-1 is the first neighbour and rank 5 is
the fifth. The different colours represent link distance. Because we do not retrieve from the same
document, 1 is the smallest value. We find, on average, the distance between random articles with a
path between them is over 5.0
provides a Wikipedia link dataset which, for each article, contains a list of neighbouring articles.
Using this, we construct a directed graph and compute the distance from one page to another. In
Fig. 9 we compute the link-distance between training sequences and the retrieved neighbours. We
find that retrieved documents tend to be from articles that are quite close to the article containing
the target. Furthermore, we find that on average the distance increases with rank, suggesting that
our neighbours are both useful and that the order is reasonable. This provides confidence for our
larger-scale experiments where document distance is less well defined.
F. Complementary quantitative results
We report tables corresponding to quantitative figures of the main text, as well as further filtered
language model results on the Pile.
F.1. Main text datasets
We report the performance of Re t ro and baseline models, measured in bits-per-bytes on evaluation
set, in Table 14.
F.2. The Pile
In Fig. 4, we compare Retro against Jurassic-1 (Lieber et al., 2021). The full bits-per-bytes results
are reported in Table 15.
F.3. Filtered results
Distribution of leaked chunks in our main evaluation sets. We evaluate leakage between the
evaluation sets and the training set by measuring the proportion of evaluation chunks with a certain
33Improving language models by retrieving from trillions of tokens
Table 14 | Full results for the main language modelling datasets. First three sets of rows correspond
to Fig. 1, last set of rows to Fig. 3.
172M Baseline
425M 1.5B 7.5B 172M Re t ro [Off]
425M 1.5B 7.5B 172M 0.98 0.92 0.84 0.78 0.98 0.92 0.84 0.78 0.82 0.77 0.71 0.66
C4 Eval bpb (900B)
C4 Eval bpb (360B)
C4 Eval bpb (180B)
C4 Eval bpb (90B)
C4 Eval bpb (36B)
C4 Eval bpb (18B)
C4 Eval bpb (9B)
C4 Eval bpb (4B)
C4 Eval bpb (2B) -
-
-
-
-
-
-
-
- -
-
-
-
-
-
-
-
- -
-
-
-
-
-
-
-
- -
-
-
-
-
-
-
-
- -
-
-
-
-
-
-
-
- -
-
-
-
-
-
-
-
- -
-
-
-
-
-
-
-
- -
-
-
-
-
-
-
-
- 0.88
0.92
0.94
0.95
0.96
0.96
0.96
0.97
0.97 0.83
0.87
0.89
0.89
0.90
0.91
0.91
0.91
0.91 0.76
0.80
0.81
0.82
0.83
0.83
0.83
0.84
0.84 0.71
0.74
0.75
0.76
0.77
0.77
0.77
0.78
0.78
C4 Eval bpb ( 𝑘 = 1)
C4 Eval bpb ( 𝑘 = 2)
C4 Eval bpb ( 𝑘 = 3)
C4 Eval bpb ( 𝑘 = 4)
C4 Eval bpb ( 𝑘 = 5)
C4 Eval bpb ( 𝑘 = 10)
C4 Eval bpb ( 𝑘 = 20)
C4 Eval bpb ( 𝑘 = 30)
C4 Eval bpb ( 𝑘 = 40)
C4 Eval bpb ( 𝑘 = 50)
C4 Eval bpb ( 𝑘 = 60)
C4 Eval bpb ( 𝑘 = 70)
C4 Eval bpb ( 𝑘 = 80)
C4 Eval bpb ( 𝑘 = 90)
C4 Eval bpb ( 𝑘 = 100) -
-
-
-
-
-
-
-
-
-
-
-
-
-
- -
-
-
-
-
-
-
-
-
-
-
-
-
-
- -
-
-
-
-
-
-
-
-
-
-
-
-
-
- -
-
-
-
-
-
-
-
-
-
-
-
-
-
- -
-
-
-
-
-
-
-
-
-
-
-
-
-
- -
-
-
-
-
-
-
-
-
-
-
-
-
-
- -
-
-
-
-
-
-
-
-
-
-
-
-
-
- -
-
-
-
-
-
-
-
-
-
-
-
-
-
- 0.84
0.83
0.82
0.82
0.82
0.82
0.82
0.82
0.83
0.83
0.84
0.84
0.85
0.85
0.85 0.79
0.78
0.78
0.77
0.77
0.77
0.77
0.77
0.77
0.78
0.78
0.79
0.79
0.79
0.79 0.73
0.72
0.71
0.71
0.71
0.71
0.71
0.71
0.71
0.71
0.72
0.72
0.73
0.73
- 0.67
0.67
0.66
0.66
0.66
0.66
0.66
0.65
0.65
0.66
0.66
0.66
0.66
0.66
0.67
0.42
0.69
25.62
0.85 0.51
0.63
19.29
0.78 0.61
0.56
13.98
0.71 0.69
0.52
10.65
0.65 0.47
0.68
25.88
0.86 0.54
0.64
19.78
0.79 0.63
0.57
13.89
0.71 0.70
0.51
10.40
0.65 0.52
0.66
3.32
0.79 0.60
0.61
2.96
0.73 0.67
0.55
2.53
0.66 0.73
0.50
2.22
0.61
C4 Eval bpb
Lambada Accuracy
Curation Corpus bpb
Wikitext103 Perplexity
Wikipedia Sept. 2021 bpb
Ret ro[On]
425M 1.5B
7.5B
overlap 𝑟 ( 𝐶 ). We show histograms in Fig. 10. We can see that 𝐶 4 has some slight overlaps between
train and evaluation. Similarly, chunks of Wikitext103 appear in the training set despite having
removed the actual Wikitext103 evaluation documents from the training set. On the other hand, our
Wikipedia September 21 dataset shows almost no leakage (data being original documents that did
not exist at training data creation), and neither does Curation Corpus.
Filtered results on the Pile. We report chunk overlap distribution and filtered performance curves
on the Pile in Fig. 12 and Fig. 11, respectively. The qualitative interpretation of the filtered curves
is the same: Re t ro models exploit leakage more, but the performance improvement they provide
remains significant even on original chunks that haven’t been observed in the training set.
34Improving language models by retrieving from trillions of tokens
Table 15 | Full results on The Pile, measured in bits-per-bytes. Jurassic-1 and GPT-3 numbers are
taken from Lieber et al. (2021). Gopher numbers are taken from Rae et al. (2021).
Subset
arxiv
books3
dm_mathematics
freelaw
github
gutenberg_pg_19
hackernews
nih_exporter
opensubtitles
philpapers
pile_cc
pubmed_abstracts
pubmed_central
stackexchange
ubuntu_irc
uspto_backgrounds
GPT-3 Jurassic-1 Gopher 7.5B Re t ro
0.742
0.792
1.177
0.576
0.420
0.803
0.971
0.650
0.974
0.760
0.771
0.639
0.588
0.714
1.200
0.603 0.838
0.802
1.371
0.612
0.645
1.163
0.975
0.612
0.932
0.723
0.698
0.625
0.690
0.773
0.946
0.566 0.680
0.835
1.037
0.514
0.358
0.890
0.869
0.590
0.879
0.742
0.669
0.587
0.579
0.655
0.857
0.537 0.641
0.706
1.135
0.506
0.367
0.652
0.888
0.590
0.894
0.682
0.688
0.578
0.512
0.638
1.081
0.545 0.714
0.653
1.164
0.499
0.199
0.400
0.860
0.635
0.930
0.699
0.626
0.542
0.419
0.624
1.178
0.583
Curation Corpus
Wikitext103
Wikipedia Sept 2021
C4
7B Baseline (Ours)
0%
50%
Eval/train chunk overlap
100%
0%
50%
Eval/train chunk overlap
100%
0%
50%
Eval/train chunk overlap
100%
0%
50%
Eval/train chunk overlap
100%
Figure 10 | Distribution of the overlap between evaluation and train chunks for C4, Curation
Corpus, Wikitext103 and Wikipedia Sept. 2021.
35Improving language models by retrieving from trillions of tokens
172M
arxiv
425M
1.5B
bookcorpus2
europarl
1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
freelaw
0.8
1.4
0.7
1.2
0.6
1.0
0.5
0.8
nih_exporter
1.0
0.75
0.9
0.70
0.8
0.7
0.65
philpapers
1.0
pile_cc
1.0
0.9
0.6 0.6
0.5 0.55
pubmed_central
0.8
0.7
0.6
0.5
0.4
0.3
50%
100%
Max allowed eval/train overlap
uspto_backgrounds
0.75
1.2 0.70
0.8 1.0 0.65
0.7 0.8 0.60
pubmed_abstracts
ubuntu_irc
1.6
0.6
100% 12.5%
Max allowed eval/train overlap
openwebtext2
1.0
0.9
0.8
0.7
0.6
0.5
0.4
12.5%
1.4
50%
0.2
opensubtitles
0.70
0.60
0.6
12.5%
0.4
0.75
0.6
0.9
gutenberg_pg_19
0.8
0.5
1.0
github
0.8 0.65
1.1
0.9
1.0
0.7
stackexchange
1.0
1.0 0.6
0.4
1.1
0.80
0.8
0.7
1.2
0.85
0.9
0.8
1.3
1.2
1.1
1.0
0.9
0.8
0.7
0.6
0.80
1.1
dm_mathematics
1.4
0.2
hackernews
1.2
RETRO [ON]
0.4
0.4
0.6
Baseline
books3
1.0
0.9
0.8
0.7
0.6
0.5
0.4
1.0
0.9
0.8
0.7
0.6
0.5
0.4
7.5B
50%
0.55
100% 12.5%
Max allowed eval/train overlap
50%
100%
Max allowed eval/train overlap
Figure 11 | Filtered evaluation losses on the Pile, with baseline Transformers and Re t ro.
36Improving language models by retrieving from trillions of tokens
bookcorpus2 books3 dm_mathematics
europarl freelaw github gutenberg_pg_19
hackernews nih_exporter opensubtitles openwebtext2
philpapers pile_cc pubmed_abstracts pubmed_central
stackexchange ubuntu_irc uspto_backgrounds arxiv
0%
50%
Eval/train chunk overlap
100%
0%
50%
Eval/train chunk overlap
100%
0%
50%
Eval/train chunk overlap
100%
Figure 12 | Distribution of the overlap between evaluation and train chunks for the Pile evaluation
sets.
37Improving language models by retrieving from trillions of tokens
Table 16 | Great Circle (novel), from Wikipedia September 21. The article is about a recent novel and chunks
𝐶 3 and 𝐶 4 are specifically about its reception. The name Publishers Weekly of the journal that reviewed the
novel appears both in the neighbours [ 𝑁 3 1 , 𝐹 3 1 ] , [ 𝑁 3 2 , 𝐹 3 2 ] of chunk 𝐶 3 and in the subsequent chunk 𝐶 4 , where the
loss for those tokens is significantly reduced by Re t ro.
𝐶 𝑢 colored by loss difference
𝐿 Re t ro [Of f ] − 𝐿 Re tro 6 −0 . 5 , = 0 , > 0 . 5 𝐶 𝑢 colored by LCP with Ret ( 𝐶 𝑢 −1)
LCP = 0 , 1 , 2 , 3 , 4 , > 5 [ 𝑁 𝑢 1 , 𝐹 𝑢 1 ] colored by LCP with 𝐶 𝑢 +1
LCP = 0 , 1 , 2 , 3 , 4 , > 5 [ 𝑁 𝑢 2 , 𝐹 𝑢 2 ] colored by LCP with 𝐶 𝑢 +1
LCP = 0 , 1 , 2 , 3 , 4 , > 5
Great Circle (novel)Great Circle i
s a 2021 novel by Maggie Shipstead,
published on May 4, 2021, by Alfred
A. Knopf.The novel has been shortl
isted for the 2021 Booker Prize.Sy
nopsis The novel consists of two pa
rallel narratives about two fictiona
l women. One is Great Circle (novel) Great Circle i
s a 2021 novel by Maggie Shipstead,
published on May 4, 2021, by Alfred
A. Knopf. The novel has been shortl
isted for the 2021 Booker Prize. Sy
nopsis The novel consists of two pa
rallel narratives about two fictiona
l women. One is The Dutch House (novel)The Dutch H
ouse is a 2019 novel by Ann Patchett
. It was published by Harper on Sept
ember 24, 2019. It tells the story o
f a brother and sister over the cour
se of five decades.The novel was a
finalist for the 2020 Pulitzer Priz
e for Fiction.PlotThe Dutch House
is a mansion located in Elkins Park
, Pennsylvania, a suburb of Philadel
phia. It was built in 1922 by the Va
nHoebeek family, a husband and wife
originally from the Netherlands who
made their fortune in the tobacco in
dustry. Cyril Conroy, a self-made re
al estate mogul The Dutch House (novel)The Dutch H
ouse is a 2019 novel by Ann Patchett
. It was published by Harper on Sept
ember 24, 2019. It tells the story o
f a brother and sister over the cour
se of five decades.[2]The novel wa
s a finalist for the 2020 Pulitzer P
rize for Fiction.[3]Plot[edit]Th
e Dutch House is a mansion located i
n Elkins Park, Pennsylvania, a subur
b of Philadelphia. It was built in 1
922 by the VanHoebeek family, a husb
and and wife originally from the Net
herlands who made their fortune in t
he tobacco industry. Cyril Conroy, a
self-
about the disappeared 20th-century
aviator Marian Graves, while the oth
er is about the struggling 21st-cent
ury Hollywood actress Hadley Baxter,
who is attempting to make a film ab
out Marian. Hadley’s narrative is to
ld in the first-person, while Marian
’s sections are told in the third-pe
rson about the disappeared 20th-century
aviator Marian Graves, while the oth
er is about the struggling 21st-cent
ury Hollywood actress Hadley Baxter,
who is attempting to make a film ab
out Marian. Hadley’s narrative is to
ld in the first-person, while Marian
’s sections are told in the third-pe
rson on becoming a filmmaker. She has fo
und a subject for her film project,
an obscure African American actress
credited only as “the watermelon wom
an” in old Hollywood films, and the
subsequent film recounts her search
for this woman even as it covers, in
the manner of the earlier Dunyement
aries, Dunye’s friendships and her l
ove life. InThe Watermelon Woman, D
unye makes the film she set out to m
ake in 1990 about African American w
omen artists, a film that both inven
ts an artistic predecessor with whom
she can identify and also “finds” C
heryl herself as the artist that she
seeks. As Dunye identifies herself based closely on her own youthful e
xperiences. (She plans the film to b
e the first of two parts, the second
dealing with the aftermath of the f
irst’s events.) Byrne plays a young
film student named Julie (Hogg’s ava
tar), who starts her artistic educat
ion with high hopes of making a movi
e about a boy named Tony, living in
working-class Sunderland, who adores
his mother — “is almost obsessed wi
th her,” as eager Julie tells her ad
visers. Her idealism is evident from
the start.The advisers are skepti
cal, and no wonder; Julie’s family i
s posh, with a comfortable country e
state and
.Reception Great Circle received
very favorable reviews, with a cumul
ative "Rave" rating at the review ag
gregator website Book Marks, based o
n 22 book reviews from mainstream li
terary critics. The novel debuted at
number fourteen on The New York Tim
es Hardcover fiction best-seller lis
t for the week ending May .Reception Great Circle received
very favorable reviews, with a cumul
ative "Rave" rating at the review ag
gregator website Book Marks, based o
n 22 book reviews from mainstream li
terary critics. The novel debuted at
number fourteen on The New York Tim
es Hardcover fiction best-seller lis
t for the week ending May first edition hardcoverReception
The novel debuted at number one on T
he New York Times fiction best-selle
r list. As of the week ending Februa
ry 20, 2021, the novel has spent 38
weeks on the list.At the review ag
gregator website Book Marks, which a
ssigns individual ratings to book re
views from mainstream literary criti
cs, the novel received a cumulative
"Rave" rating based on 38 reviews, w
ith only one "mixed" review. Publish
ers Weekly wrote, "Bennett renders h
er characters and their struggles wi
th great compassion, and explores th
e complicated state of mind that Ste
lla finds herself in while passing a
s white." In its The book also debuted at number tw
o on The New York Times Hardcover No
nfiction best-sellers list on July 2
8, 2019.[5] It spent eleven weeks on
the list.[6]Reception[edit]At t
he review aggregator website Book Ma
rks, which assigns individual rating
s to book reviews from mainstream li
terary critics, the book received a
cumulative "Positive" rating based o
n 29 reviews: 12 "Rave" reviews, 6 "
Positive" reviews, 9 "Mixed" reviews
, and 2 "Pan" reviews.[7]Publisher
s Weekly gave the book a mixed revie
w, writing, "Unfortunately, all thre
e
8, 2021. Critics praised the novel
for sustaining its length and for Sh
ipstead’s research and intricate nov
el structure for perfectly interweav
ing the parallel narratives, despite
the time and circumstances separati
ng them.In its starred review, Pub
lishers Weekly wrote, "Shipstead man
ages to portray both Marian’s and Ha
dley’s 8, 2021. Critics praised the novel
for sustaining its length and for Sh
ipstead’s research and intricate nov
el structure for perfectly interweav
ing the parallel narratives, despite
the time and circumstances separati
ng them.In its starred review, Pub
lishers Weekly wrote, "Shipstead man
ages to portray both Marian’s and Ha
dley’s
38Improving language models by retrieving from trillions of tokens
Table 17 | All-Ireland Senior Football Championship Final, from Wikipedia September 21. The name of
the team Tyrone appears both in the second neighbours [ 𝑁 1 2 , 𝐹 1 2 ] of chunk 𝐶 1 and in the subsequent chunk 𝐶 2 ,
where the loss for those tokens is significantly reduced by Re t ro.
𝐶 𝑢 colored by loss difference
𝐿 Re t ro [Of f ] − 𝐿 Re tro 6 −0 . 5 , = 0 , > 0 . 5 𝐶 𝑢 colored by LCP with Ret ( 𝐶 𝑢 −1)
LCP = 0 , 1 , 2 , 3 , 4 , > 5 [ 𝑁 𝑢 1 , 𝐹 𝑢 1 ] colored by LCP with 𝐶 𝑢 +1
LCP = 0 , 1 , 2 , 3 , 4 , > 5 [ 𝑁 𝑢 2 , 𝐹 𝑢 2 ] colored by LCP with 𝐶 𝑢 +1
LCP = 0 , 1 , 2 , 3 , 4 , > 5
2021 All-Ireland Senior Football Cha
mpionship FinalThe 2021 All-Irelan
d Senior Football Championship Final
was the 134th final of the All-Irel
and Senior Football Championship and
the culmination of the 2021 All-Ire
land Senior Football Championship. T
he match was played at Croke Park in
Dublin on 11 September 2021. It was
originally scheduled 2021 All-Ireland Senior Football Cha
mpionship Final The 2021 All-Irelan
d Senior Football Championship Final
was the 134th final of the All-Irel
and Senior Football Championship and
the culmination of the 2021 All-Ire
land Senior Football Championship. T
he match was played at Croke Park in
Dublin on 11 September 2021. It was
originally scheduled 2018 All-Ireland Senior Football Cha
mpionship FinalThe 2018 All-Irelan
d Senior Football Championship Final
was the 131st final of the All-Irel
and Senior Football Championship and
the culmination of the 2018 All-Ire
land Senior Football Championship in
Gaelic football. The match was play
ed at Croke Park in Dublin on 2 Sept
ember 2018.[3]It was the second ti
me the teams had met in the final; D
ublin won the first encounter in 199
5.The final was shown live in Irel
and on RTÉ Two as part of The Sunday
Game live programme, presented by M
ichael Lyster from Croke Park, with
studio analysis from Joe Brolly, 2018 All-Ireland Senior Football Cha
mpionship FinalThe 2018 All-Irelan
d Senior Football Championship Final
was the 131st final of the All-Irel
and Senior Football Championship and
the culmination of the 2018 All-Ire
land Senior Football Championship in
Gaelic football. The match was play
ed at Croke Park in Dublin on 2 Sept
ember 2018.It was the second time
the teams had met in the final; Dubl
in won the first encounter in 1995.
It was the third consecutive year th
at a team qualified under the system
of second chances introduced in 200
1; Tyrone qualified despite defeat i
n its provincial championship.Dubl
in won the final by a margin of six
points
for 28 August but had to be postpon
ed by two weeks when the – semi-fina
l was postponed due to a COVID-19 ou
tbreak. Ulster champions Tyrone took
on Connacht champions Mayo, in what
was their first ever meeting in a f
inal, winning their 4th title after
a 2–14 to 0–15 win. Mayo lost for 28 August but had to be postpon
ed by two weeks when the – semi-fina
l was postponed due to a COVID-19 ou
tbreak. Ulster champions Tyrone took
on Connacht champions Mayo, in what
was their first ever meeting in a f
inal, winning their 4th title after
a 2–14 to 0–15 win. Mayo lost game 23–23 after extra time, howeve
r Ulster progressed under the compet
ition rules as they scored three tir
es in the match against Leinster’s t
wo. The semi-finals took place in mi
d November and saw both the away tea
ms win, as Ulster beat Glasgow and E
dinburgh beat Connacht. The final wa
s held on Saturday December 20 at Mu
rrayfield Stadium and saw Ulster bea
t Edinburgh 21–27 to win the Celtic
Cup.2004–05 seasonThe format of
the competition was changed for the
second edition of the competition. T
he competition was moved to April an
d May to run after the conclusion of
the Celtic League competition, with
only eight with a last-ditch plan of action –
play the Munster/Ulster Semi-Final o
n March 16th, with the winners to pl
ay Connacht in the following day’s F
inal.On March 16th then Munster ha
d an easy win over Ulster (9-07 to 0
-00) but thankfully for the Munster
players, the pitch cut up so badly d
uring the game, it was decided to po
stpone the following day’s hurling F
inal (until Easter Sunday) with the
football Final going ahead on its ow
n on St. Patrick’s Day.Less than a
week later, on March 23rd, seven
their 11th consecutive final since
1989, losing 6 finals in 9 years, wi
th this latest defeat on an identica
l scoreline to 2020, when Mayo lost
to Dublin.Background were aiming
to win their fourth title and first
All-Ireland since 1951. Since then,
they had lost ten finals (1989, 1996
, 1997, 2004, 2006, their 11th consecutive final since
1989, losing 6 finals in 9 years, wi
th this latest defeat on an identica
l scoreline to 2020, when Mayo lost
to Dublin.Background were aiming
to win their fourth title and first
All-Ireland since 1951. Since then,
they had lost ten finals (1989, 1996
, 1997, 2004, 2006, 1-16 to 0-15 winners to qualify for
their 10th league final in the past
13 years.They have won seven of t
heir previous league finals under Co
dy since 2002, losing the other two
to Waterford (2007 ) and Dublin (201
1 ).Despite the defeat there were
some distinct positives from a Galwa
y perspective- most notably the soli
d displays of Daithí Burke at centre
-back, Joseph Cooney at wing-back an
d Ronan Burke at full-back. Colm Cal
lanan continued his excellent form i
n goal and also hit a stunning free
from distance.Indeed it was not th
e Galway defence that was the proble
m which Dublin won by 0-12 to 0-9.D
ublin are going for an unprecedented
fourth successive Championship win
over Kerry. Prior to their current r
un, which started with the 2011 All-
Ireland final, they had only managed
two consecutive victories over them
on two separate occasions - 1909 an
d ’24, 1976 and ’77.The longest wi
nning sequence in the rivalry was se
t by Kerry between 1941 and 1975, wh
en they won each of the six Champion
ship meetings. Kerry went nine games
unbeaten between 1978 and 2009, wit
h four victories either side of a dr
amatic draw at the quarter-final sta
ge in Thurles in 2001.Sunday will
mark their 11th
2012, 2013, 2016, 2017, 2020). app
eared in their seventh final, winnin
g on three occasions in 2003, 2005 a
nd 2008.This final was the fifth to
be contested by county teams from C
onnacht and Ulster, the other finals
were 1925 (Galway beat Cavan), 1943
(Roscommon beat Cavan), 1948 (Cavan
beat 2012, 2013, 2016, 2017, 2020). app
eared in their seventh final, winnin
g on three occasions in 2003, 2005 a
nd 2008.This final was the fifth to
be contested by county teams from C
onnacht and Ulster, the other finals
were 1925 (Galway beat Cavan), 1943
(Roscommon beat Cavan), 1948 (Cavan
beat
39Improving language models by retrieving from trillions of tokens
Table 18 | 2020 Summer Paralympics, from Wikipedia September 21. The original dates of the event,
25 August to 6 September 2020, appears both in the neighbors [ 𝑁 1 1 , 𝐹 1 1 ] , [ 𝑁 1 2 , 𝐹 1 2 ] of chunk 𝐶 1 and in the
subsequent chunk 𝐶 2 , where the loss for those tokens is significantly reduced by Re t ro. Interestingly, in this
case, the neighbors were written at a time when the event hadn’t yet been postponed.
𝐶 𝑢 colored by loss difference
𝐿 Re t ro [Of f ] − 𝐿 Re tro 6 −0 . 5 , = 0 , > 0 . 5 𝐶 𝑢 colored by LCP with Ret ( 𝐶 𝑢 −1)
LCP = 0 , 1 , 2 , 3 , 4 , > 5 [ 𝑁 𝑢 1 , 𝐹 𝑢 1 ] colored by LCP with 𝐶 𝑢 +1
LCP = 0 , 1 , 2 , 3 , 4 , > 5 [ 𝑁 𝑢 2 , 𝐹 𝑢 2 ] colored by LCP with 𝐶 𝑢 +1
LCP = 0 , 1 , 2 , 3 , 4 , > 5
2020 Summer ParalympicsThe , brand
ed as the Tokyo 2020 Paralympic Game
s, was an international multi-sport
parasports event held from 24 August
to 5 September 2021 in Tokyo, Japan
. They were the 16th Summer Paralymp
ic Games as organized by the Interna
tional Paralympic Committee (IPC). 2020 Summer Paralympics The , brand
ed as the Tokyo 2020 Paralympic Game
s, was an international multi-sport
parasports event held from 24 August
to 5 September 2021 in Tokyo, Japan
. They were the 16th Summer Paralymp
ic Games as organized by the Interna
tional Paralympic Committee (IPC). pics Games.* The 2020 Summer Paraly
mpics are an upcoming major internat
ional multi-sport event for athletes
with disabilities governed by the I
nternational Paralympic Committee. S
cheduled as the 16th Summer Paralymp
ic Games, it is planned to be held i
n Tokyo, Japan from 25 August to 6 S
eptember 2020.3. 2019 BWF Para-Bad
minton World Championships- The 20
19 BWF Para-Badminton World Champion
ships was held from 20 to 25 August
2019 in Basel, Switzerland.- Men’s
event: Gold Medal: Pramod Bhagat in
Singles SL3 Event and Pramod Bhagat
and Manoj Originally scheduled to take place f
rom 25 August to 6 September 2020, i
n March 2020 both the 2020 Summer Ol
ympics and Paralympics were postpone
d by one year due to the COVID-19 pa
ndemic, with the rescheduled Games s
till referred to as Tokyo 2020 for m
arketing and branding purposes. As
with the Olympics, the Games were la
rgely held behind Originally scheduled to take place f
rom 25 August to 6 September 2020, i
n March 2020 both the 2020 Summer Ol
ympics and Paralympics were postpone
d by one year due to the COVID-19 pa
ndemic, with the rescheduled Games s
till referred to as Tokyo 2020 for m
arketing and branding purposes. As
with the Olympics, the Games were la
rgely held behind closed doors with no outside specta
tors due to a state of emergency in
the Greater Tokyo Area and other pre
fectures. The Games were the second
Summer Paralympics hosted by Tokyo s
ince 1964, and the third Paralympics
held in Japan overall since the 199
8 Winter Paralympics in Nagano. Th
e Games featured closed doors with no outside specta
tors due to a state of emergency in
the Greater Tokyo Area and other pre
fectures. The Games were the second
Summer Paralympics hosted by Tokyo s
ince 1964, and the third Paralympics
held in Japan overall since the 199
8 Winter Paralympics in Nagano. Th
e Games featured 539 medal events in 22 sports, with
badminton and taekwondo both making
their Paralympic debut to replace f
ootball 7-a-side and sailing. China
topped the medal table for the fifth
consecutive Paralympics, with 96 go
lds and 207 total medals. Great Brit
ain finished second for the ninth t
ime, 539 medal events in 22 sports, with
badminton and taekwondo both making
their Paralympic debut to replace f
ootball 7-a-side and sailing. China
topped the medal table for the fifth
consecutive Paralympics, with 96 go
lds and 207 total medals. Great Brit
ain finished second for the ninth t
ime, once submitted.This process was u
ndertaken following the postponement
of the Tokyo 2020 Games due to the
COVID-19 pandemic, with both the Oly
mpics and Paralympics pushed back a
year.Now, the Tokyo 2020 Olympics
are scheduled for July 23 to August
8 while the Paralympics are due to f
ollow from August 24 to September 5.
The refund process is separate for
ticketholders outside of Japan, who
purchased tickets through authorise
d ticket resellers (ATR).Each ATR
has its own individual refund proced
ure.Early figures from the refund
process for the Tokyo 2020 Olympics
stated that around 18 per cent
has been rescheduled to May 1-4 bec
ause of travel restrictions under th
e current state of emergency in Toky
o and other 10 prefectures across Ja
pan.The Tokyo 2020 organizing comm
ittee announced that the first of 18
test events for the Olympic and Par
alympic Games will involve wheelchai
r rugby, which will be held in Yoyog
i National Stadium from April 3 to 4
.The FINA Diving World Cup will fo
llow from April 18 to 23 at the Toky
o Aquatics Centre, which will also s
erve as an Olympic qualifying event.
The spread of the COVID-19 pandemi
c has slowed down in Tokyo three wee
ks after the Japanese capital entere
d a state of emergency on 2020 Summer ParalympicsThe are an
upcoming major international multi-
sport event for athletes with disabi
lities governed by the International
Paralympic Committee. Scheduled as
the 16th Summer Paralympic Games, th
ey are scheduled to be held in Tokyo
, Japan between 24 August and 5 Sept
ember 2021. Originally due to take p
lace between 25 August and 6 Septemb
er 2020. On 24 March 2020, the IOC a
nd the Tokyo Organizing Committee of
ficially announced that the 2020 Sum
mer Olympics and 2020 Summer Paralym
pics would be postponed to 2021, due
to the COVID-19 pandemic, marking t
he first time that the Paralympics h
as been postponed. They will still b
e publicly marketed as
Olympiad, have now been postponed a
nd rescheduled for 23 July to 8 Augu
st 2021 in Tokyo, Japan. The Games
were postponed in March 2020 as a re
sult of the worldwide Covid-19 pande
mic, although they will still keep t
he name Tokyo 2020 for marketing and
branding purposes. This will be th
e first time the Olympic Games have
been postponed rather than cancelled
.
Olympic Games, when Tokyo became th
e first city in Asia to host the Oly
mpic and Paralympic Games, but unfor
tunately strong winds made it an imp
ossible task this time around.Memb
ers of the Tokyo Organising Committe
e of the Olympic and Paralympic Game
s (Tokyo 2020), Tokyo Metropolitan G
overnment officials, Tokyo 2020 Torc
h Relay Official Ambassadors and rep
resentatives from Miyagi Prefecture
joined the arrival ceremony.FLAME
OF RECOVERYThe Olympic flame will
now be put on display at various loc
ations in the Tohoku region, to high
light the message of hope in the are
as worst affected by the 2011 Great
East Japan Earthqu
40Improving language models by retrieving from trillions of tokens
Table 19 | Daniel Radcliffe, from Wikitext103Valid, retrieval data from c4. The chunks 𝐶 2 and 𝐶 3 are almost
entirely retrieved from neighbours [ 𝑁 1 , 𝐹 1 ] and [ 𝑁 2 , 𝐹 2 ] respectively, up to formatting differences, which
dramatically reduces the loss for these tokens. This example illustrates that when training data leaks into
evaluation sets despite deduplication, our Re t ro model can directly exploit this leakage.
𝐶 𝑢 colored by loss difference
𝐿 Re t ro [Of f ] − 𝐿 Re tro 6 −0 . 5 , = 0 , > 0 . 5 𝐶 𝑢 colored by LCP with Ret ( 𝐶 𝑢 −1)
LCP = 0 , 1 , 2 , 3 , 4 , > 5 [ 𝑁 𝑢 1 , 𝐹 𝑢 1 ] colored by LCP with 𝐶 𝑢 +1
LCP = 0 , 1 , 2 , 3 , 4 , > 5 [ 𝑁 𝑢 2 , 𝐹 𝑢 2 ] colored by LCP with 𝐶 𝑢 +1
LCP = 0 , 1 , 2 , 3 , 4 , > 5
= Daniel Radcliffe =Daniel Jacob R
adcliffe ( born 23 July 1989 ) is an
English actor who rose to prominenc
e as the title character in the Harr
y Potter film series. He made his ac
ting debut at 10 years of age in BBC
One’s 1999 television film David Co
pperfield, followed by his cinematic
debut = Daniel Radcliffe = Daniel Jacob R
adcliffe ( born 23 July 1989 ) is an
English actor who rose to prominenc
e as the title character in the Harr
y Potter film series. He made his ac
ting debut at 10 years of age in BBC
One’s 1999 television film David Co
pperfield, followed by his cinematic
debut Daniel Jacob Radcliffe (born 23 July
1989) is an English actor who rose
to prominence as the title character
in the Harry Potter film series. He
made his acting debut at 10 years o
f age in BBC One’s 1999 television f
ilm David Copperfield, followed by h
is cinematic debut in 2001’s The Tai
lor of Panama. At age 11, he was cas
t as Harry Potter in the first Harry
Potter film, and starred in the ser
ies for 10 years until the release o
f the eighth and final film in 2011.
Radcliffe began to branch out to s
tage acting in 2007, starring in the
London and New York productions of
Equus, and Daniel Jacob Radcliffe (born 23 July
1989) is an English actor who rose
to prominence as the title character
in the Harry Potter film series. He
made his acting debut at 10 years o
f age in BBC One’s 1999 television m
ovie David Copperfield, followed by
his film debut in 2001’s The Tailor
of Panama. At age 11, he was cast as
Harry Potter in the first Harry Pot
ter film, and starred in the series
for 10 years until the release of th
e eighth and final film in 2011. Rad
cliffe began to branch out to stage
acting in 2007, starring in the Lond
on and New York productions of Equus
, and in the
in 2001’s The Tailor of Panama. At
age 11, he was cast as Harry Potter
in the first Harry Potter film, and
starred in the series for 10 years u
ntil the release of the eighth and f
inal film in 2011.Radcliffe began
to branch out to stage acting in 200
7, starring in the London and New in 2001’s The Tailor of Panama. At
age 11, he was cast as Harry Potter
in the first Harry Potter film, and
starred in the series for 10 years u
ntil the release of the eighth and f
inal film in 2011.Radcliffe began
to branch out to stage acting in 200
7, starring in the London and New in 2001’s The Tailor of Panama. At
age 11, he was cast as Harry Potter
in the first Harry Potter film, and
starred in the series for 10 years u
ntil the release of the eighth and f
inal film in 2011.Radcliffe began
to branch out to stage acting in 200
7, starring in the London and New Yo
rk productions of Equus, and in the
2011 Broadway revival of the musical
How to Succeed in Business Without
Really Trying. He starred in the 201
2 horror film The Woman in Black, an
d played beat poet Allen Ginsberg in
the 2013 independent film Kill Your
Darlings.He has contributed to ma
ny charities of Panama. At age 11, he was cast a
s Harry Potter in the first Harry Po
tter film, and starred in the series
for 10 years until the release of t
he eighth and final film in 2011.R
adcliffe began to branch out to stag
e acting in 2007, starring in the Lo
ndon and New York productions of Equ
us, and in the 2011 Broadway revival
of the musical How to Succeed in Bu
siness Without Really Trying. He sta
rred in the 2012 horror film The Wom
an in Black, and played beat poet Al
len Ginsberg in the 2013 independent
film Kill Your Darlings. He has con
tributed to many charities, includin
g Demelza House Children’s
York productions of Equus, and in t
he 2011 Broadway revival of the musi
cal How to Succeed in Business Witho
ut Really Trying. He starred in the
2012 horror film The Woman in Black,
and played beat poet Allen Ginsberg
in the 2013 independent film Kill Y
our <unk>.He has contributed to ma
ny charities, York productions of Equus, and in t
he 2011 Broadway revival of the musi
cal How to Succeed in Business Witho
ut Really Trying. He starred in the
2012 horror film The Woman in Black,
and played beat poet Allen Ginsberg
in the 2013 independent film Kill Y
our <unk>.He has contributed to ma
ny charities, York productions of Equus, and in t
he 2011 Broadway revival of the musi
cal How to Succeed in Business Witho
ut Really Trying. He starred in the
2012 horror film The Woman in Black,
and played beat poet Allen Ginsberg
in the 2013 independent film Kill Y
our Darlings.He has contributed to
many charities, including Demelza H
ouse Children’s Hospice and The Trev
or Project. He also made public serv
ice announcements for the latter. In
2011, he was awarded the Trevor Pro
ject’s "Hero Award."Sources disagr
ee about Radcliffe’s personal wealth
; he was reported to have earned £1
million for the first Harry Potter in the 2011 Broadway revival of the
musical How to Succeed in Business
Without Really Trying. He starred in
the 2012 horror film The Woman in B
lack, and played beat poet Allen Gin
sberg in the 2013 independent film K
ill Your Darlings. He has contribute
d to many charities, including Demel
za House Children’s Hospice and The
Trevor Project. He also made public
service announcements for the latter
. In 2011, he was awarded the Trevor
Project’s "Hero Award."
including <unk> Hospice Care for Ch
ildren, and The Trevor Project for s
uicide prevention among LGBTQ youth,
which gave him its Hero Award in 20
11.= = Early life = =Radcliffe w
as born in West London, England. He
is the only child of Alan George Rad
cliffe, a literary agent, and including <unk> Hospice Care for Ch
ildren, and The Trevor Project for s
uicide prevention among LGBTQ youth,
which gave him its Hero Award in 20
11.= = Early life = =Radcliffe w
as born in West London, England. He
is the only child of Alan George Rad
cliffe, a literary agent, and
41Improving language models by retrieving from trillions of tokens
Table 20 | Sample - Déclaration des droits de l’homme: Article premier. The Re t ro [Of f ] sample has
correct syntax and is almost plausible but is hallucinated. The Re t ro [On ] sample is correctly copied from
neighbour data, and robustly re-formated according to our prompt.
[ 𝑁 𝑢 1 , 𝐹 𝑢 1 ] colored by LCP with 𝐶 𝑢 +1 [ 𝑁 𝑢 2 , 𝐹 𝑢 2 ] colored by LCP with 𝐶 𝑢 +1
LCP = 0 , 1 , 2 , 3 , 4 , > 5 LCP = 0 , 1 , 2 , 3 , 4 , > 5
de l’homme et du citoyen.Ar
ticle 1erLes hommes naissent
et demeurent libres et égaux
en droits. Les distinctions so
ciales ne peuvent être fondées
que sur l’utilité commune.A
rticle 2Le but de toute asso
ciation politique est la conse
rvation des droits naturels et
imprescriptibles de l’homme.
Ces droits sont la liberté, la
propriété, la sûreté, et la r
ésistance à l’oppression.Art
icle 3Le principe de toute s
ouveraineté réside essentielle
ment dans la nation. Nul corps
, nul individu ne peut exercer
d’autorité qui n’en Les hommes naissent et demeur
ent libres et égaux en droits.
Les distinctions sociales ne
peuvent être fondées que sur l
’utilité commune.Art. 2. -
Le but de toute association po
litique est la conservation de
s droits naturels et imprescri
ptibles de l’Homme. Ces droits
sont la liberté, la propriété
, la sûreté, et la résistance
à l’oppression.Art. 3. -Le
principe de toute Souverainet
é réside essentiellement dans
la Nation. Nul corps, nul indi
vidu ne peut exercer d’autorit
é qui n’en émane expressément.
Art
imprescriptibles de l’homme.
Ces droits sont la liberté, la
propriété, la sûreté et la ré
sistance à l’oppression.Arti
cle 3.- Le principe de toute
souveraineté réside essentiel
lement dans la nation. Nul cor
ps, nul individu ne peut exerc
er d’autorité qui n criptibles del’homme. Ces dro
its sont la liberté, la propri
été, la sûretéet la résistanc
e à l’oppression.Article 3 -
Le principe de toute souverai
neté résideessentiellement da
ns la Nation. Nul corps, nul i
ndividu nepeut exercer d’auto
rité qui n’en émane expresséme
nt.Article 4 - La liberté co
nsiste à pouvoir faire tout ce
quine nuit pas à autrui : ai
nsi, l’exercice des droits nat
urelsde chaque homme n’a de b
ornes que celles qui assurent
auxautres membres de la socié
té la jouissance de et imprescriptibles de l’homm
e. Ces droits sont la liberté,
la propriété, la sûreté et la
résistance à l’oppression.A
rticle 3 - Le principe de tout
e souveraineté réside essentie
llement dans la Nation. Nul co
rps, nul individu ne peut exer
cer d’autorité qui n’en émane
expressément.Article 4 - La
liberté consiste à pouvoir fai
re tout ce qui ne nuit pas à a
utrui : ainsi, l’exercice des
droits naturels de chaque homm
e n’a de bornes que celles qui
assurent aux autres membres d
e la société la jouissance de
ces mêmes droits. Ces bornes
but de toute association est
la défense des droits de l’hom
me et du citoyen. Tout citoye
n a le droit de participer à l
a direction des affaires publi
ques. Article 5. - L’impuni
té n’a jamais été et ne sera j
amais une fin en elle-même. L’
imp ’en émane expressément.Artic
le 4.- La liberté consiste à
pouvoir faire tout ce qui ne
nuit pas à autrui : ainsi, l’e
xercice des droits naturels de
chaque homme n’a de bornes qu
e celles qui assurent aux autr
es membres de la société la jo
uissance de ces mêmes mane expressément.Article 4
- La liberté consiste à pouvoi
r faire tout ce qui ne nuit pa
s à autrui : ainsi, l’exercice
des droits naturels de chaque
homme n’a de bornes que celle
s qui assurent aux autres memb
res de la société la jouissanc
e de ces mêmes droits. Ces bor
nes ne peuvent être déterminée
s que par la loi.Article 5 -
La loi n’a le droit de défend
re que les actions nuisibles à
la société. Tout ce qui n’est
pas défendu par la loi ne peu
t être empêché, et nul ne peut
être contraint à faire ce qu’
elle n mane expressément.Article 4
- La liberté consiste à pouvoi
r faire tout ce qui ne nuit pa
s à autrui : ainsi, l’exercice
des droits naturels de chaque
homme n’a de bornes que celle
s qui assurent aux autres memb
res de la société la jouissanc
e de ces mêmes droits. Ces bor
nes ne peuvent être déterminée
s que par la loi.Article 5 -
La loi n’a le droit de défend
re que les actions nuisibles à
la société. Tout ce qui n’est
pas défendu par la loi ne peu
t être empêché, et nul ne peut
être contraint à faire ce qu’
elle n
unité, comme le despotisme, s
’est toujours révélée être un
instrument d’oppression. La ty
rannie qui s’est élue juge su
prême de la conscience des aut
res ne peut être jugée. La jus
tice se trouve dans la consci
ence de chaque citoyen, droits. Ces bornes ne peuvent
être déterminées que par la l
oi.Article 5.- La loi n’a
le droit de défendre que les a
ctions nuisibles à la société.
Tout ce qui n’est pas défendu
par la loi ne peut être empêc
hé, et nul ne peut être Prompt and sample of Retro[Off] Prompt and sample of Retro[On]
colored by LCP with Ret ( 𝐶 𝑢 −1)
LCP = 0 , 1 , 2 , 3 , 4 , > 5
Article premier - Les hommes
naissent et demeurent libres e
t égaux en droits. Les distinc
tions sociales ne peuvent être
fondées que sur l’utilité com
mune. Article 2. - Le but de
toute association politique e
st la conservation des droits
naturels et Article premier - Les hommes
naissent et demeurent libres e
t égaux en droits. Les distinc
tions sociales ne peuvent être
fondées que sur l’utilité com
mune. Article 2. - Le but de
toute association politique e
st la conservation des droits
naturels et
imprescriptibles de l’homme,
et par conséquent la garantie
à chacun des droits suivants
: Article 3. - La propriété
est un droit inviolable et sa
cré. Toute personne a le droit
de procéder à sa propre cons
ervation. Article 4. - Le
42Improving language models by retrieving from trillions of tokens
Table 21 | Sample - Decimals of 𝜋 . The Re t ro [Of f ] sample quickly diverges two digits after the end
of the prompt whereas Re t ro [On ] correctly outputs a large number of 𝜋 digits, directly copied from the
neighbours data.
[ 𝑁 𝑢 1 , 𝐹 𝑢 1 ] colored by LCP with 𝐶 𝑢 +1 [ 𝑁 𝑢 2 , 𝐹 𝑢 2 ] colored by LCP with 𝐶 𝑢 +1
LCP = 0 , 1 , 2 , 3 , 4 , > 5 LCP = 0 , 1 , 2 , 3 , 4 , > 5
“1415926535 8979323846 26433
83279 5028841971 693993751058
20974944 5923078164 0628620899
8628034825 34211706798214808
651 3282306647 0938446095 5058
223172 53594081284811174502 8
410270193 8521105559 644622948
9 54930381964428810975 665933
4461 2847564823 3786783 46 2643383279 5028841971 69399
37510 5820974944 592307816406
28620899 8628034825 3421170679
8214808651 3282306647 0938446
095 50582231725359408128 4811
174502 8410270193 8521105559 6
446229489 5493038196 442881097
56659334461 2847564823 378678
3165 2712019091 4564856692 346
0
Prompt and sample of Retro[Off] Prompt and sample of Retro[On]
colored by LCP with Ret ( 𝐶 𝑢 −1)
LCP = 0 , 1 , 2 , 3 , 4 , > 5 Pi = 3. 1415926535 8979323846
2643383279 5028841971 69399375
10 5820974944 5923078164 06286
20899 8628034825 3421170679 Pi = 3. 1415926535 8979323846
2643383279 5028841971 69399375
10 5820974944 5923078164 06286
20899 8628034825 3421170679 8294049602 8988496069 9858349
065 9873246379 9644789435 8628
730709 6540159079 5944069810 5
992965913 7095378412 69378359 8214808651 3282306647 0938446
095 5058223172 53594081284811
174502 8410270193 8521105559 6
446229489 5493038196442881097
5 6659334461 284 651 3282306647 0938446095 5058
223172 5359408128 4811174502
8410270193 8521105559 64462294
89 54930381964428810975 66593
34461 2847564823 3786783165 27
12019091 4564856692 346034861
0 4543266482 1339360726 024914
12737245870066 0631558817 488
1520920 9628292540 91715364 47 0938446095 5058223172 53594
081284811174502 8410270193 85
21105559 6446229489 5493038196
4428810975 6659334461 2847564
823 3786783165 27120190914564
856692 3460348610 4543266482 1
339360726 0249141273724587006
6 0631558817 4881520920 962829
2540 91715364367892590360
10 6940372045 7088679512 85612
30857 9046461290 9276642155 56
54603269 5656128798 6366475705
6294954741 5886335339 57657 7564823 3786783165 2712019091
4564856692 3460348610 45432664
82 1339360726 024914127372458
70066 0631558817 4881520920 96
28292540 91715 23 3786783165 2712019091 4564
856692 3460348610 4543266482 1
339360726 0249141273724587006
6 0631558817 4881520920 962829
2540 9171536436 7892590360 01
13305305 4882046652 1384146951
94151160943305727036 5759591
953 0921861173 8193261179 3105
118548 0744623799 627495 165 27120190914564856692 3460
348610 4543266482 1339360726 0
2491412737245870066 063155881
7 4881520920 9628292540 917153
64367892590360 0113305305 488
2046652 1384146951 9415116094
3305727036 5759591953 09218611
73 8193261179 310511854807446
23799 6274956735 1885752724 89
1227
76345 5770886953 7988876910 79
66169745 6493974637 6345801550
6663542854 6333764630 6356284
271 7885339804 5672434 364367892590360 0113305305 48
82046652 1384146951 9415116094
3305727036 5759591953 0921861
173 8193261179 31051185480744
623799 6274
43
'''

transformer = '''
arXiv:1706.03762v5 [cs.CL] 6 Dec 2017

Attention Is All You Need

Ashish Vaswani∗
Google Brain
avaswani@google.com
Llion Jones∗
Google Research
llion@google.com

Noam Shazeer∗
Google Brain
noam@google.com

Niki Parmar∗
Google Research
nikip@google.com

Aidan N. Gomez∗ †
University of Toronto
aidan@cs.toronto.edu

Jakob Uszkoreit∗
Google Research
usz@google.com

Łukasz Kaiser∗
Google Brain
lukaszkaiser@google.com

Illia Polosukhin∗ ‡
illia.polosukhin@gmail.com

Abstract
The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to
be superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.8 after
training for 3.5 days on eight GPUs, a small fraction of the training costs of the
best models from the literature. We show that the Transformer generalizes well to
other tasks by applying it successfully to English constituency parsing both with
large and limited training data.

1

Introduction

Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks
in particular, have been firmly established as state of the art approaches in sequence modeling and
∗
Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started
the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and
has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head
attention and the parameter-free position representation and became the other person involved in nearly every
detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and
tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and
efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and
implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating
our research.
†
Work performed while at Google Brain.
‡
Work performed while at Google Research.

31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.

transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous
efforts have since continued to push the boundaries of recurrent language models and encoder-decoder
architectures [38, 24, 15].
Recurrent models typically factor computation along the symbol positions of the input and output
sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden
states ht , as a function of the previous hidden state ht−1 and the input for position t. This inherently
sequential nature precludes parallelization within training examples, which becomes critical at longer
sequence lengths, as memory constraints limit batching across examples. Recent work has achieved
significant improvements in computational efficiency through factorization tricks [21] and conditional
computation [32], while also improving model performance in case of the latter. The fundamental
constraint of sequential computation, however, remains.
Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in
the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms
are used in conjunction with a recurrent network.
In this work we propose the Transformer, a model architecture eschewing recurrence and instead
relying entirely on an attention mechanism to draw global dependencies between input and output.
The Transformer allows for significantly more parallelization and can reach a new state of the art in
translation quality after being trained for as little as twelve hours on eight P100 GPUs.

2

Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU
[16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building
block, computing hidden representations in parallel for all input and output positions. In these models,
the number of operations required to relate signals from two arbitrary input or output positions grows
in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes
it more difficult to learn dependencies between distant positions [12]. In the Transformer this is
reduced to a constant number of operations, albeit at the cost of reduced effective resolution due
to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as
described in section 3.2.
Self-attention, sometimes called intra-attention is an attention mechanism relating different positions
of a single sequence in order to compute a representation of the sequence. Self-attention has been
used successfully in a variety of tasks including reading comprehension, abstractive summarization,
textual entailment and learning task-independent sentence representations [4, 27, 28, 22].
End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and
language modeling tasks [34].
To the best of our knowledge, however, the Transformer is the first transduction model relying
entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution. In the following sections, we will describe the Transformer, motivate
self-attention and discuss its advantages over models such as [17, 18] and [9].

3

Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35].
Here, the encoder maps an input sequence of symbol representations (x1 , ..., xn ) to a sequence
of continuous representations z = (z1 , ..., zn ). Given z, the decoder then generates an output
sequence (y1 , ..., ym ) of symbols one element at a time. At each step the model is auto-regressive
[10], consuming the previously generated symbols as additional input when generating the next.
The Transformer follows this overall architecture using stacked self-attention and point-wise, fully
connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1,
respectively.
2

Figure 1: The Transformer - model architecture.

3.1

Encoder and Decoder Stacks

Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two
sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection [11] around each of
the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding
layers, produce outputs of dimension dmodel = 512.
Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two
sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
attention over the output of the encoder stack. Similar to the encoder, we employ residual connections
around each of the sub-layers, followed by layer normalization. We also modify the self-attention
sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
masking, combined with fact that the output embeddings are offset by one position, ensures that the
predictions for position i can depend only on the known outputs at positions less than i.
3.2

Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output,
where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
of the values, where the weight assigned to each value is computed by a compatibility function of the
query with the corresponding key.
3

Scaled Dot-Product Attention

Multi-Head Attention

Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several
attention layers running in parallel.
3.2.1

Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of
queries and keys of dimension dk , and
√ values of dimension dv . We compute the dot products of the
query with all keys, divide each by dk , and apply a softmax function to obtain the weights on the
values.
In practice, we compute the attention function on a set of queries simultaneously, packed together
into a matrix Q. The keys and values are also packed together into matrices K and V . We compute
the matrix of outputs as:
QK T
Attention(Q, K, V ) = softmax( √ )V
dk

(1)

The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor
of √1d . Additive attention computes the compatibility function using a feed-forward network with
k
a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is
much faster and more space-efficient in practice, since it can be implemented using highly optimized
matrix multiplication code.
While for small values of dk the two mechanisms perform similarly, additive attention outperforms
dot product attention without scaling for larger values of dk [3]. We suspect that for large values of
dk , the dot products grow large in magnitude, pushing the softmax function into regions where it has
extremely small gradients 4 . To counteract this effect, we scale the dot products by √1d .
k

3.2.2

Multi-Head Attention

Instead of performing a single attention function with dmodel -dimensional keys, values and queries,
we found it beneficial to linearly project the queries, keys and values h times with different, learned
linear projections to dk , dk and dv dimensions, respectively. On each of these projected versions of
queries, keys and values we then perform the attention function in parallel, yielding dv -dimensional
output values. These are concatenated and once again projected, resulting in the final values, as
depicted in Figure 2.
4

To illustrate why the dot products get large, assume that the components of q and k are independent random
P k
variables with mean 0 and variance 1. Then their dot product, q · k = di=1
qi ki , has mean 0 and variance dk .

4

Multi-head attention allows the model to jointly attend to information from different representation
subspaces at different positions. With a single attention head, averaging inhibits this.
MultiHead(Q, K, V ) = Concat(head1 , ..., headh )W O
where headi = Attention(QWiQ , KWiK , V WiV )
Where the projections are parameter matrices WiQ ∈ Rdmodel ×dk , WiK ∈ Rdmodel ×dk , WiV ∈ Rdmodel ×dv
and W O ∈ Rhdv ×dmodel .
In this work we employ h = 8 parallel attention layers, or heads. For each of these we use
dk = dv = dmodel /h = 64. Due to the reduced dimension of each head, the total computational cost
is similar to that of single-head attention with full dimensionality.
3.2.3

Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:
• In "encoder-decoder attention" layers, the queries come from the previous decoder layer,
and the memory keys and values come from the output of the encoder. This allows every
position in the decoder to attend over all positions in the input sequence. This mimics the
typical encoder-decoder attention mechanisms in sequence-to-sequence models such as
[38, 2, 9].
• The encoder contains self-attention layers. In a self-attention layer all of the keys, values
and queries come from the same place, in this case, the output of the previous layer in the
encoder. Each position in the encoder can attend to all positions in the previous layer of the
encoder.
• Similarly, self-attention layers in the decoder allow each position in the decoder to attend to
all positions in the decoder up to and including that position. We need to prevent leftward
information flow in the decoder to preserve the auto-regressive property. We implement this
inside of scaled dot-product attention by masking out (setting to −∞) all values in the input
of the softmax which correspond to illegal connections. See Figure 2.
3.3

Position-wise Feed-Forward Networks

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
connected feed-forward network, which is applied to each position separately and identically. This
consists of two linear transformations with a ReLU activation in between.
FFN(x) = max(0, xW1 + b1 )W2 + b2

(2)

While the linear transformations are the same across different positions, they use different parameters
from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality
df f = 2048.
3.4

Embeddings and Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input
tokens and output tokens to vectors of dimension dmodel . We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In
our model, we share the same weight matrix between the two embedding layers and the pre-softmax
√
linear transformation, similar to [30]. In the embedding layers, we multiply those weights by dmodel .
3.5

Positional Encoding

Since our model contains no recurrence and no convolution, in order for the model to make use of the
order of the sequence, we must inject some information about the relative or absolute position of the
5

Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations
for different layer types. n is the sequence length, d is the representation dimension, k is the kernel
size of convolutions and r the size of the neighborhood in restricted self-attention.
Layer Type

Complexity per Layer

Self-Attention
Recurrent
Convolutional
Self-Attention (restricted)

O(n2 · d)
O(n · d2 )
O(k · n · d2 )
O(r · n · d)

Sequential
Operations
O(1)
O(n)
O(1)
O(1)

Maximum Path Length
O(1)
O(n)
O(logk (n))
O(n/r)

tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the
bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel
as the embeddings, so that the two can be summed. There are many choices of positional encodings,
learned and fixed [9].
In this work, we use sine and cosine functions of different frequencies:
P E(pos,2i) = sin(pos/100002i/dmodel )
P E(pos,2i+1) = cos(pos/100002i/dmodel )
where pos is the position and i is the dimension. That is, each dimension of the positional encoding
corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We
chose this function because we hypothesized it would allow the model to easily learn to attend by
relative positions, since for any fixed offset k, P Epos+k can be represented as a linear function of
P Epos .
We also experimented with using learned positional embeddings [9] instead, and found that the two
versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version
because it may allow the model to extrapolate to sequence lengths longer than the ones encountered
during training.

4

Why Self-Attention

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations
(x1 , ..., xn ) to another sequence of equal length (z1 , ..., zn ), with xi , zi ∈ Rd , such as a hidden
layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we
consider three desiderata.
One is the total computational complexity per layer. Another is the amount of computation that can
be parallelized, as measured by the minimum number of sequential operations required.
The third is the path length between long-range dependencies in the network. Learning long-range
dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the
ability to learn such dependencies is the length of the paths forward and backward signals have to
traverse in the network. The shorter these paths between any combination of positions in the input
and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare
the maximum path length between any two input and output positions in networks composed of the
different layer types.
As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially
executed operations, whereas a recurrent layer requires O(n) sequential operations. In terms of
computational complexity, self-attention layers are faster than recurrent layers when the sequence
length n is smaller than the representation dimensionality d, which is most often the case with
sentence representations used by state-of-the-art models in machine translations, such as word-piece
[38] and byte-pair [31] representations. To improve computational performance for tasks involving
very long sequences, self-attention could be restricted to considering only a neighborhood of size r in
6

the input sequence centered around the respective output position. This would increase the maximum
path length to O(n/r). We plan to investigate this approach further in future work.
A single convolutional layer with kernel width k < n does not connect all pairs of input and output
positions. Doing so requires a stack of O(n/k) convolutional layers in the case of contiguous kernels,
or O(logk (n)) in the case of dilated convolutions [18], increasing the length of the longest paths
between any two positions in the network. Convolutional layers are generally more expensive than
recurrent layers, by a factor of k. Separable convolutions [6], however, decrease the complexity
considerably, to O(k · n · d + n · d2 ). Even with k = n, however, the complexity of a separable
convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer,
the approach we take in our model.
As side benefit, self-attention could yield more interpretable models. We inspect attention distributions
from our models and present and discuss examples in the appendix. Not only do individual attention
heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic
and semantic structure of the sentences.

5

Training

This section describes the training regime for our models.
5.1

Training Data and Batching

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million
sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared sourcetarget vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT
2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece
vocabulary [38]. Sentence pairs were batched together by approximate sequence length. Each training
batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000
target tokens.
5.2

Hardware and Schedule

We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using
the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We
trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the
bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps
(3.5 days).
5.3

Optimizer

We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and  = 10−9 . We varied the learning
rate over the course of training, according to the formula:
−0.5
lrate = d−0.5
, step_num · warmup_steps−1.5 )
model · min(step_num

(3)

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_steps = 4000.
5.4

Regularization

We employ three types of regularization during training:
Residual Dropout We apply dropout [33] to the output of each sub-layer, before it is added to the
sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
Pdrop = 0.1.
7

Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the
English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.
Model
ByteNet [18]
Deep-Att + PosUnk [39]
GNMT + RL [38]
ConvS2S [9]
MoE [32]
Deep-Att + PosUnk Ensemble [39]
GNMT + RL Ensemble [38]
ConvS2S Ensemble [9]
Transformer (base model)
Transformer (big)

BLEU
EN-DE
23.75
24.6
25.16
26.03
26.30
26.36
27.3
28.4

EN-FR
39.2
39.92
40.46
40.56
40.4
41.16
41.29
38.1
41.8

Training Cost (FLOPs)
EN-DE

EN-FR

1.0 · 1020
1.4 · 1020
1.5 · 1020
1.2 · 1020
8.0 · 1020
20
1.8 · 10
1.1 · 1021
19
7.7 · 10
1.2 · 1021
3.3 · 1018
2.3 · 1019
2.3 · 1019
9.6 · 1018
2.0 · 1019

Label Smoothing During training, we employed label smoothing of value ls = 0.1 [36]. This
hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

6

Results

6.1

Machine Translation

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big)
in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0
BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is
listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model
surpasses all previously published models and ensembles, at a fraction of the training cost of any of
the competitive models.
On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0,
outperforming all of the previously published single models, at less than 1/4 the training cost of the
previous state-of-the-art model. The Transformer (big) model trained for English-to-French used
dropout rate Pdrop = 0.1, instead of 0.3.
For the base models, we used a single model obtained by averaging the last 5 checkpoints, which
were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We
used beam search with a beam size of 4 and length penalty α = 0.6 [38]. These hyperparameters
were chosen after experimentation on the development set. We set the maximum output length during
inference to input length + 50, but terminate early when possible [38].
Table 2 summarizes our results and compares our translation quality and training costs to other model
architectures from the literature. We estimate the number of floating point operations used to train a
model by multiplying the training time, the number of GPUs used, and an estimate of the sustained
single-precision floating-point capacity of each GPU 5 .
6.2

Model Variations

To evaluate the importance of different components of the Transformer, we varied our base model
in different ways, measuring the change in performance on English-to-German translation on the
development set, newstest2013. We used beam search as described in the previous section, but no
checkpoint averaging. We present these results in Table 3.
In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions,
keeping the amount of computation constant, as described in Section 3.2.2. While single-head
attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.
5

We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.

8

Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base
model. All metrics are on the English-to-German translation development set, newstest2013. Listed
perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to
per-word perplexities.

base

N

dmodel

dff

h

dk

dv

Pdrop

ls

6

512

2048

8
1
4
16
32

64
512
128
32
16
16
32

64
512
128
32
16

0.1

0.1

32
128

32
128

(A)

(B)

train
steps
100K

2
4
8
(C)

256
1024
1024
4096

0.0
0.2

(D)
(E)
big

6

0.0
0.2
positional embedding instead of sinusoids
1024 4096 16
0.3

300K

PPL
(dev)
4.92
5.29
5.00
4.91
5.01
5.16
5.01
6.11
5.19
4.88
5.75
4.66
5.12
4.75
5.77
4.95
4.67
5.47
4.92
4.33

BLEU
(dev)
25.8
24.9
25.5
25.8
25.4
25.1
25.4
23.7
25.3
25.5
24.5
26.0
25.4
26.2
24.6
25.5
25.3
25.7
25.7
26.4

params
×106
65

58
60
36
50
80
28
168
53
90

213

Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23
of WSJ)
Parser
Training
WSJ 23 F1
Vinyals & Kaiser el al. (2014) [37] WSJ only, discriminative
88.3
Petrov et al. (2006) [29]
WSJ only, discriminative
90.4
Zhu et al. (2013) [40]
WSJ only, discriminative
90.4
Dyer et al. (2016) [8]
WSJ only, discriminative
91.7
Transformer (4 layers)
WSJ only, discriminative
91.3
Zhu et al. (2013) [40]
semi-supervised
91.3
Huang & Harper (2009) [14]
semi-supervised
91.3
semi-supervised
92.1
McClosky et al. (2006) [26]
Vinyals & Kaiser el al. (2014) [37]
semi-supervised
92.1
Transformer (4 layers)
semi-supervised
92.7
Luong et al. (2015) [23]
multi-task
93.0
Dyer et al. (2016) [8]
generative
93.3

In Table 3 rows (B), we observe that reducing the attention key size dk hurts model quality. This
suggests that determining compatibility is not easy and that a more sophisticated compatibility
function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected,
bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our
sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical
results to the base model.
6.3

English Constituency Parsing

To evaluate if the Transformer can generalize to other tasks we performed experiments on English
constituency parsing. This task presents specific challenges: the output is subject to strong structural
9

constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence
models have not been able to attain state-of-the-art results in small-data regimes [37].
We trained a 4-layer transformer with dmodel = 1024 on the Wall Street Journal (WSJ) portion of the
Penn Treebank [25], about 40K training sentences. We also trained it in a semi-supervised setting,
using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences
[37]. We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of 32K tokens
for the semi-supervised setting.
We performed only a small number of experiments to select the dropout, both attention and residual
(section 5.4), learning rates and beam size on the Section 22 development set, all other parameters
remained unchanged from the English-to-German base translation model. During inference, we
increased the maximum output length to input length + 300. We used a beam size of 21 and α = 0.3
for both WSJ only and the semi-supervised setting.
Our results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the
Recurrent Neural Network Grammar [8].
In contrast to RNN sequence-to-sequence models [37], the Transformer outperforms the BerkeleyParser [29] even when training only on the WSJ training set of 40K sentences.

7

Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on
attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with
multi-headed self-attention.
For translation tasks, the Transformer can be trained significantly faster than architectures based
on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014
English-to-French translation tasks, we achieve a new state of the art. In the former task our best
model outperforms even all previously reported ensembles.
We are excited about the future of attention-based models and plan to apply them to other tasks. We
plan to extend the Transformer to problems involving input and output modalities other than text and
to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs
such as images, audio and video. Making generation less sequential is another research goals of ours.
The code we used to train and evaluate our models is available at https://github.com/
tensorflow/tensor2tensor.
Acknowledgements We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful
comments, corrections and inspiration.

References
[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint
arXiv:1607.06450, 2016.
[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly
learning to align and translate. CoRR, abs/1409.0473, 2014.
[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural
machine translation architectures. CoRR, abs/1703.03906, 2017.
[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine
reading. arXiv preprint arXiv:1601.06733, 2016.
[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk,
and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical
machine translation. CoRR, abs/1406.1078, 2014.
[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv
preprint arXiv:1610.02357, 2016.
10

[7] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation
of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.
[8] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural
network grammars. In Proc. of NAACL, 2016.
[9] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.
[10] Alex Graves. Generating sequences with recurrent neural networks.
arXiv:1308.0850, 2013.

arXiv preprint

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 770–778, 2016.
[12] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in
recurrent nets: the difficulty of learning long-term dependencies, 2001.
[13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation,
9(8):1735–1780, 1997.
[14] Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations
across languages. In Proceedings of the 2009 Conference on Empirical Methods in Natural
Language Processing, pages 832–841. ACL, August 2009.
[15] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring
the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.
[16] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In Advances in Neural
Information Processing Systems, (NIPS), 2016.
[17] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference
on Learning Representations (ICLR), 2016.
[18] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2,
2017.
[19] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks.
In International Conference on Learning Representations, 2017.
[20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.
[21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint
arXiv:1703.10722, 2017.
[22] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen
Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint
arXiv:1703.03130, 2017.
[23] Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task
sequence to sequence learning. arXiv preprint arXiv:1511.06114, 2015.
[24] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attentionbased neural machine translation. arXiv preprint arXiv:1508.04025, 2015.
[25] Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated
corpus of english: The penn treebank. Computational linguistics, 19(2):313–330, 1993.
[26] David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In
Proceedings of the Human Language Technology Conference of the NAACL, Main Conference,
pages 152–159. ACL, June 2006.
11

[27] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention
model. In Empirical Methods in Natural Language Processing, 2016.
[28] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive
summarization. arXiv preprint arXiv:1705.04304, 2017.
[29] Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact,
and interpretable tree annotation. In Proceedings of the 21st International Conference on
Computational Linguistics and 44th Annual Meeting of the ACL, pages 433–440. ACL, July
2006.
[30] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv
preprint arXiv:1608.05859, 2016.
[31] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words
with subword units. arXiv preprint arXiv:1508.07909, 2015.
[32] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton,
and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts
layer. arXiv preprint arXiv:1701.06538, 2017.
[33] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine
Learning Research, 15(1):1929–1958, 2014.
[34] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory
networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors,
Advances in Neural Information Processing Systems 28, pages 2440–2448. Curran Associates,
Inc., 2015.
[35] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural
networks. In Advances in Neural Information Processing Systems, pages 3104–3112, 2014.
[36] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna.
Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.
[37] Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In
Advances in Neural Information Processing Systems, 2015.
[38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang
Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine
translation system: Bridging the gap between human and machine translation. arXiv preprint
arXiv:1609.08144, 2016.
[39] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with
fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.
[40] Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate
shift-reduce constituent parsing. In Proceedings of the 51st Annual Meeting of the ACL (Volume
1: Long Papers), pages 434–443. ACL, August 2013.
12

<pad>

<pad>

<pad>

<pad>

<pad>

<pad>

<pad>
<pad>

<EOS>

<pad>
<pad>
<pad>

.
.

<EOS>
<pad>

voting

process
more
difficult
process
more
difficult

registration
or

or
voting

registration

making
the
making
the

laws

since
2009

new

laws
since
2009

have
passed
new
have
passed

American
governments

of
of

American
governments

spirit

that
a
majority
that
a
majority

in
this
in

this
spirit

It
is

It
is

Input-Input
Layer5
Attention Visualizations

Figure 3: An example of the attention mechanism following long-distance dependencies in the
encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of
the verb ‘making’, completing the phrase ‘making...more difficult’. Attentions here shown only for
the word ‘making’. Different colors represent different heads. Best viewed in color.

13

14
my
opinion
.
<EOS>
<pad>

<EOS>
<pad>

in

my
opinion
.

,
in

missing
,

we
are

we
are
missing

is
what

this
is
what

this

be
just
-

application
should

its

perfect
,
but

<EOS>
<pad>

<EOS>
<pad>

in
my
opinion
.

my
opinion
.

,
in

missing
,

we
are

we
are
missing

is
what

this

be
just
-

application
should

its

perfect
,
but

be

will
never

The
Law

this
is
what

-

be
just

application
should

its

perfect
,
but

never
be

will

The
Law

Input-Input Layer5

-

be
just

application
should

its

perfect
,
but

be

will
never

will
never
be

The
Law

The
Law

Input-Input Layer5

Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top:
Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5
and 6. Note that the attentions are very sharp for this word.

15
is
what
we
are

we
are

<EOS>
<pad>

<EOS>
<pad>

in
my
opinion
.

my
opinion
.

,
in

missing
,

this

this
is
what

<pad>

<EOS>

my
opinion
.

,
in

missing

we
are

this
is
what

-

be
just

application
should

its

perfect
,
but

never
be

will

The
Law

Input-Input Layer5

missing

be
just
-

application
should

its

perfect
,
but

be
just
-

application
should

its

perfect
,
but

be

will
never

will
never
be

The
Law

The
Law

<EOS>
<pad>

my
opinion
.

in

missing
,

we
are

is
what

this

be
just
-

application
should

its

perfect
,
but

be

will
never

The
Law

Input-Input Layer5

Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the
sentence. We give two such examples above, from two different heads from the encoder self-attention
at layer 5 of 6. The heads clearly learned to perform different tasks.
'''