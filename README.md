# NetsForNoobs
Neural nets and deep learning self-study syllabus

This is a simple syllabus for self-study of neural nets. I've gathered readings from academic and non-academic sources that: 
1. **Contextualize deep learning/neural nets** within the machine learning (ML) and artificial intelligence (AI) literatures  

2. **Introduce core concepts** using the multilayer perceptron (MLP) feedforward architecture  
3. Explore **reinforcement learning** with neural nets  
4. Explore convolutional neural nets **(CNN's)** and recurrent neural nets **(RNN's)**.  

**Supplementary Readings** are cleverly hidden in a section titled "Supplementary Readings," and include topics like natural language processing (NLP)/speech recognition, manifold learning, hyperparameters, metacognition (machine reasoning), and evolutionary computing/genetic algorithms for neural nets.

Each unit should take ~2 weeks, depending on time available and prior familiarity. 

Unit 1: Introduction to Deep Learning and Neural Nets
---
1. *Deep Learning* Ch. 1, "Introduction" (you'll get most of what you need in p. 1-11). *Deep Learning* 2016 by Goodfellow, Bengio, and Courville, http://www.deeplearningbook.org/contents/intro.html

2. “Deep Learning”, *MIT Technology Review,* Robert D. Hof, https://www.technologyreview.com/s/513696/deep-learning/
3. *Neural Networks and Deep Learning* Ch. 1, “Using neural nets to recognize handwritten digits”, Michael Nielsen, 2016, http://neuralnetworksanddeeplearning.com/chap1.html
4. Essence of Linear Algebra lecture series, *Youtube,* from user 3Blue1Brown via Khan Academy, https://www.youtube.com/watch?v=kjBOesZCoqc&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
5. “What Minsky Still Means for AI”, *MIT Technology Review,* Will Knight, Jan 26, 2016, https://www.technologyreview.com/s/546116/what-marvin-minsky-still-means-for-ai/

Unit 2: Backpropagation, Deep Learning, Cost Functions
---
1. “Calculus on Computational Graphs: Backpropagation”, Christopher Olah, Aug 31, 2015, http://colah.github.io/posts/2015-08-Backprop/
2. *Neural Networks and Deep Learning* Ch. 2, “How the backpropagation algorithm works”, Michael Nielsen, 2016, http://neuralnetworksanddeeplearning.com/chap2.html

   Backprop is just reverse mode differentiation using chain rule – it’s important to see that Nielsen’s somewhat cryptic construct of a layer’s “error” vector δ<sup>l</sup> is really just ∂*C*/∂*z*<sub>j</sub><sup>l</sup>. That is, “error” for Nielsen is a roundabout way of saying the partial derivative of the cost function with respect to the vector of weighted inputs for a layer *l*. As the output of a layer of neurons is a function of its weighted input (i.e. the activation function acts on the weighted input to produce the neurons’ outputs), we can then use chain rule to further decompose the weighted input into its two component parts: the effect of the weights and the biases. Thus we can recover the information we want, via chain rule, which is how changing weights and biases affect the cost function.
3. "Theoretical Motivations for Deep Learning”, Rinu Boney, Oct 18, 2015, http://rinuboney.github.io/2015/10/18/theoretical-motivations-deep-learning.html
4. “Deep Learning” in *Nature* REVIEW Vol. 521, May 28, 2015, p. 436-444, https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf 
5. “Deep Learning Conspiracy In Nature, 521, p. 436-444”, Jurgen Schmidhuber, June 2015, http://people.idsia.ch/~juergen/deep-learning-conspiracy.html
6. *Neural Networks and Deep Learning* Ch. 3, “Improving the way neural networks learn”, Michael Nielsen, 2016, http://neuralnetworksanddeeplearning.com/chap3.html
7. “Visual Information Theory”, Christopher Olah, Oct 14, 2015, http://colah.github.io/posts/2015-09-Visual-Information/ (Gives insight into why cross-entropy is a good cost function)

Unit 3: Reinforcement Learning
---
1. “Demystifying Deep Reinforcement Learning”, Tambet Matiisen, Dec 21, 2015, https://www.nervanasys.com/demystifying-deep-reinforcement-learning/

2. “Human-level control through deep reinforcement learning,” Mnih et al., *Nature* Vol. 518, Feb 26, 2015, http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf (This is one of the classic Atari DeepMind papers)
3. “Deep Reinforcement Learning: Pong from Pixels,” Andrej Karpathy, May 31, 2016, http://karpathy.github.io/2016/05/31/rl/ 

Unit 4: Convolutional Neural Nets (CNN's)
---
1. “Convolutional Neural Networks” Architecture Overview, Andrej Karpathy, http://cs231n.github.io/convolutional-networks/ 

2. *Neural Networks and Deep Learning* Ch. 6, “Deep learning”, Read up until the section titled “Other approaches to deep neural nets”, Michael Nielsen, 2016, http://neuralnetworksanddeeplearning.com/chap6.html
3. “Conv Nets: A Modular Perspective”, Christopher Olah, Jul 8, 2014, http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
4. “Understanding Convolutions”, Christopher Olah, Jul 13, 2014, http://colah.github.io/posts/2014-07-Understanding-Convolutions/ 
5. “ImageNet Classification with Deep Convolutional Neural Networks,” Krizhevsky, Sutskever, Hinton, in NIPS 2012, https://papers.nips.cc/paper/4824-imagenet-classification-with-deep- (This is the classic “AlexNet” paper)

Unit 5: Recurrent Neural Nets (RNN's)
---
1. “The Unreasonable Effectiveness of Recurrent Neural Networks”, Andrej Karpathy, May 21, 2015, http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 

2. *Neural Networks and Deep Learning* Ch. 5, “Why are deep neural networks hard to train?”, Michael Nielsen, 2016, http://neuralnetworksanddeeplearning.com/chap5.html
3. Neural Networks and Deep Learning Ch. 6, “Deep learning”, From the section titled “Other approaches to deep neural nets” to the end, Michael Nielsen, 2016, http://neuralnetworksanddeeplearning.com/chap6.html
4. “Understanding LSTM Networks”, Christopher Olah, Aug 27, 2015, http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 


Supplementary Readings
---


NLP / Speech Recognition
---
1. “Distributed Representations of Words and Phrases and their Compositionality”, Mikolov et al., NIPS 2013, https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf (The famous word2vec paper, see also “Efficient Estimation of Word Representations in Vector Space” for prior work - https://arxiv.org/pdf/1301.3781.pdf)

2. “Deep Learning, NLP, and Representations,” Christopher Olah, Jul 7, 2014, http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/ 
3. “Quantifying and Reducing Stereotypes in Word Embeddings”, Bolukbasi et al, Jun 20 2016, https://arxiv.org/abs/1606.06121 (this paper applies word2vec)
4. “The Microsoft 2016 Conversational Speech Recognition System”, Xiong et al, Sep 12 2016, https://arxiv.org/abs/1609.03528 (this paper beat the Switchboard Record for speech recognition)

Manifold Learning
---
1. “Neural Networks, Manifolds, and Topology”, Christopher Olah, April 6, 2014, http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/ 

2. “The Definition of a Manifold and First Examples”, Jenny Wilson, WOMP 2012, http://web.stanford.edu/~jchw/WOMPtalk-Manifolds.pdf 

Neural Net Hyperparameters
---
1. *Learning Deep Architectures for AI*, Yoshua Bengio 2009 - www.iro.umontreal.ca/~bengioy/papers/ftml_book.pdf 
2. “Practical recommendations for gradient-based training of deep architectures”, Bengio, Yoshua, Sep 16 2012 - https://arxiv.org/abs/1206.5533. 

3. “Efficient BackProp”, LeCun, Yann et al. 1998, http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
4. “What Size Neural Network Gives Optimal Generalization? Convergence Properties of Backpropagation”, Lawrence, Steve, et al., August 1996 - https://clgiles.ist.psu.edu/papers/UMD-CS-TR-3617.what.size.neural.net.to.use.pdf 

Metacognition / Machine Reasoning
---
1. “From machine learning to machine reasoning,” Bottou, Leon, April 10 2013, http://leon.bottou.org/publications/pdf/mlj-2013.pdf 

2. “Lifelong Machine Learning Systems: Beyond Learning Algorithms”, Silver et al., March 15 2013, http://www.aaai.org/ocs/index.php/SSS/SSS13/paper/view/5802
3. “Metacognition in Computation: A selected research review”, Cox Michael T., 2005, http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.105.8914&rep=rep1&type=pdf 
4. “Counterfactual Reasoning and Learning Systems: The Example of Computational Advertising”, Bottou, Leon et al., 2013, http://jmlr.org/papers/volume14/bottou13a/bottou13a.pdf 
5. “A Roadmap towards Machine Intelligence”, Mikolov, Tomas (author of word2vec) et al, Nov. 25 2015, https://arxiv.org/abs/1511.08130 

Evolutionary Computing/Genetic Algorithms with Neural Nets
---
1. “An Evolved Circuit Intrinsic in Silicon Entwined with Physics”, Thompson, Adrian, 1996, https://www.researchgate.net/profile/Adrian_Thompson5/publication/2737441_An_Evolved_Circuit_Intrinsic_in_Silicon_Entwined_With_Physics/links/56f1ffc908ae4744a91eff8d.pdf (Shows an evolved FPGA that looks remarkably recurrent).

2. “Evolving Reusable Neural Modules”, Reisinger, Joseph et al, 2004 http://nn.cs.utexas.edu/downloads/papers/reisinger.gecco04.pdf 
3. “Artificial Brains. An Inexpensive Method for Accelerating the Evolution of Neural Netwo9rk Modules for Building Artificial Brains”, de Garis et al, http://goertzel.org/agiri06/%5B10%5D AGI-Book-deGaris-2.pdf 
4. “Automatic Multi-Module Neural Network Evolution in an Artificial Brain”, Dinerstein, Jonathan et al, 2003, http://dl.acm.org/citation.cfm?id=938186&preflayout=tabs 
5. “Population-Based Incremental Learning: A Method for Integrating Genetic Search Based Function Optimization and Competitive Learning”, Baluja, Shumeet 1994, http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=ACEC8FCF96ABC7E75AB8F4380E0700DC?doi=10.1.1.61.8554&rep=rep1&type=pdf

Licenses
---
Contributions to this Project are governed by the Contributor License Agreement (/ContributorLicenseAgreement.txt). 

Use of this Project is governed by the CC-BY-SA-4.0 License (/LICENSE.txt).
