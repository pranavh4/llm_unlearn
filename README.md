## Exploring Unlearning in State Space Models

Project developed as part of the CSE 575 Statistical Machine Learning Course taken by Dr. Kookjin Lee at Arizona State University.

### Abstract

This study explores the application of machine unlearning techniques to State Space Models (SSMs), an area that has received limited attention compared to Transformer models. The research aims to adapt existing unlearning methods for SSMs and compare their performance with Transformer models in terms of effectiveness, efficiency, and privacy-preserving capabilities. The experiment utilizes OPT-1.3B, Pythia-1.4B (Transformer models), and Mamba-1.4B (State Space Model). The PKU-SafeRLHF dataset, containing unsafe prompt-response pairs, is used as the forget dataset. Two unlearning methods are implemented: Gradient Ascent and Gradient Ascent with Mismatch. Results indicate that Transformer models respond quickly to fine-tuning methods, achieving good unlearning performance after only 2000 examples. In contrast, the Mamba model (SSM) shows more rigidity and moves very slowly towards the unlearning target. This difference in behavior might be attributed to the distinct internal knowledge representation in SSM and Transformer architectures. The study suggests that State Space Models may be more resilient to phenomena such as Catastrophic Forgetting. However, further research is needed to explore why SSMs remain rigid during fine-tuning and gradient ascent. The findings open avenues for developing custom unlearning algorithms explicitly tailored for SSMs.

The entire report with all info can be found [here](./CSE_575_Final_Report_Group_13.pdf).

