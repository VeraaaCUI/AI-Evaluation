# A Comprehensive Overview of Evaluation Methods of Large Models

## Evaluation Methods

### General Evaluations

- **2020 MMLU** [[1]](https://arxiv.org/abs/2009.03300): Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt. "Measuring massive multitask language understanding." arXiv preprint arXiv:2009.03300 (2020).
- **2021 DynaBench** [[2]](https://arxiv.org/abs/2104.14337): Douwe Kiela, Max Bartolo, Yixin Nie, Divyansh Kaushik, Atticus Geiger, Zhengxuan Wu, Bertie Vidgen, Grusha Prasad, Amanpreet Singh, Pratik Ringshia, et al. "DynaBench: Rethinking benchmarking in NLP." arXiv preprint arXiv:2104.14337 (2021).
- **2022 MT-Bench** [[3]](https://arxiv.org/abs/2211.09110): Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. "Holistic evaluation of language models." arXiv preprint arXiv:2211.09110 (2022).
- **2022 BIG-bench** [[4]](https://arxiv.org/abs/2206.04615): Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. "Beyond the imitation game: Quantifying and extrapolating the capabilities of language models." arXiv preprint arXiv:2206.04615 (2022).
- **2022 GLUE-X** [[5]](https://arxiv.org/abs/2211.08073): Linyi Yang, Shuibai Zhang, Libo Qin, Yafu Li, Yidong Wang, Hanmeng Liu, Jindong Wang, Xing Xie, and Yue Zhang. "Glue-x: Evaluating natural language understanding models from an out-of-distribution generalization perspective." arXiv preprint arXiv:2211.08073 (2022).
- **2023 Xiezhi** [[6]](https://arxiv.org/abs/2306.05783): Zhouhong Gu, Xiaoxuan Zhu, Haoning Ye, Lin Zhang, Jianchen Wang, Sihang Jiang, Zhuozhi Xiong, Zihan Li, Qianyu He, Rui Xu, et al. "Xiezhi: An Ever-Updating Benchmark for Holistic Domain Knowledge Evaluation." arXiv preprint arXiv:2306.05783 (2023).
- **2023 C-Eval** [[7]](https://arxiv.org/abs/2305.08322): Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, et al. "C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models." arXiv preprint arXiv:2305.08322 (2023).
- **2023 OpenLLM** [[8]](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): HuggingFace. "Open-source Large Language Models Leaderboard." (2023).
- **2023 Chatbot Arena** [[9]](https://lmsys.org): LMSYS. "Chatbot Arena: Benchmarking LLMs in the Wild with Elo Ratings." (2023).
- **2023 AlpacaEval** [[10]](https://github.com/tatsu-lab/alpaca_eval): Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. "AlpacaEval: An Automatic Evaluator of Instruction-following Models." (2023).
- **2023 PandaLM** [[11]](https://arxiv.org/abs/2306.05087): Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang,
- **2023 PandaLM** [[12]](https://arxiv.org/abs/2306.05087): Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, et al. "PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization." arXiv preprint arXiv:2306.05087 (2023).
- **2023 BOSS** [[13]](https://arxiv.org/abs/2306.04618): Lifan Yuan, Yangyi Chen, Ganqu Cui, Hongcheng Gao, Fangyuan Zou, Xingyi Cheng, Heng Ji, Zhiyuan Liu, Maosong Sun. "Revisiting Out-of-distribution Robustness in NLP: Benchmark, Analysis, and LLMs Evaluations." arXiv:2306.04618 [cs.CL] (2023).
- **2023 KoLA** [[14]](https://arxiv.org/abs/2306.09296): Jifan Yu, Xiaozhi Wang, Shangqing Tu, Shulin Cao, Daniel Zhang-Li, Xin Lv, Hao Peng, Zijun Yao, Xiaohan Zhang, Hanming Li, et al. "KoLA: Carefully Benchmarking World Knowledge of Large Language Models." arXiv preprint arXiv:2306.09296 (2023).
- **2023 AGIEval** [[15]](https://arxiv.org/abs/2304.06364): Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, Nan Duan. "Agieval: A human-centric benchmark for evaluating foundation models." arXiv preprint arXiv:2304.06364 (2023).
- **2023 PromptBench** [[16]](https://arxiv.org/abs/2306.04528): Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, et al. "PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts." arXiv preprint arXiv:2306.04528 (2023).
- **2023 MT-Bench** [[17]](https://arxiv.org/abs/2306.05685): Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena." arXiv:2306.05685 [cs.CL] (2023).
- **2023 LLMEval2** [[18]](https://arxiv.org/abs/2308.01862): Xinghua Zhang, Bowen Yu, Haiyang Yu, Yangyu Lv, Tingwen Liu, Fei Huang, Hongbo Xu, Yongbin Li. "Wider and deeper llm networks are fairer llm evaluators." arXiv preprint arXiv:2308.01862 (2023).

### Specifics Evaluations

- **2022 MultiMedQA** [[19]](https://arxiv.org/abs/2212.13138): Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. "Large Language Models Encode Clinical Knowledge." arXiv preprint arXiv:2212.13138 (2022).
- **2023 ARB** [[20]]: Tomohiro Sawada, Daniel Paleka, Alexander Havrilla, Pranav Tadepalli, Paula Vidas, Alexander Kranias, John J. Nay, Kshitij Gupta, and Aran Komatsuzaki. "ARB: Advanced Reasoning Benchmark for Large Language Models." (2023).
- **2023 CVALUES** [[21]](https://arxiv.org/abs/2307.09705): Guohai Xu, Jiayi Liu, Ming Yan, Haotian Xu, Jinghui Si, Zhuoran Zhou, Peng Yi, Xing Gao, Jitao Sang, Rong Zhang, Ji Zhang, Chao Peng, Fei Huang, and Jingren Zhou. "CValues: Measuring the Values of Chinese Large Language Models from Safety to Responsibility." arXiv:2307.09705 [cs.CL] (2023).
- **2023 ToolBench** [[22]](https://github.com/sambanova/toolbench): ToolBench. "Open-source tools learning benchmarks." (2023).
- **2023 FRESHQA** [[23]](https://arxiv.org/abs/2310.03214): Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, and Thang Luong. "FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation." arXiv:2310.03214 [cs.CL] (2023).
- **2023 CMB** [[24]](https://arxiv.org/abs/2308.08833): Xidong Wang, Guiming Hardy Chen, Dingjie Song, Zhiyi Zhang, Zhihong Chen, Qingying Xiao, Feng Jiang, Jianquan Li, Xiang Wan, Benyou Wang, et al. "CMB: A Comprehensive Medical Benchmark in Chinese." arXiv preprint arXiv:2308.08833 (2023).
- **2023 MINT** [[25]](https://arxiv.org/abs/2305.11792): Hongru Wang, Rui Wang, Fei Mi, Zezhong Wang, Ruifeng Xu, and Kam-Fai Wong. "Chain-of-thought prompting for responding to in-depth dialogue questions with LLM." arXiv:2305.11792 [cs.CL] (2023).
- **2023 Dialogue CoT**: Related to MINT; same reference due to lack of distinct reference. (2023)
- **2023 M3Exam** [[26]](https://arxiv.org/abs/2306.05179): Wenxuan Zhang, Sharifah Mahani Aljunied, Chang Gao, Yew Ken Chia, and Lidong Bing. "M3Exam: A Multilingual Multimodal Multilevel Benchmark for Examining Large Language Models." arXiv preprint arXiv:2306.05179 (2023).
- **2023 GAOKAO-Bench** [[27]](https://arxiv.org/abs/2306.02408): Beichen Zhang, Kun Zhou, Xilin Wei, Wayne Xin Zhao, Jing Sha, Shijin Wang, and Ji-Rong Wen. "Evaluating and Improving Tool-Augmented Computation-Intensive Math Reasoning." arXiv preprint arXiv:2306.02408 (2023).
- **2023 SafetyBench** [[28]](https://arxiv.org/abs/2309.07045): Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, and Minlie Huang. "SafetyBench: Evaluating the Safety of Large Language Models with Multiple Choice Questions." arXiv preprint arXiv:2309.07045 (2023).

### Multimodal Evaluations

- **2023 MME** [[29]](https://arxiv.org/abs/2306.13394): Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. "MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models." arXiv preprint arXiv:2306.13394 (2023).
- **2023 MMBench** [[30]](https://arxiv.org/abs/2307.06281): Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. "MMBench: Is Your Multi-modal Model an All-around Player?" arXiv:2307.06281 [cs.CV] (2023).
- **2023 SEED-Bench** [[31]](https://arxiv.org/abs/2307.16125): Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. "Seed-bench: Benchmarking multimodal llms with generative comprehension." arXiv preprint arXiv:2307.16125 (2023).
- **2023 MM-Vet** [[32]](https://arxiv.org/abs/2308.02490): Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. "Mm-vet: Evaluating large multimodal models for integrated capabilities." arXiv preprint arXiv:2308.02490 (2023).
- **2023 LAMM** [[33]](https://arxiv.org/abs/2306.06687): Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi, Dingning Liu, Mukai Li, Lu Sheng, Lei Bai, Xiaoshui Huang, Zhiyong Wang, et al. "LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark." arXiv preprint arXiv:2306.06687 (2023).
- **2023 LVLM-eHub** [[34]](https://arxiv.org/abs/2306.09265): Peng Xu, Wenqi Shao, Kaipeng Zhang, Peng Gao, Shuo Liu, Meng Lei, Fanqing Meng, Siyuan Huang, Yu Qiao, and Ping Luo. "LVLM-eHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models." arXiv:2306.09265 [cs.CV] (2023).

# Large Models

## Language Models
- **2018**: BERT, GPT-1
- **2019**: DistilBERT, BART, T5, RoBERTa, GPT-2
- **2020**: Wav2Vec2, GPT-3
- **2021**: HuBERT, GLM
- **2022**: Instruct GPT, PaLM
- **2023**: Whisper, LLaMA, GPT-4, PaLM2, Claude2

## Vision Models
- **2020**: DETR, ImageGPT
- **2021**: ViT, DeiT, YOLOS
- **2022**: Swin Transformer, ViTDet, ViTMAE, BEiT
- **2023**: DINOv2, LVM

## Multimodal Models
- **2020**: UniTER
- **2021**: CLIP, ALIGN, ALBEF, DALL-E, Stable Diffusion
- **2022**: FLAVA, GLIP, Flamingo, CoCa
- **2023**: PaLM-E, Kosmos, CODi, NExT-GPT, LLavA, SAM, ImageBind, BLIP2, Minigpt-4, EMU2, GPT4V, Gemini
- **2024**: Sora, EMO

# Evaluation Leaderboards
## GLUE 2018 Leaderboard
| Rank | Team Name                     | Model                     | URL Score | CoLA  | SST-2 | MRPC      | STS-B     | QQP       |
|------|-------------------------------|---------------------------|-----------|-------|-------|-----------|-----------|-----------|
| 1    | Microsoft Alexander v-team    | Turing ULR v6             | 91.3      | 73.3  | 97.5  | 94.2/92.3 | 93.5/93.1 | 76.4/90.9 |
| 2    | JDExplore d-team              | Vega v1                   | 91.3      | 73.8  | 97.9  | 94.5/92.6 | 93.5/93.1 | 76.7/91.1 |
| 3    | Microsoft Alexander v-team    | Turing NLR v5             | 91.2      | 72.6  | 97.6  | 93.8/91.7 | 93.7/93.3 | 76.4/91.1 |
| 4    | DIRL Team                     | DeBERTa + CLEVER          | 91.1      | 74.7  | 97.6  | 93.3/91.1 | 93.4/93.1 | 76.5/91.0 |
| 5    | ERNIE Team - Baidu            | ERNIE                     | 91.1      | 75.5  | 97.8  | 93.9/91.8 | 93.0/92.6 | 75.2/90.9 |
| 6    | AliceMind & DIRL              | StructBERT + CLEVER       | 91        | 75.3  | 97.7  | 93.9/91.9 | 93.5/93.1 | 75.6/90.8 |
| 7    | DeBERTa Team - Microsoft      | DeBERTa / TuringNLRv4     | 90.8      | 71.5  | 97.5  | 94.0/92.0 | 92.9/92.6 | 76.2/90.8 |
| 8    | HFL iFLYTEK                   | MacALBERT + DKM           | 90.7      | 74.8  | 97    | 94.5/92.6 | 92.8/92.6 | 74.7/90.6 |
| 9    | PING-AN Omni-Sinitic          | ALBERT + DAAF + NAS       | 90.6      | 73.5  | 97.2  | 94.0/92.0 | 93.0/92.4 | 76.1/91.0 |
| 10   | T5 Team - Google              | T5                        | 90.3      | 71.6  | 97.5  | 93.1/92.0 | 75.1/90.6 | 75.1/90.6 |

## Benchmark Tasks

### Single-sentence tasks:
- **CoLA:** Determine if an English sentence is grammatically correct.
- **SST-2:** Determine if the sentiment of a movie review sentence is positive or negative.

### Similarity and paraphrase tasks:
- **MRPC:** Judge whether two sentences are equivalent in meaning.
- **QQP:** Determine whether two Quora questions are semantically equivalent.
- **STS-B:** Evaluate the similarity between two sentences on a scale of 1 to 5.

### Inference tasks:
- **MNLI:** Determine whether the hypothesis sentence is entailment, contradiction, or neutral with respect to the given premise sentence.
- **QNLI:** Convert question-answering problems into sentence pair classification problems.
- **RTE:** Determine whether one sentence entails another.
- **WNLI:** Identify the correct reference of ambiguous pronouns in sentences.

## SuperGLUE 2019 Leaderboard

| Rank | Team Name                    | Model                          | URL Score | BoolQ    | CB  | COPA | MultiRC   | ReCoRD | RTE  | WiC  | WSC | Ax-b        |
|------|------------------------------|--------------------------------|-----------|----------|-----|------|-----------|--------|------|------|-----|-------------|
| 1    | JDExplore d-team             | Vega v2                        | 91.3      | 90.5/86.9 | 89.2 | 94.4 | 90.7/44.9 | 98     | 76.4 | 90.1 | 100 | 100.0/50.0  |
| 2    | Liam Fedus                   | ST-MoE-32B                     | 91.2      | 92.4/96.9 | 89.2 | 94.4 | 93.5/77.7 | 99.5   | 76.6 | 92.6 | 96.1| 96.2/94.1  |
| 3    | Microsoft Alexander v-team   | Turing NLR v5                  | 90.9      | 92.0/95.9 | 88.2 | 94.1 | 94.1/71.9 | 97.1   | 97.3 | 93.3 | 95.5| 95.3/95.5  |
| 4    | ERNIE Team - Baidu           | ERNIE 3.0                      | 90.6      | 91.0/89.9 | 88.6 | 94.7 | 92.2/74.4 | 97.3   | 77.4 | 92.6 | 97.4| 92.6/94.7  |
| 5    | Ty Tay                       | PaLM 540B                      | 90.4      | 91.9/94.4 | 88.7 | 94.3 | 91.0/72.3 | 97.4   | 77.9 | 95.2 | 95  | 92.5/95.0  |
| 6    | Zirui Wang                   | T5 + UDG, Single Model (Google Brain) | 90.4 | 91.4/85.9 | 88.3 | 94.2 | 92.3/59.5 | 96.6   | 76.9 | 92.1 | 96.1| 92.1/79.1  |
| 7    | DeBERTa Team - Microsoft     | DeBERTa / TuringNLRv4          | 90.3      | 90.4/95.7 | 88.4 | 94.5 | 94.1/53.2 | 97.5   | 77.9 | 95.6 | 93.8| 96.3/93.8  |
| 8    | SuperGLUE Human Baselines    | SuperGLUE Human Baselines      | 89.8      | 89.0/95.8 | 81.8 | 91.7 | 93.6/80.0 | 100    | 75.6 | 96.6 | 99.9| 96.9/99.7  |
| 9    | T5 Team - Google             | T5                             | 89.3      | 91.2/93.9 | 84.8 | 93.4 | 93.4/52.5 | 96.8   | 83.8 | 92.5 | 78.9| 92.6/79.1  |
| 10   | SPoT Team - Google           | Frozen T5 1.1 + SPoT           | 89.2      | 91.1/95.8 | 96.5 | 93.9 | 92.4/75.8 | 96.7   | 88.3 | 85.7 | 85.7| 85.3/82.6  |

## Benchmark Tasks

- **BoolQ:** Evaluate sentence judgment and the relatedness of sentence pairs. Determine whether the answer to a question based on a passage is "Yes" or "No".

- **CB:** Perform inference judgment and sentence pair relevance determination. Each instance includes a sentence pair tagged with one of the following labels: [entailment | contradiction | neutral].

- **COPA:** Execute inference judgment by choosing one out of two sentences that best relates logically to a given premise.

- **MultiRC:** Assess sentence judgment and answer multiple-choice questions. For each question related to a passage, indicate whether each proposed answer is True or False.

- **ReCoRD:** Address sentence completion to identify the correct entity from a passage as the answer to a given question, without a set of options, requiring an exact match from the passage.

- **RTE:** Engage in sentence judgment by evaluating whether one sentence can logically infer the content of another sentence.

- **WiC:** Judge the meaning of a word to determine whether the same word used in two different sentences has the same meaning.

- **WSC:** Analyze meaning judgment by resolving which entity in a sentence a given pronoun refers to.
