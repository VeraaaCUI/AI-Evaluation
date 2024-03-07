# A Comprehensive Overview of Evaluation Methods of Large Models

## Evaluation Methods

### General Evaluations

- **2020 MMLU** [[70]](https://arxiv.org/abs/2009.03300): Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt. "Measuring massive multitask language understanding." arXiv preprint arXiv:2009.03300 (2020).
- **2021 DynaBench** [[94]](https://arxiv.org/abs/2104.14337): Douwe Kiela, Max Bartolo, Yixin Nie, Divyansh Kaushik, Atticus Geiger, Zhengxuan Wu, Bertie Vidgen, Grusha Prasad, Amanpreet Singh, Pratik Ringshia, et al. "DynaBench: Rethinking benchmarking in NLP." arXiv preprint arXiv:2104.14337 (2021).
- **2022 MT-Bench** [[114]](https://arxiv.org/abs/2211.09110): Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. "Holistic evaluation of language models." arXiv preprint arXiv:2211.09110 (2022).
- **2022 BIG-bench** [[182]](https://arxiv.org/abs/2206.04615): Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. "Beyond the imitation game: Quantifying and extrapolating the capabilities of language models." arXiv preprint arXiv:2206.04615 (2022).
- **2022 GLUE-X** [[234]](https://arxiv.org/abs/2211.08073): Linyi Yang, Shuibai Zhang, Libo Qin, Yafu Li, Yidong Wang, Hanmeng Liu, Jindong Wang, Xing Xie, and Yue Zhang. "Glue-x: Evaluating natural language understanding models from an out-of-distribution generalization perspective." arXiv preprint arXiv:2211.08073 (2022).
- **2023 Xiezhi** [[59]](https://arxiv.org/abs/2306.05783): Zhouhong Gu, Xiaoxuan Zhu, Haoning Ye, Lin Zhang, Jianchen Wang, Sihang Jiang, Zhuozhi Xiong, Zihan Li, Qianyu He, Rui Xu, et al. "Xiezhi: An Ever-Updating Benchmark for Holistic Domain Knowledge Evaluation." arXiv preprint arXiv:2306.05783 (2023).
- **2023 C-Eval** [[78]](https://arxiv.org/abs/2305.08322): Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, et al. "C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models." arXiv preprint arXiv:2305.08322 (2023).
- **2023 OpenLLM** [[80]](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): HuggingFace. "Open-source Large Language Models Leaderboard." (2023).
- **2023 Chatbot Arena** [[128]](https://lmsys.org): LMSYS. "Chatbot Arena: Benchmarking LLMs in the Wild with Elo Ratings." (2023).
- **2023 AlpacaEval** [[112]](https://github.com/tatsu-lab/alpaca_eval): Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. "AlpacaEval: An Automatic Evaluator of Instruction-following Models." (2023).
- **2023 PandaLM** [[216]](https://arxiv.org/abs/2306.05087): Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang,
- **2023 PandaLM** [[216]](https://arxiv.org/abs/2306.05087): Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, et al. "PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization." arXiv preprint arXiv:2306.05087 (2023).
- **2023 BOSS** [[239]](https://arxiv.org/abs/2306.04618): Lifan Yuan, Yangyi Chen, Ganqu Cui, Hongcheng Gao, Fangyuan Zou, Xingyi Cheng, Heng Ji, Zhiyuan Liu, Maosong Sun. "Revisiting Out-of-distribution Robustness in NLP: Benchmark, Analysis, and LLMs Evaluations." arXiv:2306.04618 [cs.CL] (2023).
- **2023 KoLA** [[236]](https://arxiv.org/abs/2306.09296): Jifan Yu, Xiaozhi Wang, Shangqing Tu, Shulin Cao, Daniel Zhang-Li, Xin Lv, Hao Peng, Zijun Yao, Xiaohan Zhang, Hanming Li, et al. "KoLA: Carefully Benchmarking World Knowledge of Large Language Models." arXiv preprint arXiv:2306.09296 (2023).
- **2023 AGIEval** [[262]](https://arxiv.org/abs/2304.06364): Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, Nan Duan. "Agieval: A human-centric benchmark for evaluating foundation models." arXiv preprint arXiv:2304.06364 (2023).
- **2023 PromptBench** [[264]](https://arxiv.org/abs/2306.04528): Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, et al. "PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts." arXiv preprint arXiv:2306.04528 (2023).
- **2023 MT-Bench** [[260]](https://arxiv.org/abs/2306.05685): Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena." arXiv:2306.05685 [cs.CL] (2023).
- **2023 LLMEval2** [[252]](https://arxiv.org/abs/2308.01862): Xinghua Zhang, Bowen Yu, Haiyang Yu, Yangyu Lv, Tingwen Liu, Fei Huang, Hongbo Xu, Yongbin Li. "Wider and deeper llm networks are fairer llm evaluators." arXiv preprint arXiv:2308.01862 (2023).

### Specifics Evaluations

- **2022 MultiMedQA** [[177]](https://arxiv.org/abs/2212.13138): Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. "Large Language Models Encode Clinical Knowledge." arXiv preprint arXiv:2212.13138 (2022).
- **2023 ARB** [[171]]: Tomohiro Sawada, Daniel Paleka, Alexander Havrilla, Pranav Tadepalli, Paula Vidas, Alexander Kranias, John J. Nay, Kshitij Gupta, and Aran Komatsuzaki. "ARB: Advanced Reasoning Benchmark for Large Language Models." (2023).
- **2023 CVALUES** [[230]](https://arxiv.org/abs/2307.09705): Guohai Xu, Jiayi Liu, Ming Yan, Haotian Xu, Jinghui Si, Zhuoran Zhou, Peng Yi, Xing Gao, Jitao Sang, Rong Zhang, Ji Zhang, Chao Peng, Fei Huang, and Jingren Zhou. "CValues: Measuring the Values of Chinese Large Language Models from Safety to Responsibility." arXiv:2307.09705 [cs.CL] (2023).
- **2023 ToolBench** [[191]](https://github.com/sambanova/toolbench): ToolBench. "Open-source tools learning benchmarks." (2023).
- **2023 FRESHQA** [[198]](https://arxiv.org/abs/2310.03214): Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, and Thang Luong. "FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation." arXiv:2310.03214 [cs.CL] (2023).
- **2023 CMB** [[211]](https://arxiv.org/abs/2308.08833): Xidong Wang, Guiming Hardy Chen, Dingjie Song, Zhiyi Zhang, Zhihong Chen, Qingying Xiao, Feng Jiang, Jianquan Li, Xiang Wan, Benyou Wang, et al. "CMB: A Comprehensive Medical Benchmark in Chinese." arXiv preprint arXiv:2308.08833 (2023).
- **2023 MINT** [[213]](https://arxiv.org/abs/2305.11792): Hongru Wang, Rui Wang, Fei Mi, Zezhong Wang, Ruifeng Xu, and Kam-Fai Wong. "Chain-of-thought prompting for responding to in-depth dialogue questions with LLM." arXiv:2305.11792 [cs.CL] (2023).
- **2023 Dialogue CoT**: Related to MINT; same reference due to lack of distinct reference. (2023)
- **2023 M3Exam** [[250]](https://arxiv.org/abs/2306.05179): Wenxuan Zhang, Sharifah Mahani Aljunied, Chang Gao, Yew Ken Chia, and Lidong Bing. "M3Exam: A Multilingual Multimodal Multilevel Benchmark for Examining Large Language Models." arXiv preprint arXiv:2306.05179 (2023).
- **2023 GAOKAO-Bench** [[245]](https://arxiv.org/abs/2306.02408): Beichen Zhang, Kun Zhou, Xilin Wei, Wayne Xin Zhao, Jing Sha, Shijin Wang, and Ji-Rong Wen. "Evaluating and Improving Tool-Augmented Computation-Intensive Math Reasoning." arXiv preprint arXiv:2306.02408 (2023).
- **2023 SafetyBench** [[254]](https://arxiv.org/abs/2309.07045): Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, and Minlie Huang. "SafetyBench: Evaluating the Safety of Large Language Models with Multiple Choice Questions." arXiv preprint arXiv:2309.07045 (2023).

### Multimodal Evaluations

- **2023 MME** [[46]](https://arxiv.org/abs/2306.13394): Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. "MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models." arXiv preprint arXiv:2306.13394 (2023).
- **2023 MMBench** [[126]](https://arxiv.org/abs/2307.06281): Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. "MMBench: Is Your Multi-modal Model an All-around Player?" arXiv:2307.06281 [cs.CV] (2023).
- **2023 SEED-Bench** [[107]](https://arxiv.org/abs/2307.16125): Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. "Seed-bench: Benchmarking multimodal llms with generative comprehension." arXiv preprint arXiv:2307.16125 (2023).
- **2023 MM-Vet** [[238]](https://arxiv.org/abs/2308.02490): Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. "Mm-vet: Evaluating large multimodal models for integrated capabilities." arXiv preprint arXiv:2308.02490 (2023).
- **2023 LAMM** [[235]](https://arxiv.org/abs/2306.06687): Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi, Dingning Liu, Mukai Li, Lu Sheng, Lei Bai, Xiaoshui Huang, Zhiyong Wang, et al. "LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark." arXiv preprint arXiv:2306.06687 (2023).
- **2023 LVLM-eHub** [[231]](https://arxiv.org/abs/2306.09265): Peng Xu, Wenqi Shao, Kaipeng Zhang, Peng Gao, Shuo Liu, Meng Lei, Fanqing Meng, Siyuan Huang, Yu Qiao, and Ping Luo. "LVLM-eHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models." arXiv:2306.09265 [cs.CV] (2023).

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
