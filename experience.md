# Work Experience

**Goldman Sachs -- Quant Researcher** 2024 - Present
- Develop Goldman Sachs’ first AI Scientist, a multi-agent framework that autonomously conducts end-to-end quant research
- Implement evaluation loop where (1) a Research Agent initiates a multi-step plan given existing literature, a research question, and a dataset, (2) an Implementation Agent implements and debugs code in a controlled Jupyter notebook environment, and (3) the Research Agent interprets the results, reflects on the plan, and explores future research directions
- Design code correctness and statistical model verification systems to ensure research reproducibility
- Built a distributed risk monitoring software in Python using Dask & Asyncio to process 70k+ daily portfolio optimizations and systematically flag high-risk assets, reducing runtime from ~3 hours to 20 mins and directly impacting PnL by $10M+
- Engineered data pipeline in Python to convert 1000+ daily free-text unstructured client inputs into structured formats, decrease error rate from ~5% to <0.1%, increasing PnL by $5M+ and reducing 1000+ hours yearly of manual data entry
- Implemented async architecture in Python to query 10M+ rows of position data, cutting runtime from 4 hours to 5 mins
- Training Laplacian graph learning models to model cross-asset momentum effects

**Stealth Startup -- Research Scientist Intern** 2021 - 2021
- Built attention-based LSTMs in PyTorch for an AI shopping agent, achieving 99% test accuracy and surpassing prior models
- Built cleaning/preprocessing pipeline for e-commerce corpus; added dropout, L2, and data augmentation to boost generalization
- Created interpretable attention visualizations; deployed model via an API that was used in production ML system

**UCI NLP Group -- Research Scientist** 2017 - 2020
- Identified faulty moderation in online forums on sensitive discussions about race and gender, which inspired an independent research project on AI for safer debate; awarded Top 10 Regeneron STS Finalist, presented work at RE-WORK conference
- Built attention-based LSTM classifier with GloVe embeddings in Tensorflow/Keras and Python preprocessing/tokenizing pipeline
- Collaborated with Prof. Sameer Singh (UCI NLP) on a debiasing data augmentation process, reducing false positive bias by 44%
