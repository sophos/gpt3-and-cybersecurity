# Sophos AI GPT-3 for Cybersecurity Repository

A key lesson of recent deep learning successes is that as we scale neural networks, they get better, and sometimes in game-changing ways.
This repo provides two applications which demonstrate how GPT-3 opens new vistas for cybersecurity.

## How do I get started?

There are two use cases for spam detection and command analysis.
As the code in this repo uses OpenAI API, set the OPENAI_API_KEY enviroment variable as your api key.
Refer to the OpenAI documentation in https://beta.openai.com/docs/introduction.

### Spam detector
Spam detector demonstrates how to identify spam messages using GPT-3 few-shot learning or fine-tuning.

Change directory to spam_detector folder and follow the [instructions](./spam_detector/README.md).


### Command analyzer
Command analyzer shows how to analyzer complex command lines using GPT-3 few-shot learning.

Change directory to command_analyzer folder and follow the [instructions](./command_analyzer/README.md).


## How do I cite GPT-3 for Cybersecurity?

*Questions, ideas, feedback appreciated, please email younghoo.lee@sophos.com*

@misc{Lee2022,
  author = {Lee, Younghoo},
  title = {GPT-3 for Cybersecurity},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sophos-ai/gpt3-cybersecurity/}}
}
