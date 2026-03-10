# Image Caption extension for Stable Diffusion Webui 👁️📜🖋️
[[paper]](paper.pdf)

The project explores integrating Large Concept Models into image captioning, achieving improved semantic fidelity and generalization over traditional GPT-2 baselines. The extension enables practical deployment of these models for dataset creation and real-world use in Stable Diffusion WebUI.

<img src="Screenshot.png" alt="GitHub Logo" style="width: 100%;">

## Results

Frozen LM setting

| Model         | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE | CIDEr | BERTScore |
|--------------|--------|--------|--------|--------|--------|-------|-------|-----------|
| GPT-2        | 0.5245 | 0.3516 | 0.2285 | 0.1460 | 0.1961 | 0.4844 | 0.4384 | 0.8922 |
| Cocomix      | 0.5847 | 0.4016 | 0.2647 | 0.1737 | 0.2033 | 0.4982 | 0.4660 | 0.8953 |
| Cocomix+PCE  | 0.6221 | 0.4267 | 0.2822 | 0.1841 | 0.2052 | 0.5060 | 0.4587 | 0.8967 |

Cocomix+PCE achieves the best results in the frozen language model setting, demonstrating the benefit of Prefix Concept Extraction for semantic alignment and caption quality.

## Installation

Require A1111 WebUI, paste the git link to install this extension