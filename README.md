# EAB-FL
This repository contains the implementation of the IJCAI-2024 Accepted Paper "EAB-FL: Exacerbating Algorithmic Bias Through Model Poisoning Attacks in Federated Learning" by Syed Irfan Ali Meerza, and Jian Liu. 

## Introduction

We introduce a novel model poisoning attack, EAB-FL, designed to exacerbate group unfairness in FL systems while maintaining model utility. While existing attacks on Federated Learning (FL) primarily target model accuracy, they often overlook the potential to intensify model unfairness. To fully understand the attack surface of the FL framework, our proposed attack focuses on revealing and exploiting these vulnerabilities.

## Structure

```bash
EAB-FL/
├── data_utils.py            # Contains dataset-related classes and data transformation functions
├── eab_fl.py                # Functions for handling malicious clients and inducing model poisoning
├── influence.py             # Functions for calculating the influence score of data samples
├── quick_start.py           # Main script to execute the federated learning process
├── model_utils.py           # Model training, evaluation, and federated averaging functions
└── requirements.txt         # Required Python packages
```

## Quick Start

Use the ```QuickStart.py``` script for a quick start.
In the script, you can find the minimal implementation for attacking the FL model trained on the CelebA dataset.

