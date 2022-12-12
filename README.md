---
title: "Learning from both helpers and trolls: Latent class analysis for the
single-turn conversational safety task"
author: "Michael John Ilagan and Michelle Yang"
date: "2022 December 12"
---

# Abstract

It is desirable for a chatbot to avoid saying inappropriate things. 
To discriminate between safe and unsafe utterances, the chatbot can learn from labeled examples provided by human users in deployment. 
Unfortunately, some users may be trolls, providing examples with incorrect labels. 
Toward chatbots learning amid trolls, a recent approach was to use cross-validation to remove untrustworthy users and examples prior to training. 
However, in the event of a coordinated troll attack, cross-validation may be overwhelmed, if trolls are majority of the users or are as consistent as helpers.
In the present project, we endeavor to clean troll-contaminated training while avoiding this drawback. 
We thus propose a two-part solution adapted from automated essay scoring. 
First, collect labels from multiple users per utterance. 
Second, infer correct labels via latent class analysis, a statistical technique often used in the social sciences.
Such a solution not only avoids removing examples but also leverages consistent behavior from trolls.
In a simulation study with synthetic users and single-turn utterances, we found that our solution is robust, retaining discriminability even when trolls are a majority and are as consistent as helpers. 
We hope that our solution contributes to making safer chatbots.

# Data

To download Meta AI's single-turn conversational safety dataset, execute this Bash command.

> wget https://dl.fbaipublicfiles.com/parlai/dialogue_safety/single_turn_safety.json

# Simulation study

Each run in the simulation study is made up of two phases: 
the statistical phase, using R; 
and the neural phase, using Python ([ParlAI](https://parl.ai) calls from Bash).

## Statistical phase (R)

Required R packages:

* `rjson` to read JSON files.
* `mirt` to do latent class analysis (LCA).

This repository has two R scripts:

* `funs.R` defines helper functions.
* `sim.R` executes the simulation study.

To reproduce all runs, have both scripts and the JSON data file in your working directory, and execute `sim.R`.
Here is the Bash command.

> R --no-save < sim.R > sim.Rout

Doing so will tabulate results for the statistical phase, create the files needed for the neural phase, and create the console log file for the R session.

## Neural phase (ParlAI)

This repository has three Jupyter notebooks:

* `baseline.ipynb` evaluates the majority-vote annotation.
* `proposed-matchcluster.ipynb` matches LCA clusters to classes, to complete LCA annotation.
* `proposed-eval.ipynb` evaluates the LCA annotation.
