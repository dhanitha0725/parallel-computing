## Parallel and High Performance Computing

## Parallel Classification Using the SUSY Dataset Group Assignment

## Introduction

The Standard Model of particle physics describes many of the known fundamental par- ticles and their interactions. However, it does not completely explain several important observations in modern physics, including the nature of dark matter.

Supersymmetry, commonly known as SUSY, is a theoretical extension of the Standard Model. It proposes that each known particle has a corresponding partner particle with different spin properties. Although supersymmetric particles have not yet been experi- mentally confirmed, particle-physics experiments search for collision patterns that may indicate their existence.

The SUSY dataset is a Monte Carlo simulated dataset containing particle-collision events. Monte Carlo simulation uses repeated random sampling based on physical models and probability distributions to generate possible collision events. Therefore, the dataset does not directly contain events collected from a physical detector. Instead, it contains simulated examples of possible SUSY signal events and ordinary background events.

Each record in the dataset represents one simulated collision event. It contains 18 numerical features describing the characteristics of the event and a known class label.

- Class 1 – SUSY Signal: an event representing a process that may produce su- persymmetric particles.

- Class 0 – Background: an event representing an ordinary non-SUSY background process.

The complete SUSY dataset contains approximately five million simulated collision events. For this assignment, students must use the common 0.5% subset provided by the lecturer.

The dataset is available at:

[https://www.kaggle.com/datasets/janus137/supersymmetry-dataset](https://www.kaggle.com/datasets/janus137/supersymmetry-dataset)

## Classification Problem

The objective is to develop a binary-classification model that can distinguish between simulated SUSY signal events and simulated background events.


The labelled records must be divided into training and testing subsets. The training records are used by the selected classification algorithm to learn, calculate, or store the patterns associated with the two classes. The testing records are used to evaluate whether the developed model can correctly classify previously unseen events.

When a new collision event containing the same 18 numerical features is provided to the model, it should predict whether the event belongs to:

- the SUSY signal class; or

- the background class.

This prediction does not confirm the discovery of a supersymmetric particle. It only indicates whether the numerical characteristics of the event are more similar to the sim- ulated SUSY signal class or the simulated background class.

## Assignment Task

Each group must develop:

- 1. a sequential implementation of a classification model; and

- 2. a parallel implementation of the same classification model.

Students may select any suitable classification algorithm, such as:

- K-Nearest Neighbours;

- Logistic Regression;

- Naive Bayes;

- Decision Tree;

- Linear Classifier;

- Perceptron;

- a simple Neural Network; or

- another suitable classification algorithm.

The parallel implementation may use:

- OpenMP;

- MPI; or

- a combination of OpenMP and MPI.

Each group must identify the computationally significant stage of the selected algo- rithm and parallelize that stage. Depending on the selected algorithm, this may be:

- model training;

- prediction;

- distance calculation;


- gradient calculation;

- statistical calculation; or

- another computationally intensive operation.

Students must measure the execution time of the computational stage selected for parallelization and compare the sequential and parallel implementations.

The speedup must be calculated as:

The parallel efficiency must be calculated as:

where:

- Ts is the sequential execution time;

- Tp is the parallel execution time; and

- p is the number of threads or processes.

Students should test the parallel implementation using different numbers of threads or processes and explain the observed performance.

The main objective is not to develop the most accurate machine-learning model. The main objective is to correctly apply and evaluate parallel and high-performance com- puting concepts.

## Use of Generative AI

Generative AI tools may be used to assist with:

- selecting a classification algorithm;

- generating the initial classification implementation;

However, all group members must understand:

- how the selected classification algorithm works;

- how the dataset is used;

- what is being classified;

- which computational stage is parallelized;

- why OpenMP, MPI, or a hybrid approach was selected;

- how work is distributed among threads or processes;

- how execution time is measured;

- how speedup and efficiency are calculated; and

- why the observed performance was obtained.


Students are responsible for testing, verifying, modifying, and correcting all AI-generated code and content. Failure to explain the submitted implementation may result in a re- duction of marks.

## Group Size

Each group shall consist of three to four students.

## Deliverables

- 1. Complete source code

- 2. Video evidence showing the execution and generated results

- 3. Five-minute presentation explaining the strategy followed
