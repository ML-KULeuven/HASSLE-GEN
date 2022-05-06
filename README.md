# HASSLE-GEN
This repository contains the code for our paper:
*Senne Berden, Mohit Kumar, Samuel Kolb, and Tias Guns (2022): Learning MAX-SAT Models from Examples using Genetic Algorithms and Knowledge Compilation, CP 2022*

## Overview of the code
The code is structured as follows:

```bash
├── __init__.py
├── examples                      # Directory containing some examples that illustrate how to start a run of HASSLE-SLS or HASSLE-GEN
├── experiments                   # Directory containing the experiments that made it into the paper, along with the associated data and results
├── hassle_sls                    # Directory containing the code associated with HASSLE-SLS (also see https://github.com/mohitKULeuven/HassleWithLocalSearch)
├── tests                         # Directory containing tests
├── auxiliary.py                  # Auxiliary functions that get called from various places
├── crossover.py                  # Various crossover operators
├── evaluation.py                 # Code related to the evaluation of MAX-SAT models
├── generation_script             # Example Bash script illustrating how to generate synthetic learning problems and target models
├── generator.py                  # Auxiliary code for the generation of synthetic learning problems and target models
├── genetic.py                    # General GeneticAlgorithm class for the learning of MAX-SAT models from labeled examples
├── hassle_gen.py                 # The HassleGen class, a subclass of GeneticAlgorithm, that contains the high-level HASSLE-GEN loop (Algorithm 2 in the paper) 
├── knowledge_compilation.py      # Code related to the use of knowledge-compilation in the evaluation of MAX-SAT models. Used in evaluation.py
├── learn_max_sat_model.py        # Auxiliary layer between experiment files and hassle_gen.py. Constructs, configures and runs an instance of the HassleGen class based on provided hyperparameters
├── mutation.py                   # Various mutation operators
├── observe.py                    # Code for Observers, which, when along to a run of HASSLE-SLS or HASSLE-GEN, keep track of various statistics throughout the iterations/generations. Useful for plotting
├── plotting.py                   # Code that takes (collections of) observers and generates various kinds of plots
├── reporting.py                  # Code that takes (collections of) observers and generates a report as a CSV file 
└── synthetic.py                  # Auxiliary layer between generation_script and generator.py. Takes command line input and calls the appropriate code from generator.py
```

## Using the code
The examples and experiments come equipped with training data and target models (as pickle files) and can be ran directly.

To create custom learning problems, first alter the *generation_script* file as desired, and then execute it using the `bash generation_script` command. This creates a directory named *pickles* in the project's top-level directory. To use the generated learning problem(s) in an example or an experiment, replace the contents of the example's or experiment's pickles folder with the contents of the newly created top-level pickles folder.
