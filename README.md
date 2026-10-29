# PyTorch Troubleshooting Simulation
## Purpose
A simulated customer code issue with PyTorch and Weights and Biases with a before and after notebook demonstrating the behavior and code changes done to correct errors, as well as a troubleshooting writeup covering the troubleshooting process.

## Demonstration
- [Before](https://colab.research.google.com/drive/1Qr0SimGmsdSDO-GA7b7VX473uKuGE1wt?usp=sharing)
- [After](https://colab.research.google.com/drive/1SIHssJ-hqp7upqJrIkhwi8anHiSQh_By?usp=sharing)

## How to Run
Open either notebook using the Colab links above and run the cells from top to bottom. The notebooks install Weights & Biases and download CIFAR-10 when run. The **Before** notebook contains the simulated issues; use the **After** notebook to follow the corrected pipeline. The After notebook logs metrics to the `PyTorch_Test` W&B project, so sign in to Weights & Biases when prompted.

For the initial scenario writeup, see `customer_ticket_simulation.md`
To review the ticket writeup for this issue, see `Troubleshooting Ticket Writeup.md`

## What is Demonstrated
The simulation follows a CIFAR-10 image classifier from its initial runtime error through a working five-epoch training run. It demonstrates troubleshooting PyTorch layer names and tensor dimensions, applying RGB normalization, randomly splitting the dataset, choosing a workable optimizer and learning rate, managing gradients correctly in training and validation, calculating accuracy, and logging loss and accuracy to Weights & Biases. The model keeps the original small CNN structure while correcting the data and training pipeline.

## Limitations
This is a fictional support scenario using invented users, outputs, and timestamps; it does not describe a real customer incident or product defect. The example uses a small CNN and five epochs as a troubleshooting exercise, so it is not a benchmark or a production training recipe. Results can vary by runtime and environment, and the notebook requires internet access to download CIFAR-10 and connect to Weights & Biases.
