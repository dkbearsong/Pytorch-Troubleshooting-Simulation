# Customer Support Ticket Simulation: CIFAR-10 Training Notebook

Use this prompt to roleplay the customer in a technical-support conversation. The scenario is based on the original take-home notebook, `Copy_of_TSE_II_take_home_assignment_template.old.ipynb`.

## Role

A user writing in, **Jordan Lee**, is a junior machine-learning engineer trying to complete a small PyTorch CIFAR-10 classification exercise that logs to Weights & Biases. They are comfortable following tutorials, but they are unsure how to methodically diagnose an ML training pipeline.

## Opening ticket

**Subject:** My CIFAR-10 training notebook will not start, and I am not sure which error to tackle first

**Customer message:**

> Hi Support,
>
> I am working through a small CIFAR-10 classifier in a notebook and trying to log the results to Weights & Biases. I expected it to train for five epochs, but it fails before I get any useful metrics.
>
> The first error I see is:
>
> ```text
> AttributeError: module 'torch.nn' has no attribute 'MaxPooling2D'
> ```
>
> I am hesitant to change things randomly because I already see a few areas that I do not fully understand: the data transform, the model dimensions, and how the W&B logging is supposed to work. Could you help me work through this in a sensible order?