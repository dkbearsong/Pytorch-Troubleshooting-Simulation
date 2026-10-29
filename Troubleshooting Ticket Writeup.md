---
Project Type: Fictional Troubleshooting Simulation
Ticket Review By: Dereck Goolsby-Bearsong
Product: Weights and Biases
Data: All users, documents, IDs, outputs, and timestamps below are invented. No customer or employer data was used.
tags:
  - project
  - portfolio
  - simulation
---

## Summary
A fictional scenario where a customer is having issues with their CIFAR.10 training using PyTorch and Weights & Biases. The user has made several errors, which may or may not be apparent, and needs troubleshooting. I tested the code and reproduced errors. In this simulation, the code errors out when you try to run it. 

I traced the errors through the code and corrected them to make the model learn reasonably well while maintaining the original structure of the code.
## Reported Issue
##### Customer Message
> Hi Support,
>
> I am working through a small CIFAR-10 classifier in a notebook and trying to log the results to Weights & Biases. I expected it to train for five epochs, but it fails before I get any useful metrics.
>
> The first error I see is:
>
> ```
> AttributeError: module 'torch.nn' has no attribute 'MaxPooling2D'
> ```
>
> I am hesitant to change things randomly because I already see a few areas that I do not fully understand: the data transform, the model dimensions, and how the W&B logging is supposed to work. Could you help me work through this in a sensible order?

##### Expected behavior
Should train on the dataset for 5 Epochs

##### Simulated Actual Behavior
User receives an error when running the notebook
```
AttributeError: module 'torch.nn' has no attribute 'MaxPooling2D'
```
## Investigation
- Following the error message, the first problem immediately found is in the 5th cell in the class CifarClassifier. Two errors can be quickly spotted here
```
self.downsample = nn.MaxPoolings2D(2, 2)
...
batch = torch.Flattening(batch, 1)
```

Both of these are using incorrect module names. `nn.MaxPoolings2D` should be `nn.MaxPool2d`, and `torch.Flattening` should be `torch.flatten`
- Now when run, we get to "Training and Validation". We get this error
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x400 and 600x120)
wandb: ERROR The nbformat package was not found. It is required to save notebook history.
```

The w&b error is pretty easy, just need to run `pip install nbformat` and that will correct it.
- For the `RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x400 and 600x120)` error, going back to the `class CifarClassifier`, we can follow the math through the forward step. 
	- CIFAR-10 images are 32x32 with 3 bit channel
	- Per https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html when running the initial 2D convolution layer, `self.feature_one = nn.Conv2d(3, 6, 5)`, the formula becomes output size = input size - kernel size + 1. for conv1 this is input size of 32 - kernel size of 5 + 1. 32 - 5 + 1 = 28. Shape after conv1 is the 6 channels and new size of 28 x 28, so (6, 28, 28)
	- Then it goes through F.relu to take the max of either 0 or the output number in the last step
	- Then it goes to `self.downsample = nn.MaxPool2d(2, 2)`, so a 2x2 pool with stride 2 which halves the height and width (((input - kernel)/stride)+1) so ((28 - 2)/2) + 1 = 14 , but keeps the number of channels, so turns it into (6, 14, 14)
	- Next line, `x = self.downsample(F.relu(self.feature_two(x)))`
	- `self.feature_two = nn.Conv2d(6, 16, 5)` so we feed in (6,14,14) in (14x14), then run `conv2d` which does (14-5)/1 + 1 which leads to 10 x 10, leads to (16, 10, 10) which using the same `MaxPool2d` so we halve ((10-2)/2) + 1 = 5 so (16, 5, 5)
	- Next we flatten with `x = torch.flatten(x, 1)`, x flattened at 1 dimension, so multiply it together. 16x5x5 = 400. So the dimensions needs to be adjusted from 600 to 400
- New Error: `TypeError: wandb.log must be passed a dictionary.` `line 18, in <module> run.log('train_loss', current_loss / (i + 1))` so that's the issue line right there.
		- Reviewing this line, the log is supposed to be entered as a dictionary, but instead it's being passed as a set. Looking more closely at this, we can optimize this by creating a single log at the end of the epoch that accounts for the training and validation, both the loss and the accuracy. We also need to add accuracy tracking to the run, so we'll do that here. 
		- Added to both training and validation loops, then replaced run.log with a single dictionary at the end of the epoch.
- Ran test again. Completes successfully, but returns no training and accuracy data.
- Reviewing from the top again, there's more problems in the code. In Cell 4, in the transforms.Compose, the normalize should have mean and standard deviation defined. Looks like `transforms.Normalize((0.5,), (0.5,)),` right now but should instead look like `transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))`. Also, while `configured_transform` is defined, it's not implemented in the all_images dataset. Correcting both of these
- Additionally, ordered_by_label is set as sorted. This though will cause the images to be ordered by class, and when it is split at 40,000, it will leave a majority of the classes in the training data but only the last few in the validation data. Setting this to random_split
- Correcting that and we're now getting some movement on the training accuracy, but the training loss and validation are both broken still. 
- Ok I think I found it. It's in cell 13. the learning rate is set to 1000. Need to reduce this down. If I recall we'd want 1e-3 for this sort of image dataset being so small.
- Much better movement for this now, although the accuracy and loss seems to be going in the opposite directions.
	- Found it. We're not clearing out the gradient in the training loop. And we have forgotten to turn off the gradient in the validation loop.  Changing that.
- Ok, this is pretty much close to the most accurate given the constraint of not changing the code up too aggressively. A few more quality of life fixes to apply here
	- For the accumulated loss in both training and validation, need to add `.item()` to ensure we only get the values and don't pass along the gradient
	- Created a seed generator and added it to the random shuffling of the dataset to standardize the shuffle across all training instances

## Finding
Several errors were present in the code, ranging from incorrect math calculations in the `CifarClassifier` initialization, incorrect method calls, failing to connect transformer to dataset, sorting dataset before split rather than shuffling, incorrect optimizer and learning rates, missing `zero_grad()` to clear the gradients, missing `no_grad()` for the validation loop, and incorrect usage of set vs. log in run.log. Correcting all of these gets training working in near target accuracy without introducing cop and flip transformations to dataset which could drastically alter the current code outside of customer's target parameters

## Response to Customer
"Hello Jordan,

Thank you for contacting support. My name is Dereck and I will be assisting you today. I understand that you are getting some errors when running your training loop. Don't worry, I am here to help you get your code up and running.

I took a look through the code that you sent me and found a few issues that I corrected. The following issues were causing errors:
- The initial error, `AttributeError: module 'torch.nn' has no attribute 'MaxPooling2D'`, was caused by using the wrong module name. The correct module name is `MaxPool2d`, with a lower case d at the end. I found a similar error with `torch.Flattening`, so I corrected that to `torch.flatten`
- There were additional errors caused by incorrect dimensions in your self.hidden_layer, resulting in `RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x400 and 600x120)`. I adjusted 600 down to 400 after following the math through the forward step, as well as adjusted the narrow_layer's nodes up to 80 due to it being too small given the 10 classes in CIFAR-10
- Another error appeared after that, `TypeError: wandb.log must be passed a dictionary.`. Your initial run.log occured after both the training and validation, but it used a set instead of a dictionary. I reduced this to a single run.log at the end and placed them in a combined dictionary

In addition to fixing the errors, I made some changes to optimize the performance of your training so it works accurately:
- In the data preparation, I hooked up `configured_transform` to your `all_images` dataset under transorm. I also switched your data from a sorted dataset to a shuffled with a set seed, so that the results will be reproducible and to ensure even distribution of the training classes across both your training and validation =data sets
- In your optimizer declaration, the initial learning rate was 1000. I changed the optimizer to use Adam optimization and set the learning rate to 1e-3
- I added accuracy tracking to both the training and validation loops. I also modified your validation to remove the gradient, as it's not tracked during validation. In additiona, I noticed that `optimizer.zero_grad()` was not in the training loop, so I added that at the beginning to clear the gradient so it does not compound the gradient through the training loop.

Give that a go and see if that works for you. From my tests I wa sseeing about 65% training accuracy.

Sincerely,
Dereck Goolsby-Bearsong"

## Customer Response
> "That's great to hear! I just got the updated code you sent and it's working much better now."

## Limits of this project
This is a fictional demonstration of support investigation and communication. It does not establish production AI expertise, validate a real product defect, or claim customer or business impact. The simulated statuses and findings were created for the case study.