# HWRS640_HW4 Created By: Roz Roberts IV
The HWRS640_HW4 Repo implements an LSTM model on observed streamflow from the CAMELS database by using MiniCamels.

The LSTM model is implemented through the Python library CLICK as a CLI interface system.

The different flags for this CLI are listed bellow and what they do, as well as an example snippet showcasing how
to use this library.

The CLI is commanded from main.py, to run this code in terminal use the following:
```terminaloutput
python source/main.py 
```

To see how each command is used just pass the "help" flag:
```terminaloutput
python source/main.py --help
```
This will show the following:
```
Usage: main.py [OPTIONS] COMMAND [ARGS]...

  Main Command Line Interface for HWRS640 HW 4

Options:
  --help  Show this message and exit.

Commands:
  evaluate        Evaluate the model
  plot            Plot a given file, if none given plot best LSTM results
  summarize-data  Summary of dataset
  train           Train the LSTM model with given parameters
```

Although the commands are designed to be straight forward to understand, "help" flags like the 
one above have been provided for every command. Some of the "help" information the flags 
provide is listed bellow for convince of the user. If at any point you are confused about
what a flag or option does just pass ```--help``` for a given command and read the output.

---

To start summarize-data gives a summary of the full dataset.
This summary is given as a graphical and command line output.
The graphical output is the default and is saved to the outputs folder and is called summary_plots.png
The in-line results are gotten by passing the "structure" flag and give the full datasets structure. 

```terminaloutput
python source/main.py summarize-data
```
It should be noted that for this LSTM the static attributes are ignored. 

---

```terminaloutput
python source\main.py plot
```
Provides many avenues to plot data from the LSTM. Passing the "explore" flag generates the 
exploratory plots from Question 1 of this homework. The "history-plot" flag plots the last,
or a specified training history.

--- 

```terminaloutput
python source\main.py train
python source\main.py evaluate
```
Are the two most curcial commands. The first one trains an LSTM model on the dataset with many
parameters to customize. The "train" command also has a "just" flag which gives the traing/validation/testing
dataset split justification as requested in Question 2 of the homework. All other CLI options are 
for hyperparamter tuning and data tuning. 
```terminaloutput
python source\main.py train --just
```

The second command shown is "evaluate" this evaluates the trained model using the evaluation testing.
This command **_requires_** that you as the user remember the hyperparameters used in "train" as unless
you are using the **_default parameters_** you will need to specify them again. In addition, you will also need
to specify where the checkpoint path (saved .pt file) is. See the example bellow to reproduce the "best LSTM"
to see how this is done. 

---

To reproduce the best LSTM model that I was able to train use the following commands for training and evaluation:

```terminaloutput
source/main.py train --epochs 80 --nse-interval 1 --batch-size 64 --hidden-size 64 --num-layers 2 --seed 117

python source/main.py evaluate --checkpoint-path outputs/training_results/best_lstm.pt --batch-size 64 --hidden-size 64 --num-layers 2 --seed 117      
```

---

All of the above information is provided in the various "help" flags of the CLI, so it
should be very simple to navigate and use. The total PDF writeup for this homework is provided under
```outputs/writeup```.
