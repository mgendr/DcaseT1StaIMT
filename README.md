# dcase-2020-task1-subtaskA

This repository includes our metadata and code for the submission of IMT Atlantique - BRAIN to the DCASE 2020 challenge, Task 1, subtask A. 
http://dcase.community/challenge2021/task-acoustic-scene-classification


This github takes the basis of the project code https://github.com/brain-bzh/dcase-2020-task1-subtaskB . Here we try to adapt and deepen the work done to a different task of the challenge.



Code is based on pytorch (1.5), sklearn, pandas, numpy, scipy.

### 1. Resample audio at 18 kHz

Dataset from TAU Urban Acoustic Scenes 2020 10Class can be download at https://zenodo.org/record/3819968#.YHMeVOgzZPY

To resample run,
```
python resample.py --input_path [path dataset] --output_path [save path]
```

Note: The output path to the audio dataset has to be specified in create_dataset.py before running the next scripts.

### 2. Train a model

Launch training
```
python main_training.py --saving [0 (no), 1 (yes)] --model_type ["ModelA", "ModelB", "ModelC", "ModelD"] --lr [learning rate (float)] --epochs=20 --batch_size=64 --da_mode ['cutmix', 'mixup', 'random_crop'] 
```

Exemple with ModelB and 5% of the dataset without Data Augmentation (Quick check)
```
python main_training.py --saving=1 --model_type="ModelB" --lr=1e-3 --epochs=3 --batch_size=32 --frac_data=0.05
```

Note: Model filename can be found in 'models/' after training with --saving=1

### 3. Prune and quantify model

Prune model
```
python pruning_torch.py --model_filename [Model filename] --da_mode ['cutmix', 'mixup', 'random_crop'] 
```

Quantify model
```
python quantify_torch.py --model_filename [Model filename]
```

### 4. Get statistics and create submission

Get statistics of the model
```
python model_statistics.py --model_filename [Model filename]
```

Create submission csv file
```
python create_submission.py --model_filename [Model filename] --eval_csv_path [Path to the evaluation csv file] --eval_data_path [Path to evaluation audio directory]
```
