# RATE: Overcoming Noise and Sparsity of Textual Features in Real-Time Location Estimation

This repository contains the source code for [**RATE: Overcoming Noise and Sparsity of Textual Features in Real-Time Location Estimation**](https://arxiv.org/abs/2111.06515).

## Code
To run our code, please use the following commands:
```
g++ RATE.cpp -o RATE -std=c++11
./RATE [Training File] [Test File] [L, optional, default = 30] [T, optional, default = 1]
```
For example,
```
g++ RATE.cpp -o RATE -std=c++11
./RATE Dataset/train.txt Dataset/test.txt 40 1
```
The prediction results will be in ```./result.txt``` (the first row is the classification result). Then you can run
```
python eval.py
```
to obtain evaluation metrics.

## Dataset
We release the Europe dataset (```Dataset/data.json```), where each line is a json file with tweet text and metadata. Due to privacy issues, we have anonymized the whole dataset by representing each word/feature as an integer. An example is shown below.
```
{ 
   "label":0,
   "language":"3",
   "timezone":"5",
   "offset":"7",
   "userlang":"5",
   "latitude":"36.8901",
   "longitude":"30.6809",
   "text":"3332 2608 29"
}
```
Given the json file, one can run 
```
cd Dataset/
python preprocess.py
```
to get training and testing data (```Dataset/train.txt``` and ```Dataset/test.txt```).

## Result
| Method | Micro-F1 (Acc) | Macro-F1 | Mean Distance Error (km) | Acc@161 |
| ------ | -------------- | -------- | ------------------------ | ------- |
| RATE   | 0.8905         | 0.5230   | 365.16                   | 0.4315  |

## Citation
```
@inproceedings{zhang2017rate,
  title={RATE: Overcoming Noise and Sparsity of Textual Features in Real-Time Location Estimation},
  author={Zhang, Yu and Wei, Wei and Huang, Binxuan and Carley, Kathleen M and Zhang, Yan},
  booktitle={Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},
  pages={2423--2426},
  year={2017},
  organization={ACM}
}
```
