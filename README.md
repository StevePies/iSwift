# iSwift: Fast Impact Identification and Traffic Switching for Large-scale CDNs

## Dataset
The datasets are put in the folder iswift/input/. There are totally 60 folders under the input folder, where each folder contains a dataset with different anomaly detection accuracies. 

Specifically, the files in data0-9 have the same false negative rate and false positive rate 0. The files in data10-19 have the same false negative rate and false positive rate 0.1. Similar rules are applied for later files. Data50-59 have the same false negative rate and false positive rate 0.5.

Detailed parameter settings are list in the configuration file “config.yaml” and algorithm source code “iSwift.py”.

The file “PROOF APPENDIX.pdf” contains the detailed proof of Section 3.1.2.
  
## Usage
  cd iswift;
  
  python iswift.py
  


