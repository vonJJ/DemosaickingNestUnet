### A short demo code for research on Bayer pattern image demosaicking 
### (paper "A Compact High-Quality Image Demosaicking Neural Net-work for Edge Computing Devices" being previewed by MDPI)


- Usage:
    - Run `python train.py` to train a new model. The default saving file is `./TrainingResult`
    - Run `python eval.py` to test running time and psnr value of chosen dataset. The PSNR value of each level will saved in ` ./TestingResult/YOUR TESTING SETS NAME/`
    - For testing, you may change the `crop` and `ignore` value to set the padding value and the ignored pixel number of each edge when calculating PSNR value (Default is both 0).
    - Settings should be changed in config.py


- Dataset:
    - We have two prepared demo datesets Kodak24 and McMaster
    - For Training, you should prepare on your own
    - For evaluating, you can use our prepared one and also use your own one(it should be arranged in the manner as below)
    - If using your own datasets, donnot forget to change settings (filepath, etc.) in config.py
    
- The folder structure of the data folder should be:

``` DataSets File Arrangement:

DMUnet++  
|
└── Datasets  
    ├── Training  
    |   └── YOUR TRAINING SETS  
    |       └── train 
    |           └── img00001.jpg (example) 
    |           └── img00002.jpg (example) 
    |           └── img00003.jpg (example) 
    |           └── ...
    └── Evaluating  
        ├── Kodak24
        |   └── test 
        |       └── kodim01.png 
        |       └── kodim02.png 
        |       └── kodim03.png
        |       └── ...
        ├── McMaster
        |   └── test 
        |       └── 1.tif 
        |       └── 2.tif 
        |       └── 3.tif
        |       └── ...
        |
        └── YOUR TESTING SETS
            └── test 
                └── img01.png 
                └── img02.png 
                └── img03.png
                └── ...

```


