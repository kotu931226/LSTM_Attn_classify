# LSTM_Attn_classify
## summary
自作記号'T'をLSTMとself-Attentionを使って学習させる

## 使いかた
pytorch, numpy, tqdmをインストール(pytorchは公式からdounload) 
```
pip install numpy, tqdm
```
バージョンは  
pytorch = 0.4.1  
numpy = 1.15.4  
tqdm = 4.28.1  

config.pyの設定を見つつ  
gen_arithmetic.py, train.py を実行  
n_epoch=100でも時間が結構かかる  
収束するのは1000以上？  
```
python gen_arithmetic.py
```
```
python train.py
```

最後にtest.pyで各accuracyを表示  
```
python test.py
```

n_epoch = 100での結果は  

Accuracy of all : 0.4322916666666667  
Accuracy of 1 : 1.0  
Accuracy of 2 : 0.39622641509433965  
Accuracy of 3 : 0.04081632653061224  
Accuracy of 4 : 0.25  
 
となった 
