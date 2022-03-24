NLPCC 2022 Shared Task 3 - Multimodal Product Summarization Demo
====
m-kplug is an extension of the pre-trained model [k-plug](https://github.com/xu-song/k-plug), which simply injects the visual signals to the decoder layer.  

## Dependency

```sh
pip install torch==1.6.0 torchvision==0.7.0
pip install psutil transformers==4.12.5
cd fairseq-mirror
pip install -e ./
```

## Command

- train

```
cd m_kplug 
bash run_train.sh
```

- predict

```
cd m_kplug 
bash run_predict.sh
```

- eval(rouge-1/2/L)

```
cd m_kplug 
bash run_eval.sh
```

## Data Process
### raw text preprocessing
```
user_tag=liweikang
cd m_kplug/raw_data_process
python get_data_raw_from_json.py /data/${user_tag}/jdsum/home_appliances/home_appliances_dev_new.json /data/${user_tag}/jdsum/home_appliances/raw_demo/ valid
```
### image vector extracting
```
user_tag=liweikang
cd m_kplug/data_process 
python extract_patch_vector.py /data/${user_tag}/jdsum/home_appliances/raw_demo/valid.sku /data/${user_tag}/jdsum/home_appliances/img /data/${user_tag}/jdsum/home_appliances/raw_demo/ valid
```

### text bpe and bin processing
``` 
cd m_kplug 
bash preprocess_pipline.sh
```
