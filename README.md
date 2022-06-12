# Sound event detection
### 环境配置
并没有用到额外的包，只需要将该任务原本配好的环境重命名为common就行

或者运行如下脚本
```bash
conda env create -f environment.yml
conda activate common
```
### 特征提取
#### 提取n_mels=64的特征：
```bash
cd data;
sbatch prepare_data.sh
cd ..;
```

#### 提取n_mels=256的特征：
   首先将extract_feature.py中的
```bash
python extract_feature.py "dev/wav.csv" "dev/feature.h5" --sr 44100 --num_worker 1
```
改为
```bash
python extract_feature.py "dev/wav.csv" "dev/feature.h5" --sr 44100 --num_worker 1 --n_mels 256
```
然后
```bash
cd data_new;
sbatch prepare_data.sh
cd ..;
```

### 运行Baseline:
```bash
sbatch run_baseline.sh
```
代码实现在./models_1dpool/models_baseline.py中

### 复现最佳结果:
#### n_mels=64
当使用n_mels=64的数据时（Baseline数据，/data），复现最佳结果
```bash
sbatch run_1dpool_nmels_64.sh
```
代码实现在./models_1dpool/try1.py中
#### n_mels=256
当使用n_mels=256的数据时（Baseline数据，/data_new），复现最佳结果
```bash
sbatch run_1dpool_nmels_256.sh
```
代码实现在./models_1dpool/try7.py中

./models_1dpool中放着所有的模型，在__init__实现了load_model函数，并在run.py中调用load_model函数实现模型读取

./models_1dpool/try1.py和./models_1dpool/try7.py中分别是n_mels=64/256时的最优模型
