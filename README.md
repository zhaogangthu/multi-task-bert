easytransfer参考下面
https://github.com/wellinxu/EasyTransfer/tree/master/scripts/%E5%A4%A9%E6%B1%A0%E5%A4%A7%E8%B5%9B%E4%B8%93%E5%8C%BA?spm=5176.12282029.0.0.3be57d025WOYrN

# multi-task-bert
NLP中文预训练模型泛化能力挑战赛

1. python generate_train_data2.py
   生成训练集和测试集（测试集指定一个假标签）
2. ./run_convert_csv_to_tfrecords.sh
3. python multitask_finetune.py
4. python test.py
   生成提交版json文件
