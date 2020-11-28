# multi-task-bert
NLP中文预训练模型泛化能力挑战赛

1. python generate_train_data2.py
   生成训练集和测试集（测试集指定一个假标签）
2. ./run_convert_csv_to_tfrecords.sh
3. python multitask_finetune.py
4. python test.py
   生成提交版json文件
