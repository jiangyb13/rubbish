### WEb 前端使用方法
#### 数据准备
假设当前目录为/root
Step 1：解压代码：将flask.rar解压到/root目录下
Step 2: 在/root目录下创建outputs文件夹。复制原始数据，目前我们准备的数据在bucket-8713-huanan ＞ code ＞ jwx1416454 ＞ HUAWEI_CrossPairDataset ＞ outputs ＞ one_shot_process_huawei_final，将数据复制到outputs文件夹下。

#### 运行代码

##### 文件夹格式
|--root
|----flask
|----outputs
|------one_shot_process_huawei_final

##### 运行命令
python flask/app.py --data-dir outputs/one_shot_process_huawei_final --port 7860

####注意事项
由于目前s3的磁盘解压比较慢，目前只解压出了phase0和phase2，还有3个phase正在解压当中，目前能看的只有phase0和phase2。
