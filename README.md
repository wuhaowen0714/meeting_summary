安装依赖
```bash
pip3 install -r requirement.txt
```

下载模型
```bash
./scripts/download.sh
```

运行程序
```bash
python3 main.py --bmodel ./models/qwen1.5-7b_int4_6k_1dev.bmodel --dev_id 0 --text_path ./meeting.txt
```

参数说明
```bash
- bmodel          bmodel文件路径
- dev_id          芯片id
- text_path       会议文本路径
```