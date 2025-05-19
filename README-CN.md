[中文](README-CN.md) | [English](README.md) 

# ComfyUI 的 MegaTTS3 声音克隆节点

声音克隆质量非常高, 支持中英文, 并可跨语言克隆.

![image](https://github.com/billwuhao/ComfyUI_MegaTTS3/blob/main/images/2025-04-06_13-52-57.png)

## 📣 更新

[2025-04-28]⚒️: 新增预览音色节点, 先预览音色, 满意再进行克隆. 感谢 @chenpipi0807 的 idea😍. 可在 `speakers` 文件夹下分门别类建更多文件夹.

[2025-04-06]⚒️: 发布 v1.0.0.

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_MegaTTS3.git
cd ComfyUI_MegaTTS3
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

- 模型和音色需要手动下载放到 `ComfyUI\models\TTS` 路径下:

[MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3/tree/main)  整个文件夹全部下载放到 `TTS` 文件夹下.

`MegaTTS3` 文件夹中新建 `speakers` 文件夹, 从 [Google drive](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr) 下载所有 `.wav` 和 `.npy` 文件, 放到 `speakers` 文件夹下.

![image](https://github.com/billwuhao/ComfyUI_MegaTTS3/blob/main/images/2025-04-06_14-49-12.png)

## 鸣谢

- [MegaTTS3](https://github.com/bytedance/MegaTTS3)