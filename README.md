### ComfyUI-PainterFluxImageEdit


## Flux2 文生图 & 图生图编辑一体化节点，适合9B，4B的FLUX2文生图and图片编辑工作流。


###  文生图：
<img width="2138" height="889" alt="image" src="https://github.com/user-attachments/assets/0c1fbfcf-4deb-4ff1-8f3e-55d714fe0dd1" />

###  单图编辑：
<img width="2608" height="1220" alt="AA%NPS$_SCQKAUZ)QD$%JDI" src="https://github.com/user-attachments/assets/0832e665-93b1-4568-91fd-a685e99f99d4" />

###  多图编辑：
<img width="2789" height="1266" alt="A%GJ96GX_9NJF5)HJZF9CN6" src="https://github.com/user-attachments/assets/b15e6d4c-b3b3-4162-885d-3c834d59c564" />



## 功能特点

- **三合一简化**：将 CLIPTextEncode、VAEEncode、ReferenceLatent 三个节点整合为单节点
- **双模式支持**：无输入图片时为纯文生图，接入图片自动切换为图生图编辑
- **多图编辑**：最多支持 3 张参考图同时编辑，自动编码为 reference latents
- **智能遮罩**：首图支持 mask 遮罩，实现精确区域重绘
- **分辨率直设**：直接在节点内指定输出图片尺寸

## 安装

通过 ComfyUI-Manager 搜索 `ComfyUI-PainterFluxImageEdit` 安装，或手动克隆到 `custom_nodes` 目录：

```
git clone https://github.com/princepainter/Comfyui-PainterFluxImageEdit.git
```

## 使用

在工作流中替换原有的文本编码+VAE编码+参考潜空间三节点组合，直接连接 Flux2 采样器即可。

## 注意事项

- 需要配合 Flux2 模型使用
- mask 仅对第一张输入图片生效
- 图片尺寸建议为 8 的倍数

---
