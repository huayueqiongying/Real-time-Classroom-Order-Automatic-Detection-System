# ONNX Runtime 安装指南

## 问题说明

`onnxruntime` 包名在不同情况下可能不同，需要根据您的系统选择合适的版本。

## 安装方法

### 方法1：CPU版本（推荐）

```bash
# 阿里云镜像源
pip install onnxruntime -i https://mirrors.aliyun.com/pypi/simple/

# 豆瓣镜像源
pip install onnxruntime -i https://pypi.douban.com/simple/

# 中科大镜像源
pip install onnxruntime -i https://pypi.mirrors.ustc.edu.cn/simple/

# 华为云镜像源
pip install onnxruntime -i https://repo.huaweicloud.com/repository/pypi/simple/
```

### 方法2：GPU版本（如果有NVIDIA GPU）

```bash
# 阿里云镜像源
pip install onnxruntime-gpu -i https://mirrors.aliyun.com/pypi/simple/

# 豆瓣镜像源
pip install onnxruntime-gpu -i https://pypi.douban.com/simple/

# 中科大镜像源
pip install onnxruntime-gpu -i https://pypi.mirrors.ustc.edu.cn/simple/
```

### 方法3：指定版本安装

```bash
# CPU版本
pip install onnxruntime==1.16.3 -i https://mirrors.aliyun.com/pypi/simple/

# GPU版本
pip install onnxruntime-gpu==1.16.3 -i https://mirrors.aliyun.com/pypi/simple/
```

### 方法4：使用conda安装

```bash
conda install -c conda-forge onnxruntime
```

## 验证安装

安装完成后，运行以下命令验证：

```bash
python -c "import onnxruntime; print('ONNX Runtime版本:', onnxruntime.__version__)"
```

## 常见问题

### 1. 找不到包
如果遇到"找不到包"的错误，尝试：
```bash
pip install --upgrade pip
pip install onnxruntime --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/
```

### 2. 版本冲突
如果遇到版本冲突，可以：
```bash
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime
```

### 3. Windows特定问题
在Windows上，可能需要：
```bash
pip install onnxruntime --only-binary=all -i https://mirrors.aliyun.com/pypi/simple/
```

## 备选方案

如果无法安装ONNX Runtime，系统仍然可以正常运行，只是行为检测功能不可用：

1. 人脸识别功能正常
2. 人脸注册功能正常
3. 行为检测功能会显示"功能不可用"提示

## 系统启动

安装完成后，正常启动系统：

```bash
# 后端
python app.py

# 前端
cd frontend
npm run dev
```

系统会自动检测ONNX Runtime是否可用，并在控制台显示相应信息。 