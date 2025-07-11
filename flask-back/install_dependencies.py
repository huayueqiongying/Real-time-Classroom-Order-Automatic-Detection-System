#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动安装依赖脚本
支持多个国内镜像源
"""

import subprocess
import sys
import os

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n🔧 {description}")
    print(f"执行命令: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 成功")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ 失败")
            if result.stderr:
                print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        return False

def test_mirror(mirror_url, mirror_name):
    """测试镜像源是否可用"""
    print(f"\n🔍 测试 {mirror_name} 镜像源...")
    try:
        result = subprocess.run(f"pip search flask -i {mirror_url}", 
                              shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ {mirror_name} 可用")
            return True
        else:
            print(f"❌ {mirror_name} 不可用")
            return False
    except:
        print(f"❌ {mirror_name} 连接超时")
        return False

def install_dependencies():
    """安装依赖"""
    print("🚀 开始安装依赖...")
    print("=" * 60)
    
    # 镜像源列表
    mirrors = [
        ("https://mirrors.aliyun.com/pypi/simple/", "阿里云"),
        ("https://pypi.douban.com/simple/", "豆瓣"),
        ("https://pypi.mirrors.ustc.edu.cn/simple/", "中科大"),
        ("https://repo.huaweicloud.com/repository/pypi/simple/", "华为云")
    ]
    
    # 测试镜像源
    available_mirrors = []
    for mirror_url, mirror_name in mirrors:
        if test_mirror(mirror_url, mirror_name):
            available_mirrors.append((mirror_url, mirror_name))
    
    if not available_mirrors:
        print("\n❌ 所有镜像源都不可用，尝试使用官方源...")
        available_mirrors = [("", "官方源")]
    
    # 选择最佳镜像源
    best_mirror_url, best_mirror_name = available_mirrors[0]
    print(f"\n🎯 使用 {best_mirror_name} 镜像源")
    
    # 升级pip
    run_command("python -m pip install --upgrade pip", "升级pip")
    
    # 安装基础依赖
    if best_mirror_url:
        mirror_param = f"-i {best_mirror_url}"
    else:
        mirror_param = ""
    
    # 分步安装依赖
    dependencies = [
        ("Flask", "Flask框架"),
        ("Flask-CORS", "Flask跨域支持"),
        ("Pillow", "图像处理库"),
        ("numpy", "数值计算库"),
        ("opencv-python", "OpenCV计算机视觉库"),
        ("face-recognition", "人脸识别库"),
        ("dlib", "机器学习库"),
        ("onnxruntime", "ONNX推理引擎")
    ]
    
    success_count = 0
    for package, description in dependencies:
        command = f"pip install {package} {mirror_param}"
        if run_command(command, f"安装 {description} ({package})"):
            success_count += 1
        else:
            print(f"⚠️ {package} 安装失败，继续安装其他包...")
    
    print(f"\n📊 安装结果: {success_count}/{len(dependencies)} 个包安装成功")
    
    # 验证关键依赖
    print("\n🔍 验证关键依赖...")
    critical_packages = ["flask", "cv2", "dlib", "face_recognition"]
    missing_packages = []
    
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package} 可用")
        except ImportError:
            print(f"❌ {package} 不可用")
            missing_packages.append(package)
    
    # 尝试安装ONNX Runtime
    try:
        import onnxruntime
        print("✅ onnxruntime 可用")
    except ImportError:
        print("⚠️ onnxruntime 不可用，行为检测功能将不可用")
        print("可以稍后手动安装: pip install onnxruntime")
    
    if missing_packages:
        print(f"\n⚠️ 缺少关键依赖: {', '.join(missing_packages)}")
        print("请手动安装这些包")
    else:
        print("\n🎉 所有关键依赖安装成功！")
    
    return success_count == len(dependencies)

def install_frontend():
    """安装前端依赖"""
    print("\n🌐 安装前端依赖...")
    print("=" * 60)
    
    if not os.path.exists("frontend"):
        print("❌ frontend目录不存在")
        return False
    
    os.chdir("frontend")
    
    # 检查Node.js
    if not run_command("node --version", "检查Node.js"):
        print("❌ Node.js未安装，请先安装Node.js")
        return False
    
    # 检查npm
    if not run_command("npm --version", "检查npm"):
        print("❌ npm未安装")
        return False
    
    # 安装前端依赖
    if run_command("npm install", "安装前端依赖"):
        print("✅ 前端依赖安装成功")
        os.chdir("..")
        return True
    else:
        print("❌ 前端依赖安装失败")
        os.chdir("..")
        return False

def main():
    """主函数"""
    print("🎯 人脸识别系统依赖安装工具")
    print("=" * 60)
    
    # 安装Python依赖
    python_success = install_dependencies()
    
    # 安装前端依赖
    frontend_success = install_frontend()
    
    print("\n" + "=" * 60)
    print("📋 安装总结:")
    print(f"Python依赖: {'✅ 成功' if python_success else '❌ 失败'}")
    print(f"前端依赖: {'✅ 成功' if frontend_success else '❌ 失败'}")
    
    if python_success and frontend_success:
        print("\n🎉 所有依赖安装完成！")
        print("\n🚀 启动系统:")
        print("1. 启动后端: python app.py")
        print("2. 启动前端: cd frontend && npm run dev")
    else:
        print("\n⚠️ 部分依赖安装失败，请检查错误信息并手动安装")
        print("参考 install_onnx.md 文件获取详细安装指南")

if __name__ == "__main__":
    main() 