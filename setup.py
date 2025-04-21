import os
import urllib.request
from setuptools import setup,find_packages
from setuptools.command.install import install

class CustomInstall(install):
    """直接从GitHub下载所需文件"""
    def run(self):
        # 目标目录路径
        target_dir = os.path.join("src", "janc", "amr")
        os.makedirs(target_dir, exist_ok=True)

        # 文件下载配置（替换为实际URL）
        files = {
            "amraux.py": "https://raw.githubusercontent.com/luofx23/JAX-AMR/src/amr/amraux.py",
            "jaxamr.py": "https://raw.githubusercontent.com/luofx23/JAX-AMR/src/amr/jaxamr.py"
        }

        # 下载文件
        for filename, url in files.items():
            try:
                urllib.request.urlretrieve(url, os.path.join(target_dir, filename))
                print(f"✅ 下载成功: {filename}")
            except Exception as e:
                raise RuntimeError(f"文件下载失败: {url}\n错误信息: {str(e)}")

        super().run()

setup(
    name="janc",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmdclass={"install": CustomInstall},
    install_requires=['cantera'],  # 其他依赖可在此声明
)