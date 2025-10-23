#!/usr/bin/env python3
"""
修复 pymatgen 版本兼容性问题的脚本
问题：新版本的 pymatgen 中，SpacegroupAnalyzer._space_group_data 是字典而不是对象
"""

import os
import sys
import shutil
from pathlib import Path

def backup_original_file(file_path):
    """备份原始文件"""
    backup_path = f"{file_path}.backup"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"已备份原始文件到: {backup_path}")
    return backup_path

def fix_spacegroup_analyzer():
    """修复 SpacegroupAnalyzer 的兼容性问题"""
    
    # 找到 pymatgen 安装路径
    try:
        import pymatgen
        import pymatgen.symmetry.analyzer
        
        # 获取 analyzer 模块的文件路径
        analyzer_module = pymatgen.symmetry.analyzer
        analyzer_file = Path(analyzer_module.__file__)
        
        if not analyzer_file.exists():
            print(f"找不到文件: {analyzer_file}")
            return False
            
        print(f"找到 pymatgen 安装路径: {analyzer_file.parent}")
        print(f"修复文件: {analyzer_file}")
        
        # 备份原始文件
        backup_original_file(analyzer_file)
        
        # 读取文件内容
        with open(analyzer_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经修复过
        if "def get_space_group_symbol(self):" in content and "self._space_group_data['international']" in content:
            print("文件已经修复过，跳过")
            return True
        
        # 修复 get_space_group_symbol 方法
        old_method = """    def get_space_group_symbol(self) -> str:
        \"\"\"Get the spacegroup symbol (e.g., Pnma) for structure.

        Returns:
            str: Spacegroup symbol for structure.
        \"\"\"
        return self._space_group_data.international"""
        
        new_method = """    def get_space_group_symbol(self) -> str:
        \"\"\"Get the spacegroup symbol (e.g., Pnma) for structure.

        Returns:
            str: Spacegroup symbol for structure.
        \"\"\"
        # 兼容性修复：处理字典和对象两种格式
        if isinstance(self._space_group_data, dict):
            return self._space_group_data['international']
        else:
            return self._space_group_data.international"""
        
        if old_method in content:
            content = content.replace(old_method, new_method)
            print("已修复 get_space_group_symbol 方法")
        else:
            # 尝试更简单的替换
            old_line = "return self._space_group_data.international"
            new_line = """# 兼容性修复：处理字典和对象两种格式
        if isinstance(self._space_group_data, dict):
            return self._space_group_data['international']
        else:
            return self._space_group_data.international"""
            
            if old_line in content:
                content = content.replace(old_line, new_line)
                print("已修复 get_space_group_symbol 方法（简化版本）")
            else:
                print("未找到需要修复的方法，可能版本不同")
                return False
        
        # 写回文件
        with open(analyzer_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("修复完成！")
        return True
        
    except Exception as e:
        print(f"修复过程中出现错误: {e}")
        return False

def test_fix():
    """测试修复是否成功"""
    print("\n测试修复结果...")
    try:
        import warnings
        warnings.filterwarnings('ignore')
        from mofchecker import MOFChecker
        from pymatgen.core import Structure
        import numpy as np
        
        # 创建测试结构
        lattice = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        species = ['C', 'C', 'C', 'C']
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        
        structure = Structure(lattice=lattice, species=species, coords=coords, coords_are_cartesian=True)
        checker = MOFChecker(structure)
        desc = checker.get_mof_descriptors()
        
        print("✅ 修复成功！MOFChecker 现在可以正常工作")
        print(f"获取到 {len(desc)} 个描述符")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    print("开始修复 pymatgen 兼容性问题...")
    
    if fix_spacegroup_analyzer():
        if test_fix():
            print("\n🎉 修复完成！现在可以正常运行 python preprocess.py")
        else:
            print("\n⚠️  修复可能不完整，请检查错误信息")
    else:
        print("\n❌ 修复失败")

if __name__ == "__main__":
    main()
