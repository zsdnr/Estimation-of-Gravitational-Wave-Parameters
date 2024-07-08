# import jax
# # import jaxlib
#
# print(jax.__version__)
# # print(jaxlib.__version__)
#

import sys
import pkg_resources

# 获取 ripple 库的安装路径
try:
    ripple_dist = pkg_resources.get_distribution("ripple")
    ripple_path = ripple_dist.location
    print(f"ripple 库安装路径: {ripple_path}")

    # 确保安装路径在 sys.path 中
    if ripple_path not in sys.path:
        sys.path.append(ripple_path)
        print(f"已将 {ripple_path} 添加到 sys.path")
except pkg_resources.DistributionNotFound:
    print("ripple 库未找到，请确保已安装")

# 导入模块
try:
    from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
    print("模块导入成功")
except ImportError as e:
    print(f"导入失败: {e}")
