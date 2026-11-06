"""淘宝 UserBehavior 购买转化归因分析。

分层：
    config      配置与常量（路径可用环境变量覆盖）
    analysis    纯计算：清洗、漏斗、归因指标（不写文件，可直接被测试调用）
    pipeline    编排：读数据 → analysis → 落盘 outputs/
    synthetic   合成数据生成（schema 对齐公开 UserBehavior 数据集）
    figures     读 outputs/ 画图
    report      Markdown → 单文件 HTML
"""

__version__ = "1.1.0"
