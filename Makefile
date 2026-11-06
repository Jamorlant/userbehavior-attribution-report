PY      ?= python
DATA    ?=
OUTPUTS ?= outputs
FIGURES ?= figures
REPORT  ?= report
ROWS    ?= 1000000

# DATA 留空时读取 .env / 环境变量里的 UBA_DATA_PATH（真实数据默认走这条）；
# 需要显式指定时：make analyze DATA=/path/to/taobao.parquet
ANALYZE_DATA = $(if $(DATA),--data $(DATA),)

.PHONY: help install data analyze figures report all verify test lint fmt clean

help:
	@echo "make install   安装依赖（运行 + 开发）"
	@echo "make all       重算指标 → 重画图表 → 重建 HTML 报告"
	@echo "make data      生成合成回退数据（seed=42，仅 CI/冒烟用）"
	@echo "make verify    本地 CI：lint + 格式 + 单测"
	@echo "make clean     清理缓存与临时产物"

install:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

data:
	$(PY) -m uba.synthetic --rows $(ROWS) --out data/userbehavior.csv.gz

analyze:
	$(PY) -m uba.pipeline $(ANALYZE_DATA) --out $(OUTPUTS)

figures:
	$(PY) -m uba.figures --out $(OUTPUTS) --figdir $(FIGURES)

report:
	$(PY) -m uba.report --report-dir $(REPORT)

all: analyze figures report

verify: lint fmt test

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

fmt:
	$(PY) -m ruff format --check .

clean:
	$(PY) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache', '.ruff_cache', 'build', 'build-outputs', 'build-figures')]"
